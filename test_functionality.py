"""
Script para probar el dataset musk1 en todos los modelos disponibles de MIClustering,
incluyendo aceleración por GPU (CUDA / Apple Silicon MPS), paralelismo multinúcleo
y validación de la API estándar de Scikit-Learn.

Modelos evaluados:
  1. MIDBSCAN   - Clustering basado en densidad adaptado a Multi-Instance Learning
  2. MIKMeans   - Clustering basado en centroides (bolsas agregadas) para MIL
  3. MIKMedoids - Clustering basado en medoides (PAM) para MIL
  4. MIKnn      - Clasificador k-Nearest Neighbors supervisado para MIL
  5. COSMIC     - Clustering jerárquico basado en densidad (OPTICS) para MIL
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from contextlib import contextmanager
import numpy as np
import pandas as pd
from tqdm import tqdm

# 1. Asegurar acceso al paquete miclustering desde lib-code
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = REPO_ROOT / "lib-code" / "MIClustering" / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from miclustering.data.arff_reader import ArffToMIData
from miclustering.preprocessing.scaler import MinMaxScaler
from miclustering.evaluation.bcm import MILEvaluator
from miclustering.models.midbscan import MIDBSCAN
from miclustering.models.mikmeans import MIKMeans
from miclustering.models.mikmedoids import MIKMedoids
from miclustering.models.miknn import MIKnn
from miclustering.models.cosmic import COSMIC
from miclustering.distances.distance_matrix import compute_distance_matrix
from miclustering.distances import DISTANCE_REGISTRY
from miclustering.distances.torch_backend import is_torch_available, get_torch_device


# ─── Barra de progreso por modelo ───────────────────────────────────────
PROGRESS_BAR_FORMAT = (
    "{desc}: {bar} {n_fmt}/{total_fmt} "
    "[{elapsed}<{remaining}]"
)

@contextmanager
def model_progress(model_name: str, steps: list[str] | None = None):
    """Context manager que muestra una barra de progreso para la ejecución de un modelo.
    
    Devuelve una función `advance(step_name)` que avanza la barra al siguiente paso.
    """
    steps = steps or ["Fitting", "Predicting", "Evaluating"]
    bar = tqdm(
        total=len(steps),
        desc=f"  [{model_name}]",
        bar_format=PROGRESS_BAR_FORMAT,
        ncols=75,
        leave=True,
        colour="#00d4aa",
    )
    
    def advance(step_name: str):
        bar.set_description(f"  >>> {model_name} → {step_name}")
        bar.update(1)
        bar.refresh()
    
    try:
        yield advance
    finally:
        bar.set_description(f"  <<< {model_name}")
        bar.refresh()
        bar.close()


def resolve_dataset_path(dataset_filename: str = "musk1.arff") -> Path:
    """Busca el archivo ARFF del dataset en el árbol del proyecto."""
    candidates = [
        REPO_ROOT / "MIClustering-experiments" / "data" / dataset_filename,
        REPO_ROOT / "experiments" / "data" / dataset_filename,
        REPO_ROOT / "data" / dataset_filename,
        REPO_ROOT / "lib-code" / "MIClustering" / "tests" / "data" / dataset_filename,
        Path(dataset_filename),
    ]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"No se encontró el archivo '{dataset_filename}' en las rutas habituales.")


def benchmark_hardware_acceleration(bags, metric_name: str = "hausdorff_avg"):
    """Compara los tiempos de cálculo de la matriz de distancias en CPU secuencial, CPU multinúcleo y GPU."""
    print("\n" + "=" * 75)
    print("BENCHMARK DE ACELERACION DE HARDWARE (MATRIZ DE DISTANCIAS)")
    print("=" * 75)
    
    metric_func = DISTANCE_REGISTRY[metric_name]
    num_bags = len(bags)
    print(f"Cálculo de matriz {num_bags}x{num_bags} ({(num_bags * (num_bags - 1)) // 2} pares únicos) con '{metric_name}'")
    
    # 1. CPU Secuencial (n_jobs=1)
    t0 = time.perf_counter()
    m_seq = compute_distance_matrix(bags, metric_func, metric_name=metric_name, n_jobs=1, device="cpu")
    t_seq = time.perf_counter() - t0
    print(f"  1. CPU Secuencial (1 core)   : {t_seq * 1000:7.2f} ms (1.00x - Referencia)")

    # 2. CPU Multinúcleo (n_jobs=-1)
    t0 = time.perf_counter()
    m_par = compute_distance_matrix(bags, metric_func, metric_name=metric_name, n_jobs=-1, device="cpu")
    t_par = time.perf_counter() - t0
    speedup_cpu = t_seq / max(t_par, 1e-9)
    print(f"  2. CPU Multinúcleo (joblib)  : {t_par * 1000:7.2f} ms ({speedup_cpu:.2f}x speedup)")

    # 3. GPU / MPS / CUDA Acelerado
    if is_torch_available():
        dev = get_torch_device("auto")
        dev_name = dev.type.upper() if dev is not None else "GPU"
        t0 = time.perf_counter()
        m_gpu = compute_distance_matrix(bags, metric_func, metric_name=metric_name, n_jobs=1, device="auto")
        t_gpu = time.perf_counter() - t0
        speedup_gpu = t_seq / max(t_gpu, 1e-9)
        print(f"  3. GPU Acelerado ({dev_name})    : {t_gpu * 1000:7.2f} ms ({speedup_gpu:.2f}x speedup)")
        
        # Validación de concordancia numérica
        diff = np.max(np.abs(m_seq - m_gpu))
        print(f"  -> Verificación de exactitud numérica (Max Diff CPU vs GPU): {diff:.2e}")
    else:
        print("  3. GPU Acelerado : PyTorch no disponible en el entorno.")
    print("=" * 75)


def run_benchmark(
    dataset_name: str = "musk1.arff",
    train_pct: float = 70.0,
    seed: int = 42,
    metric: str = "hausdorff_avg",
    n_jobs: int = -1,
    device: str = "auto",
    run_hw_bench: bool = True,
) -> pd.DataFrame:
    """Ejecuta y evalúa todos los modelos de MIClustering sobre el dataset especificado."""
    arff_path = resolve_dataset_path(dataset_name)
    
    print("\n" + "=" * 75)
    print(f"BENCHMARK MICLUSTERING — DATASET: {arff_path.name}")
    print(f"Ruta del archivo: {arff_path}")
    print(f"Métrica: {metric} | Paralelismo (n_jobs): {n_jobs} | Dispositivo: {device} | Semilla: {seed}")
    print("=" * 75)

    # 1. Cargar datos
    dataset = ArffToMIData.from_arff(arff_path)
    total_bags = dataset.get_num_bags()
    print(f"Total de bolsas cargadas: {total_bags}")

    # 2. Partición Train / Test
    train_data, test_data = dataset.split_data(percentage_train=train_pct, seed=seed)
    print(f"Partición {train_pct:.0f}% Train / {100 - train_pct:.0f}% Test:")
    print(f"  - Train: {train_data.get_num_bags()} bolsas")
    print(f"  - Test : {test_data.get_num_bags()} bolsas")

    # 3. Escalado (MinMaxScaler)
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    full_scaled = scaler.fit_transform(dataset)

    # Benchmark de aceleración opcional
    if run_hw_bench:
        benchmark_hardware_acceleration(train_scaled.bags, metric_name=metric)

    results = []

    # -------------------------------------------------------------
    # 1. MIDBSCAN
    # -------------------------------------------------------------
    print("\n" + "-" * 75)
    print(">>> 1. MIDBSCAN (Density-Based Clustering)")
    print("-" * 75)
    with model_progress("MIDBSCAN") as advance:
        midbscan = MIDBSCAN(epsilon=2.0, min_pts=2, metric=metric, n_jobs=n_jobs, device=device)
        advance("Fitting")
        midbscan.fit(train_scaled)
        advance("Predicting")
        preds_midbscan = midbscan.predict(test_scaled)
        advance("Evaluating")
        res_midbscan = MILEvaluator.evaluate(test_scaled, preds_midbscan, title="MIDBSCAN (Test)")
    print(f"  -> Propiedad Scikit-Learn (labels_): {midbscan.labels_[:8]}... (len={len(midbscan.labels_)})")
    print(f"  -> Clusters detectados: {midbscan.cluster_count}")
    results.append({
        "Modelo": "MIDBSCAN",
        "Paradigma": "Clustering (Inductivo)",
        "Configuración": f"eps=2.0, min_pts=2, {metric}, dev={midbscan.device}",
        **res_midbscan
    })

    # -------------------------------------------------------------
    # 2. MIKMeans
    # -------------------------------------------------------------
    print("\n" + "-" * 75)
    print(">>> 2. MIKMeans (Centroid-Based Clustering)")
    print("-" * 75)
    with model_progress("MIKMeans") as advance:
        mikmeans = MIKMeans(k=2, metric=metric, random_state=seed, n_jobs=n_jobs, device=device)
        advance("Fitting")
        mikmeans.fit(train_scaled)
        advance("Predicting")
        preds_mikmeans = mikmeans.predict(test_scaled)
        advance("Evaluating")
        res_mikmeans = MILEvaluator.evaluate(test_scaled, preds_mikmeans, title="MIKMeans (Test)")
    print(f"  -> Propiedad Scikit-Learn (labels_): {mikmeans.labels_[:8]}... (len={len(mikmeans.labels_)})")
    print(f"  -> Centroides calculados (cluster_centers_): {len(mikmeans.cluster_centers_)} centroides sintéticos")
    results.append({
        "Modelo": "MIKMeans",
        "Paradigma": "Clustering (Inductivo)",
        "Configuración": f"k=2, {metric}, dev={mikmeans.device}",
        **res_mikmeans
    })

    # -------------------------------------------------------------
    # 3. MIKMedoids
    # -------------------------------------------------------------
    print("\n" + "-" * 75)
    print(">>> 3. MIKMedoids (PAM Medoid-Based Clustering)")
    print("-" * 75)
    with model_progress("MIKMedoids") as advance:
        mikmedoids = MIKMedoids(k=2, metric=metric, random_state=seed, n_jobs=n_jobs, device=device)
        advance("Fitting")
        mikmedoids.fit(train_scaled)
        advance("Predicting")
        preds_mikmedoids = mikmedoids.predict(test_scaled)
        advance("Evaluating")
        res_mikmedoids = MILEvaluator.evaluate(test_scaled, preds_mikmedoids, title="MIKMedoids (Test)")
    print(f"  -> Propiedad Scikit-Learn (labels_): {mikmedoids.labels_[:8]}... (len={len(mikmedoids.labels_)})")
    print(f"  -> Índices de medoides (medoid_indices_): {mikmedoids.medoid_indices_}")
    results.append({
        "Modelo": "MIKMedoids",
        "Paradigma": "Clustering (Inductivo)",
        "Configuración": f"k=2, {metric}, dev={mikmedoids.device}",
        **res_mikmedoids
    })

    # -------------------------------------------------------------
    # 4. MIKnn
    # -------------------------------------------------------------
    print("\n" + "-" * 75)
    print(">>> 4. MIKnn (Supervised k-NN Classifier)")
    print("-" * 75)
    with model_progress("MIKnn") as advance:
        miknn = MIKnn(k=3, metric=metric, n_jobs=n_jobs, device=device)
        advance("Fitting")
        miknn.fit(train_scaled)
        advance("Predicting")
        preds_miknn = miknn.predict(test_scaled)
        advance("Evaluating")
        res_miknn = MILEvaluator.evaluate(test_scaled, preds_miknn, title="MIKnn (Test)")
    print(f"  -> Clases conocidas (classes_): {miknn.classes_}")
    print(f"  -> Propiedad Scikit-Learn (labels_): {miknn.labels_[:8]}... (len={len(miknn.labels_)})")
    results.append({
        "Modelo": "MIKnn",
        "Paradigma": "Supervisado (Inductivo)",
        "Configuración": f"k=3, {metric}, dev={miknn.device}",
        **res_miknn
    })

    # -------------------------------------------------------------
    # 5. COSMIC
    # -------------------------------------------------------------
    print("\n" + "-" * 75)
    print(">>> 5. COSMIC (OPTICS-Based Transductive Clustering)")
    print("-" * 75)
    with model_progress("COSMIC", steps=["Fitting", "Evaluating", "Extracting"]) as advance:
        cosmic = COSMIC(epsilon=1.5, min_pts=2, metric=metric, n_jobs=n_jobs, device=device)
        advance("Fitting")
        cosmic.fit(full_scaled)
        preds_cosmic = cosmic.labels
        advance("Evaluating")
        res_cosmic = MILEvaluator.evaluate(full_scaled, preds_cosmic, title="COSMIC (Dataset Completo)")
        advance("Extracting")
        # Demostración de extracción a diferente granularidad sin reentrenar
        fine_labels = cosmic.extract_clusters(epsilon_prime=0.8)
    print(f"  -> Propiedad Scikit-Learn (labels_): {cosmic.labels_[:8]}... (len={len(cosmic.labels_)})")
    print(f"  -> Clusters detectados con eps=1.5: {cosmic.cluster_count}")
    print(f"  -> Extracción jerárquica con eps'=0.8: {cosmic.cluster_count} clusters detectados")
    print(f"  -> Longitud del perfil de alcanzabilidad (reachability_plot): {len(cosmic.reachability_plot)}")
    
    results.append({
        "Modelo": "COSMIC",
        "Paradigma": "Clustering (Transductivo)",
        "Configuración": f"eps=1.5, min_pts=2, {metric}, dev={cosmic.device}",
        **res_cosmic
    })

    # -------------------------------------------------------------
    # Tabla resumen final
    # -------------------------------------------------------------
    df_results = pd.DataFrame(results)
    print("\n" + "=" * 75)
    print("RESUMEN COMPARATIVO DE RESULTADOS")
    print("=" * 75)
    cols = ["Modelo", "Paradigma", "Precision", "Recall", "F1-Score", "Specificity"]
    formatted_df = df_results[cols].copy()
    for metric_col in ["Precision", "Recall", "F1-Score", "Specificity"]:
        formatted_df[metric_col] = formatted_df[metric_col].map(lambda v: f"{v:.4f}")
    
    print(formatted_df.to_string(index=False))
    print("=" * 75 + "\n")

    return df_results


def main():
    parser = argparse.ArgumentParser(description="Prueba musk1 en todos los modelos de MIClustering con aceleración GPU y multinúcleo.")
    parser.add_argument("--dataset", type=str, default="musk1.arff", help="Nombre del archivo ARFF (por defecto: musk1.arff)")
    parser.add_argument("--train-pct", type=float, default=70.0, help="Porcentaje de entrenamiento (por defecto: 70.0)")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para partición y reproducibilidad (por defecto: 42)")
    parser.add_argument("--metric", type=str, default="hausdorff_avg", help="Métrica de distancia entre bolsas (por defecto: hausdorff_avg)")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Número de núcleos paralelos para CPU (-1 para todos los núcleos)")
    parser.add_argument("--device", type=str, default="auto", help="Dispositivo de cómputo: auto, mps, cuda, cpu (por defecto: auto)")
    parser.add_argument("--no-hw-bench", action="store_true", help="Desactiva la prueba comparativa de hardware")
    
    args = parser.parse_args()
    run_benchmark(
        dataset_name=args.dataset,
        train_pct=args.train_pct,
        seed=args.seed,
        metric=args.metric,
        n_jobs=args.n_jobs,
        device=args.device,
        run_hw_bench=not args.no_hw_bench
    )


if __name__ == "__main__":
    main()
