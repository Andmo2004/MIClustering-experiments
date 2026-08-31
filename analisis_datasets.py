"""
analisis_datasets.py — Análisis exploratorio de los 10 datasets MIL.

Para cada dataset, bajo diferentes estrategias de escalado (None, MinMax,
Standard), calcula estadísticas descriptivas de las bolsas y de las matrices
de distancia. El objetivo es informar los rangos de hiperparámetros para la
Fase 2 (Optuna) del protocolo experimental.

Uso:
    python analisis_datasets.py                     # Todos los datasets
    python analisis_datasets.py --datasets musk1 musk2  # Subconjunto
    python analisis_datasets.py --distances hausdorff hausdorff_avg  # Subconjunto de distancias
    python analisis_datasets.py --no-distance-stats  # Solo stats de bolsas (rápido)

Salida:
    - Tabla resumen por consola
    - CSV en resultados/analisis_datasets/ con todas las estadísticas
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

# Importar configuración central
from config import (
    PROJECT_ROOT,
    SRC_PATH,
    DATA_DIR,
    RESULTS_ROOT,
    DATASETS,
    DISTANCES,
    SCALERS,
    dataset_path,
    get_scaler,
    get_distance_func,
    ensure_result_dirs,
    setup_logging,
)

# Importaciones de MIClustering
from miclustering.data.arff_reader import ArffToMIData
from miclustering.data.midata import MIData
from miclustering.data.utils import parse_label
from miclustering.distances.distance_matrix import compute_distance_matrix

logger = setup_logging("analisis_datasets")

# Directorio de salida para este análisis
ANALYSIS_DIR = RESULTS_ROOT / "analisis_datasets"


# ═══════════════════════════════════════════════════════════════════════════
# Análisis de estructura de bolsas
# ═══════════════════════════════════════════════════════════════════════════

def analyze_bag_structure(dataset: MIData, dataset_name: str) -> Dict[str, Any]:
    """Analiza la estructura de un dataset MIL: bolsas, instancias, clases.

    Args:
        dataset: MIData cargado.
        dataset_name: Nombre identificador.

    Returns:
        Diccionario con estadísticas de estructura.
    """
    bags = dataset.bags
    n_bags = len(bags)

    # Instancias por bolsa
    instances_per_bag = [len(bag) for bag in bags]
    inst_arr = np.array(instances_per_bag)

    # Dimensionalidad (features de la primera instancia)
    first_bag = bags[0]
    first_instance = first_bag[0]
    n_features = len([
        v for i, v in enumerate(first_instance.values)
        if first_instance.schema[i].type.lower().strip() in ('real', 'integer', 'numeric', 'float', 'int')
    ])

    # Distribución de clases
    labels = []
    for bag in bags:
        lv = parse_label(bag.label) if isinstance(bag.label, (str, float)) else int(bag.label)
        labels.append(lv)
    label_arr = np.array(labels)
    unique, counts = np.unique(label_arr, return_counts=True)
    class_dist = {int(u): int(c) for u, c in zip(unique, counts)}
    imbalance_ratio = float(counts.min()) / float(counts.max()) if counts.max() > 0 else 0.0

    return {
        "dataset": dataset_name,
        "n_bags": n_bags,
        "n_features": n_features,
        "instances_mean": float(inst_arr.mean()),
        "instances_std": float(inst_arr.std()),
        "instances_min": int(inst_arr.min()),
        "instances_max": int(inst_arr.max()),
        "instances_median": float(np.median(inst_arr)),
        "class_distribution": class_dist,
        "imbalance_ratio": round(imbalance_ratio, 4),
        "k_max_sqrt": max(2, int(math.sqrt(n_bags))),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Análisis de distribución de distancias
# ═══════════════════════════════════════════════════════════════════════════

def analyze_distance_distribution(
    dist_matrix: np.ndarray,
    dataset_name: str,
    scaler_name: str,
    distance_name: str,
) -> Dict[str, Any]:
    """Calcula estadísticas sobre la distribución de distancias par-a-par.

    Estas estadísticas sirven para informar los rangos de epsilon
    (MIDBSCAN/COSMIC) de forma adaptativa.

    Args:
        dist_matrix: Matriz N×N de distancias.
        dataset_name: Nombre del dataset.
        scaler_name: Nombre del scaler aplicado.
        distance_name: Nombre de la métrica de distancia.

    Returns:
        Diccionario con percentiles y estadísticas de la distribución.
    """
    # Extraer triángulo superior (sin diagonal) para evitar duplicados
    n = dist_matrix.shape[0]
    upper_tri = dist_matrix[np.triu_indices(n, k=1)]

    if len(upper_tri) == 0:
        return {
            "dataset": dataset_name,
            "scaler": scaler_name,
            "distance": distance_name,
            "n_pairs": 0,
        }

    percentiles = [1, 5, 10, 25, 50, 60, 75, 90, 95, 99]
    pct_values = np.percentile(upper_tri, percentiles)

    result = {
        "dataset": dataset_name,
        "scaler": scaler_name,
        "distance": distance_name,
        "n_pairs": len(upper_tri),
        "mean": float(upper_tri.mean()),
        "std": float(upper_tri.std()),
        "min": float(upper_tri.min()),
        "max": float(upper_tri.max()),
    }

    for p, v in zip(percentiles, pct_values):
        result[f"p{p}"] = float(v)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline principal
# ═══════════════════════════════════════════════════════════════════════════

def run_analysis(
    dataset_names: Optional[List[str]] = None,
    distance_names: Optional[List[str]] = None,
    compute_distance_stats: bool = True,
    n_jobs: int = -1,
    device: str = "auto",
) -> None:
    """Ejecuta el análisis completo de datasets.

    Args:
        dataset_names: Lista de datasets a analizar (None = todos).
        distance_names: Lista de distancias a probar (None = todas las 6).
        compute_distance_stats: Si True, calcula matrices de distancia y sus estadísticas.
        n_jobs: Paralelismo para el cálculo de distancias.
        device: Dispositivo para el cálculo de distancias.
    """
    datasets = dataset_names or DATASETS
    distances = distance_names or DISTANCES
    scaler_names = list(SCALERS.keys())

    # Crear directorio de salida
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # ─── 1. Análisis de estructura ─────────────────────────────────────
    print("\n" + "═" * 75)
    print("  ANÁLISIS DE ESTRUCTURA DE DATASETS MIL")
    print("═" * 75)

    bag_stats: List[Dict] = []
    for ds_name in tqdm(datasets, desc="Analizando estructura"):
        arff_path = dataset_path(ds_name)
        if not arff_path.exists():
            logger.warning(f"Dataset no encontrado: {arff_path}")
            continue

        dataset = ArffToMIData.from_arff(arff_path)
        stats = analyze_bag_structure(dataset, ds_name)
        bag_stats.append(stats)

        logger.info(
            f"[{ds_name}] {stats['n_bags']} bolsas | "
            f"{stats['n_features']} features | "
            f"inst/bag: {stats['instances_mean']:.1f} ± {stats['instances_std']:.1f} | "
            f"clases: {stats['class_distribution']} | "
            f"imbalance: {stats['imbalance_ratio']}"
        )

    # Tabla resumen
    df_bags = pd.DataFrame(bag_stats)
    if not df_bags.empty:
        display_cols = [
            "dataset", "n_bags", "n_features",
            "instances_mean", "instances_std", "instances_min", "instances_max",
            "imbalance_ratio", "k_max_sqrt",
        ]
        print("\n" + df_bags[display_cols].to_string(index=False))

        # Guardar
        csv_path = ANALYSIS_DIR / "estructura_datasets.csv"
        df_bags.to_csv(csv_path, index=False)
        logger.info(f"Estructura guardada en {csv_path}")

        # Guardar también la distribución de clases como JSON aparte
        class_dist_path = ANALYSIS_DIR / "distribucion_clases.json"
        class_data = {row["dataset"]: row["class_distribution"] for row in bag_stats}
        with open(class_dist_path, "w", encoding="utf-8") as f:
            json.dump(class_data, f, indent=2, ensure_ascii=False)

    if not compute_distance_stats:
        print("\n[INFO] Análisis de distancias omitido (--no-distance-stats)")
        return

    # ─── 2. Análisis de distancias ─────────────────────────────────────
    print("\n" + "═" * 75)
    print("  ANÁLISIS DE DISTRIBUCIÓN DE DISTANCIAS")
    print(f"  {len(datasets)} datasets × {len(scaler_names)} scalers × {len(distances)} distancias")
    print("═" * 75)

    dist_stats: List[Dict] = []
    total_combos = len(datasets) * len(scaler_names) * len(distances)

    with tqdm(total=total_combos, desc="Calculando distancias") as pbar:
        for ds_name in datasets:
            arff_path = dataset_path(ds_name)
            if not arff_path.exists():
                pbar.update(len(scaler_names) * len(distances))
                continue

            dataset = ArffToMIData.from_arff(arff_path)

            for scaler_name in scaler_names:
                # Escalar
                scaler = get_scaler(scaler_name)
                scaled = scaler.fit_transform(dataset)
                bags = scaled.bags

                for dist_name in distances:
                    pbar.set_postfix_str(f"{ds_name}/{scaler_name}/{dist_name}")

                    try:
                        metric_func = get_distance_func(dist_name)
                        t0 = time.perf_counter()
                        matrix = compute_distance_matrix(
                            bags, metric_func,
                            metric_name=dist_name,
                            n_jobs=n_jobs,
                            device=device,
                        )
                        elapsed = time.perf_counter() - t0

                        stats = analyze_distance_distribution(
                            matrix, ds_name, scaler_name, dist_name
                        )
                        stats["compute_time_sec"] = round(elapsed, 4)
                        dist_stats.append(stats)

                    except Exception as e:
                        logger.error(f"Error en {ds_name}/{scaler_name}/{dist_name}: {e}")
                        dist_stats.append({
                            "dataset": ds_name,
                            "scaler": scaler_name,
                            "distance": dist_name,
                            "error": str(e),
                        })

                    pbar.update(1)

    # Tabla resumen de distancias
    df_dist = pd.DataFrame(dist_stats)
    if not df_dist.empty:
        # Mostrar resumen compacto
        display_cols = [c for c in [
            "dataset", "scaler", "distance",
            "mean", "std", "min", "max",
            "p5", "p25", "p50", "p75", "p95",
            "compute_time_sec",
        ] if c in df_dist.columns]

        if display_cols:
            print("\n" + df_dist[display_cols].to_string(index=False))

        # Guardar
        csv_path = ANALYSIS_DIR / "distribucion_distancias.csv"
        df_dist.to_csv(csv_path, index=False)
        logger.info(f"Distribución de distancias guardada en {csv_path}")

    # ─── 3. Resumen de rangos sugeridos para hiperparámetros ────────────
    if not df_dist.empty and "p5" in df_dist.columns:
        print("\n" + "═" * 75)
        print("  RANGOS SUGERIDOS DE EPSILON POR DATASET/SCALER/DISTANCIA")
        print("  (basados en percentiles de la distribución de distancias)")
        print("═" * 75)

        summary_rows = []
        for (ds, sc), group in df_dist.groupby(["dataset", "scaler"]):
            for _, row in group.iterrows():
                if "p5" in row and "p60" in row and pd.notna(row.get("p5")):
                    summary_rows.append({
                        "dataset": ds,
                        "scaler": sc,
                        "distance": row["distance"],
                        "epsilon_low (p5)": round(row["p5"], 4),
                        "epsilon_high (p60)": round(row.get("p60", row.get("p50", 0)), 4),
                        "epsilon_median (p50)": round(row["p50"], 4),
                    })

        if summary_rows:
            df_summary = pd.DataFrame(summary_rows)
            print("\n" + df_summary.to_string(index=False))

            csv_path = ANALYSIS_DIR / "rangos_epsilon_sugeridos.csv"
            df_summary.to_csv(csv_path, index=False)
            logger.info(f"Rangos de epsilon guardados en {csv_path}")

    print("\n" + "═" * 75)
    print("  ANÁLISIS COMPLETO")
    print(f"  Resultados en: {ANALYSIS_DIR}")
    print("═" * 75 + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Análisis exploratorio de datasets MIL para informar rangos de hiperparámetros.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help=f"Datasets a analizar (default: todos). Opciones: {DATASETS}",
    )
    parser.add_argument(
        "--distances", nargs="+", default=None,
        help=f"Distancias a probar (default: todas). Opciones: {DISTANCES}",
    )
    parser.add_argument(
        "--no-distance-stats", action="store_true",
        help="Omitir cálculo de matrices de distancia (solo estructura de bolsas).",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=-1,
        help="Número de procesos paralelos para cómputo de distancias (default: -1 = todos).",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Dispositivo de cómputo: auto, mps, cuda, cpu (default: auto).",
    )

    args = parser.parse_args()

    run_analysis(
        dataset_names=args.datasets,
        distance_names=args.distances,
        compute_distance_stats=not args.no_distance_stats,
        n_jobs=args.n_jobs,
        device=args.device,
    )


if __name__ == "__main__":
    main()
