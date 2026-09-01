"""
fase1_distancias.py — Fase 1: Precómputo de matrices de distancia.

Para cada combinación (dataset, escalado, distancia) = 10×2×6 = 120 matrices,
calcula la matriz de distancias par-a-par entre bolsas del dataset completo,
una única vez, y la persiste en disco (.npy).

Los 4 modelos no supervisados (COSMIC, MIDBSCAN, MIKMEANS, MIKMEDOIDS)
reutilizan estas matrices directamente. MIKNN las usa en la búsqueda de
hiperparámetros (Fase 2) pero recalcula submatrices por fold en la
evaluación final (Fase 3).

Uso:
    python fase1_distancias.py                          # Todas las 120 matrices
    python fase1_distancias.py --datasets musk1 musk2   # Subconjunto de datasets
    python fase1_distancias.py --distances hausdorff     # Solo una distancia
    python fase1_distancias.py --force                   # Recalcular aunque existan
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    DATASETS,
    DISTANCES,
    SCALERS,
    RESULTS_DISTANCIAS,
    RESULTS_LOGS,
    dataset_path,
    distance_matrix_path,
    get_scaler,
    get_distance_func,
    ensure_result_dirs,
    setup_logging,
)
from time_estimator import estimate_fase1, print_phase_header

from miclustering.data.arff_reader import ArffToMIData
from miclustering.distances.distance_matrix import compute_distance_matrix

logger = setup_logging("fase1")


def compute_and_save_matrix(
    dataset_name: str,
    scaler_name: str,
    distance_name: str,
    n_jobs: int = -1,
    device: str = "auto",
    force: bool = False,
) -> dict:
    """Calcula y guarda una matriz de distancias.

    Args:
        dataset_name: Nombre del dataset.
        scaler_name: Nombre del escalador.
        distance_name: Nombre de la métrica de distancia.
        n_jobs: Paralelismo.
        device: Dispositivo de cómputo.
        force: Si True, recalcula aunque ya exista en disco.

    Returns:
        Dict con metadatos del cómputo (tiempos, shape, path, etc.).
    """
    out_path = distance_matrix_path(dataset_name, scaler_name, distance_name)
    combo_id = f"{dataset_name}/{scaler_name}/{distance_name}"

    # ¿Ya existe?
    if out_path.exists() and not force:
        matrix = np.load(out_path)
        logger.info(f"[CACHE HIT] {combo_id} → {out_path.name} ({matrix.shape})")
        return {
            "dataset": dataset_name,
            "scaler": scaler_name,
            "distance": distance_name,
            "shape": list(matrix.shape),
            "cached": True,
            "file": str(out_path),
        }

    # Cargar dataset
    arff_path = dataset_path(dataset_name)
    if not arff_path.exists():
        logger.error(f"Dataset no encontrado: {arff_path}")
        return {
            "dataset": dataset_name,
            "scaler": scaler_name,
            "distance": distance_name,
            "error": "FILE_NOT_FOUND",
        }

    dataset = ArffToMIData.from_arff(arff_path)

    # Escalar
    scaler = get_scaler(scaler_name)
    scaled_dataset = scaler.fit_transform(dataset)
    bags = scaled_dataset.bags

    # Calcular distancias
    metric_func = get_distance_func(distance_name)
    logger.info(f"[COMPUTING] {combo_id} ({len(bags)} bolsas)...")

    t0 = time.perf_counter()
    matrix = compute_distance_matrix(
        bags, metric_func,
        metric_name=distance_name,
        n_jobs=n_jobs,
        device=device,
    )
    elapsed = time.perf_counter() - t0

    # Validar
    assert matrix.shape[0] == matrix.shape[1] == len(bags), \
        f"Shape inesperado: {matrix.shape} para {len(bags)} bolsas"
    assert np.allclose(matrix, matrix.T, atol=1e-10), \
        f"Matriz no simétrica para {combo_id}"
    assert np.allclose(np.diag(matrix), 0, atol=1e-10), \
        f"Diagonal no cero para {combo_id}"

    # Guardar
    np.save(out_path, matrix)
    logger.info(
        f"[SAVED] {combo_id} → {out_path.name} "
        f"({matrix.shape}, {elapsed:.2f}s)"
    )

    return {
        "dataset": dataset_name,
        "scaler": scaler_name,
        "distance": distance_name,
        "shape": list(matrix.shape),
        "n_bags": len(bags),
        "compute_time_sec": round(elapsed, 4),
        "cached": False,
        "file": str(out_path),
    }


def run_fase1(
    dataset_names: Optional[List[str]] = None,
    distance_names: Optional[List[str]] = None,
    n_jobs: int = -1,
    device: str = "auto",
    force: bool = False,
) -> None:
    """Ejecuta la Fase 1 completa: precómputo de matrices de distancia.

    Args:
        dataset_names: Datasets a procesar (None = todos).
        distance_names: Distancias a computar (None = todas).
        n_jobs: Paralelismo.
        device: Dispositivo.
        force: Recalcular matrices existentes.
    """
    datasets = dataset_names or DATASETS
    distances = distance_names or DISTANCES
    scaler_names = list(SCALERS.keys())

    ensure_result_dirs()

    total = len(datasets) * len(scaler_names) * len(distances)

    # Estimación de tiempo
    _, _, time_str, details = estimate_fase1(
        dataset_names=datasets,
        distance_names=distances,
        scaler_names=scaler_names,
        force=force,
    )
    print_phase_header(
        phase_title="FASE 1 — PRECÓMPUTO DE MATRICES DE DISTANCIA",
        estimated_time_str=time_str,
        details=details,
        logger=logger,
    )

    results = []
    errors = []

    with tqdm(total=total, desc="Fase 1", ncols=80) as pbar:
        for ds_name in datasets:
            for scaler_name in scaler_names:
                for dist_name in distances:
                    pbar.set_postfix_str(
                        f"{ds_name[:12]}/{scaler_name[:6]}/{dist_name[:10]}"
                    )
                    try:
                        result = compute_and_save_matrix(
                            ds_name, scaler_name, dist_name,
                            n_jobs=n_jobs, device=device, force=force,
                        )
                        results.append(result)
                        if "error" in result:
                            errors.append(result)
                    except Exception as e:
                        logger.error(f"Error fatal en {ds_name}/{scaler_name}/{dist_name}: {e}")
                        errors.append({
                            "dataset": ds_name,
                            "scaler": scaler_name,
                            "distance": dist_name,
                            "error": str(e),
                        })
                    pbar.update(1)

    # Guardar resumen de tiempos
    df = pd.DataFrame(results)
    if not df.empty:
        csv_path = RESULTS_LOGS / "fase1_tiempos.csv"
        df.to_csv(csv_path, index=False)
        # Guardar también en RESULTS_DISTANCIAS
        df.to_csv(RESULTS_DISTANCIAS / "fase1_matrices_resumen.csv", index=False)
        logger.info(f"Tiempos guardados en {csv_path} y {RESULTS_DISTANCIAS / 'fase1_matrices_resumen.csv'}")

        # Si hay tiempos calculados, crear tabla pivote (dataset x distancia) por scaler
        if "compute_time_sec" in df.columns and not df.empty:
            for sc in scaler_names:
                df_sc = df[df["scaler"] == sc]
                if not df_sc.empty and "distance" in df_sc.columns:
                    piv = df_sc.pivot_table(index="dataset", columns="distance", values="compute_time_sec", aggfunc="mean")
                    piv.to_csv(RESULTS_DISTANCIAS / f"fase1_tiempos_{sc}.csv")

        # Resumen por consola
        computed = df[df.get("cached", False) == False]
        cached = df[df.get("cached", False) == True]
        print(f"\n  Matrices calculadas: {len(computed)}")
        print(f"  Matrices cacheadas:  {len(cached)}")
        if "compute_time_sec" in computed.columns and not computed.empty:
            print(f"  Tiempo total de cómputo: {computed['compute_time_sec'].sum():.2f}s")
            print(f"  Tiempo medio por matriz: {computed['compute_time_sec'].mean():.2f}s")

    if errors:
        print(f"\n  ⚠ Errores: {len(errors)}")
        for e in errors:
            print(f"    - {e.get('dataset')}/{e.get('scaler')}/{e.get('distance')}: {e.get('error')}")

    print("\n" + "═" * 75)
    print("  FASE 1 COMPLETADA")
    print(f"  Matrices en: {RESULTS_DISTANCIAS}")
    print("═" * 75 + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Fase 1: Precómputo de matrices de distancia entre bolsas.",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help=f"Datasets a procesar (default: todos). Opciones: {DATASETS}",
    )
    parser.add_argument(
        "--distances", nargs="+", default=None,
        help=f"Distancias a computar (default: todas). Opciones: {DISTANCES}",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=-1,
        help="Paralelismo para cómputo de distancias (default: -1 = todos).",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Dispositivo: auto, mps, cuda, cpu (default: auto).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Recalcular matrices aunque ya existan en disco.",
    )

    args = parser.parse_args()

    run_fase1(
        dataset_names=args.datasets,
        distance_names=args.distances,
        n_jobs=args.n_jobs,
        device=args.device,
        force=args.force,
    )


if __name__ == "__main__":
    main()
