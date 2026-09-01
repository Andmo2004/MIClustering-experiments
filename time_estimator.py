"""
time_estimator.py — Estimación de tiempos de ejecución para las fases del protocolo experimental.

Proporciona estimaciones previas al cómputo basadas en mediciones empíricas
de hardware y estado actual del disco (archivos cacheados, bases de datos Optuna, etc.).
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


# Tiempos de referencia empíricos (segundos por cálculo de matriz en hardware local)


BENCHMARK_MATRIX_SECONDS: Dict[Tuple[str, str, str], float] = {
    # (dataset, scaler, distance): segundos
    ("musk1", "MinMaxScaler", "hausdorff"): 4.80,
    ("musk1", "MinMaxScaler", "hausdorff_min"): 2.61,
    ("musk1", "MinMaxScaler", "hausdorff_avg"): 2.62,
    ("musk1", "MinMaxScaler", "cauchy_schwarz"): 4.10,
    ("musk1", "MinMaxScaler", "earth_movers"): 0.17,
    ("musk1", "MinMaxScaler", "mahalanobis"): 13.23,
    ("musk1", "StandardScaler", "hausdorff"): 2.66,
    ("musk1", "StandardScaler", "hausdorff_min"): 2.43,
    ("musk1", "StandardScaler", "hausdorff_avg"): 2.57,
    ("musk1", "StandardScaler", "cauchy_schwarz"): 3.99,
    ("musk1", "StandardScaler", "earth_movers"): 0.17,
    ("musk1", "StandardScaler", "mahalanobis"): 13.05,

    ("musk2", "MinMaxScaler", "hausdorff"): 14.44,
    ("musk2", "MinMaxScaler", "hausdorff_min"): 10.12,
    ("musk2", "MinMaxScaler", "hausdorff_avg"): 3.55,
    ("musk2", "MinMaxScaler", "cauchy_schwarz"): 4.93,
    ("musk2", "MinMaxScaler", "earth_movers"): 7.44,
    ("musk2", "MinMaxScaler", "mahalanobis"): 16.81,
    ("musk2", "StandardScaler", "hausdorff"): 3.67,
    ("musk2", "StandardScaler", "hausdorff_min"): 3.31,
    ("musk2", "StandardScaler", "hausdorff_avg"): 3.46,
    ("musk2", "StandardScaler", "cauchy_schwarz"): 4.84,
    ("musk2", "StandardScaler", "earth_movers"): 7.15,
    ("musk2", "StandardScaler", "mahalanobis"): 16.81,

    ("mutagenesis3_atoms", "MinMaxScaler", "hausdorff"): 12.45,
    ("mutagenesis3_atoms", "MinMaxScaler", "hausdorff_min"): 10.69,
    ("mutagenesis3_atoms", "MinMaxScaler", "hausdorff_avg"): 10.98,
    ("mutagenesis3_atoms", "MinMaxScaler", "cauchy_schwarz"): 16.64,
    ("mutagenesis3_atoms", "MinMaxScaler", "earth_movers"): 0.71,
    ("mutagenesis3_atoms", "MinMaxScaler", "mahalanobis"): 36.91,
    ("mutagenesis3_atoms", "StandardScaler", "hausdorff"): 11.13,
    ("mutagenesis3_atoms", "StandardScaler", "hausdorff_min"): 10.11,
    ("mutagenesis3_atoms", "StandardScaler", "hausdorff_avg"): 10.83,
    ("mutagenesis3_atoms", "StandardScaler", "cauchy_schwarz"): 16.56,
    ("mutagenesis3_atoms", "StandardScaler", "earth_movers"): 0.72,
    ("mutagenesis3_atoms", "StandardScaler", "mahalanobis"): 36.68,

    ("mutagenesis3_chains", "MinMaxScaler", "hausdorff"): 34.01,
    ("mutagenesis3_chains", "MinMaxScaler", "hausdorff_min"): 22.81,
    ("mutagenesis3_chains", "MinMaxScaler", "hausdorff_avg"): 11.75,
    ("mutagenesis3_chains", "MinMaxScaler", "cauchy_schwarz"): 16.75,
    ("mutagenesis3_chains", "MinMaxScaler", "earth_movers"): 1.64,
    ("mutagenesis3_chains", "MinMaxScaler", "mahalanobis"): 37.86,
    ("mutagenesis3_chains", "StandardScaler", "hausdorff"): 12.33,
    ("mutagenesis3_chains", "StandardScaler", "hausdorff_min"): 11.33,
    ("mutagenesis3_chains", "StandardScaler", "hausdorff_avg"): 12.09,
    ("mutagenesis3_chains", "StandardScaler", "cauchy_schwarz"): 17.05,
    ("mutagenesis3_chains", "StandardScaler", "earth_movers"): 1.67,
    ("mutagenesis3_chains", "StandardScaler", "mahalanobis"): 37.61,

    ("BirdsChestnut-backedChickadee", "MinMaxScaler", "hausdorff"): 119.78,
    ("BirdsChestnut-backedChickadee", "MinMaxScaler", "hausdorff_min"): 93.38,
    ("BirdsChestnut-backedChickadee", "MinMaxScaler", "hausdorff_avg"): 96.72,
    ("BirdsChestnut-backedChickadee", "MinMaxScaler", "cauchy_schwarz"): 144.22,
    ("BirdsChestnut-backedChickadee", "MinMaxScaler", "earth_movers"): 10.34,
    ("BirdsChestnut-backedChickadee", "MinMaxScaler", "mahalanobis"): 327.06,
    ("BirdsChestnut-backedChickadee", "StandardScaler", "hausdorff"): 99.59,
    ("BirdsChestnut-backedChickadee", "StandardScaler", "hausdorff_min"): 89.22,
    ("BirdsChestnut-backedChickadee", "StandardScaler", "hausdorff_avg"): 96.94,
    ("BirdsChestnut-backedChickadee", "StandardScaler", "cauchy_schwarz"): 142.95,
    ("BirdsChestnut-backedChickadee", "StandardScaler", "earth_movers"): 10.26,
    ("BirdsChestnut-backedChickadee", "StandardScaler", "mahalanobis"): 326.12,

    ("BirdsHammondsFlycatcher", "MinMaxScaler", "hausdorff"): 99.64,
    ("BirdsHammondsFlycatcher", "MinMaxScaler", "hausdorff_min"): 89.00,
    ("BirdsHammondsFlycatcher", "MinMaxScaler", "hausdorff_avg"): 96.67,
    ("BirdsHammondsFlycatcher", "MinMaxScaler", "cauchy_schwarz"): 141.24,
    ("BirdsHammondsFlycatcher", "MinMaxScaler", "earth_movers"): 10.33,
    ("BirdsHammondsFlycatcher", "MinMaxScaler", "mahalanobis"): 393.14,
    ("BirdsHammondsFlycatcher", "StandardScaler", "hausdorff"): 99.70,
    ("BirdsHammondsFlycatcher", "StandardScaler", "hausdorff_min"): 90.56,
    ("BirdsHammondsFlycatcher", "StandardScaler", "hausdorff_avg"): 97.84,
    ("BirdsHammondsFlycatcher", "StandardScaler", "cauchy_schwarz"): 141.96,
    ("BirdsHammondsFlycatcher", "StandardScaler", "earth_movers"): 10.43,
    ("BirdsHammondsFlycatcher", "StandardScaler", "mahalanobis"): 325.52,

    ("Harddrive1", "MinMaxScaler", "hausdorff"): 547.56,
    ("Harddrive1", "MinMaxScaler", "hausdorff_min"): 383.04,
    ("Harddrive1", "MinMaxScaler", "hausdorff_avg"): 54.72,
    ("Harddrive1", "MinMaxScaler", "cauchy_schwarz"): 67.06,
    ("Harddrive1", "MinMaxScaler", "earth_movers"): 334.27,
    ("Harddrive1", "MinMaxScaler", "mahalanobis"): 159.86,
    ("Harddrive1", "StandardScaler", "hausdorff"): 54.64,
    ("Harddrive1", "StandardScaler", "hausdorff_min"): 47.73,
    ("Harddrive1", "StandardScaler", "hausdorff_avg"): 54.66,
    ("Harddrive1", "StandardScaler", "cauchy_schwarz"): 67.07,
    ("Harddrive1", "StandardScaler", "earth_movers"): 291.13,
    ("Harddrive1", "StandardScaler", "mahalanobis"): 155.29,

    ("ImageElephant", "MinMaxScaler", "hausdorff"): 13.10,
    ("ImageElephant", "MinMaxScaler", "hausdorff_min"): 11.88,
    ("ImageElephant", "MinMaxScaler", "hausdorff_avg"): 12.78,
    ("ImageElephant", "MinMaxScaler", "cauchy_schwarz"): 19.88,
    ("ImageElephant", "MinMaxScaler", "earth_movers"): 0.97,
    ("ImageElephant", "MinMaxScaler", "mahalanobis"): 92.35,
    ("ImageElephant", "StandardScaler", "hausdorff"): 13.13,
    ("ImageElephant", "StandardScaler", "hausdorff_min"): 11.93,
    ("ImageElephant", "StandardScaler", "hausdorff_avg"): 12.76,
    ("ImageElephant", "StandardScaler", "cauchy_schwarz"): 19.66,
    ("ImageElephant", "StandardScaler", "earth_movers"): 0.97,
    ("ImageElephant", "StandardScaler", "mahalanobis"): 92.72,

    ("Newsgroups1", "MinMaxScaler", "hausdorff"): 173.29,
    ("Newsgroups1", "MinMaxScaler", "hausdorff_min"): 55.71,
    ("Newsgroups1", "MinMaxScaler", "hausdorff_avg"): 3.94,
    ("Newsgroups1", "MinMaxScaler", "cauchy_schwarz"): 4.94,
    ("Newsgroups1", "MinMaxScaler", "earth_movers"): 2.87,
    ("Newsgroups1", "MinMaxScaler", "mahalanobis"): 23.32,
    ("Newsgroups1", "StandardScaler", "hausdorff"): 3.83,
    ("Newsgroups1", "StandardScaler", "hausdorff_min"): 3.40,
    ("Newsgroups1", "StandardScaler", "hausdorff_avg"): 3.58,
    ("Newsgroups1", "StandardScaler", "cauchy_schwarz"): 4.88,
    ("Newsgroups1", "StandardScaler", "earth_movers"): 2.87,
    ("Newsgroups1", "StandardScaler", "mahalanobis"): 20.66,

    ("Thioredoxin", "MinMaxScaler", "hausdorff"): 562.62,
    ("Thioredoxin", "MinMaxScaler", "hausdorff_min"): 233.56,
    ("Thioredoxin", "MinMaxScaler", "hausdorff_avg"): 14.70,
    ("Thioredoxin", "MinMaxScaler", "cauchy_schwarz"): 18.65,
    ("Thioredoxin", "MinMaxScaler", "earth_movers"): 33.39,
    ("Thioredoxin", "MinMaxScaler", "mahalanobis"): 45.25,
    ("Thioredoxin", "StandardScaler", "hausdorff"): 15.82,
    ("Thioredoxin", "StandardScaler", "hausdorff_min"): 12.68,
    ("Thioredoxin", "StandardScaler", "hausdorff_avg"): 15.17,
    ("Thioredoxin", "StandardScaler", "cauchy_schwarz"): 18.74,
    ("Thioredoxin", "StandardScaler", "earth_movers"): 32.34,
    ("Thioredoxin", "StandardScaler", "mahalanobis"): 38.80,
}

DEFAULT_MATRIX_SECONDS = 30.0



# Formateo de tiempos para lectura humana


def format_duration(seconds: float) -> str:
    """Convierte segundos a una cadena legible (ej: '< 1 segundo', '~45 s', '~12 min 30 s', '~2 h 06 min')."""
    if seconds < 1.0:
        return "< 1 segundo"
    elif seconds < 60.0:
        sec = round(seconds)
        return f"~{sec} segundo{'s' if sec != 1 else ''}"
    elif seconds < 3600.0:
        mins = int(seconds // 60)
        rem_sec = round(seconds % 60)
        if rem_sec == 60:
            mins += 1
            rem_sec = 0
        if rem_sec == 0:
            return f"~{mins} minuto{'s' if mins != 1 else ''}"
        return f"~{mins} min {rem_sec:02d} s"
    else:
        hours = int(seconds // 3600)
        mins = round((seconds % 3600) / 60)
        if mins == 60:
            hours += 1
            mins = 0
        if mins == 0:
            return f"~{hours} hora{'s' if hours != 1 else ''}"
        return f"~{hours} h {mins:02d} min"


def format_duration_range(min_seconds: float, max_seconds: float) -> str:
    """Formatea un rango de estimación (ej: '~2 – 5 segundos', '~2 min 30 s – ~3 min 30 s')."""
    if max_seconds < 1.0:
        return "< 1 segundo"
    if min_seconds == max_seconds or abs(max_seconds - min_seconds) < 1.0:
        return format_duration(min_seconds)

    if max_seconds < 60.0:
        return f"~{round(min_seconds)} – {round(max_seconds)} segundos"

    return f"{format_duration(min_seconds)} – {format_duration(max_seconds)}"



# Estimadores por fase


def estimate_fase0() -> Tuple[float, float, str, List[str]]:
    """Estimación para Fase 0: Setup y reproducibilidad."""
    min_sec = 2.0
    max_sec = 5.0
    details = [
        "Verificación de 10 datasets .arff (SHA-256)",
        "Inspección de versiones del entorno y commit git",
        "Documentación de diseño y derivación de semillas",
    ]
    return min_sec, max_sec, format_duration_range(min_sec, max_sec), details


def estimate_analisis_datasets(
    dataset_names: Optional[List[str]] = None,
    include_distances: bool = True,
    distance_names: Optional[List[str]] = None,
) -> Tuple[float, float, str, List[str]]:
    """Estimación para analisis_datasets.py."""
    from config import DATASETS, DISTANCES, SCALERS

    datasets = dataset_names or DATASETS
    distances = distance_names or DISTANCES
    scalers = list(SCALERS.keys())

    struct_time = len(datasets) * 0.25

    if not include_distances:
        min_sec = max(1.0, struct_time * 0.8)
        max_sec = max(2.0, struct_time * 1.5 + 1.0)
        details = [
            f"Solo análisis de estructura ({len(datasets)} datasets)",
            "Distancias omitidas (--no-distance-stats)",
        ]
        return min_sec, max_sec, format_duration_range(min_sec, max_sec), details

    total_dist_time = 0.0
    for ds in datasets:
        for sc in scalers:
            for dist in distances:
                t = BENCHMARK_MATRIX_SECONDS.get((ds, sc, dist), DEFAULT_MATRIX_SECONDS)
                total_dist_time += t

    total_est = struct_time + total_dist_time
    min_sec = total_est * 0.9
    max_sec = total_est * 1.1

    n_combos = len(datasets) * len(scalers) * len(distances)
    details = [
        f"Estructura: {len(datasets)} datasets ({format_duration(struct_time)})",
        f"Distancias: {n_combos} combinaciones ({len(datasets)} datasets × {len(scalers)} scalers × {len(distances)} distancias)",
        "Datasets pesados: Birds (~44 min), Harddrive1 (~34 min), Thioredoxin (~16 min)",
    ]
    return min_sec, max_sec, format_duration_range(min_sec, max_sec), details


def estimate_fase1(
    dataset_names: Optional[List[str]] = None,
    distance_names: Optional[List[str]] = None,
    scaler_names: Optional[List[str]] = None,
    force: bool = False,
) -> Tuple[float, float, str, List[str]]:
    """Estimación para Fase 1: Precómputo de matrices de distancias."""
    from config import DATASETS, DISTANCES, SCALERS, distance_matrix_path

    datasets = dataset_names or DATASETS
    distances = distance_names or DISTANCES
    scalers = scaler_names or list(SCALERS.keys())

    total_matrices = len(datasets) * len(scalers) * len(distances)
    cached_count = 0
    pending_count = 0
    pending_seconds = 0.0

    for ds in datasets:
        for sc in scalers:
            for dist in distances:
                matrix_p = distance_matrix_path(ds, sc, dist)
                if matrix_p.exists() and not force:
                    cached_count += 1
                else:
                    pending_count += 1
                    t = BENCHMARK_MATRIX_SECONDS.get((ds, sc, dist), DEFAULT_MATRIX_SECONDS)
                    pending_seconds += t

    if pending_count == 0:
        min_sec = 0.1
        max_sec = 0.5
        details = [
            f"{cached_count}/{total_matrices} matrices ya presentes en caché (.npy)",
            "No se requiere cómputo nuevo (usar --force para forzar recálculo)",
        ]
        return min_sec, max_sec, "< 1 segundo (100% en caché)", details

    min_sec = max(1.0, pending_seconds * 0.9)
    max_sec = max(2.0, pending_seconds * 1.15)

    details = [
        f"Total matrices: {total_matrices} ({cached_count} en caché, {pending_count} pendientes a computar)",
        "Los modelos de las Fases 2 y 3 reutilizarán estas matrices directamente",
    ]
    if pending_count > 0:
        heavy = []
        for ds in ["BirdsChestnut-backedChickadee", "BirdsHammondsFlycatcher", "Harddrive1", "Thioredoxin"]:
            if ds in datasets:
                heavy.append(ds.split("-")[0][:12])
        if heavy:
            details.append(f"Datasets con mayor carga de cómputo: {', '.join(heavy)}")

    return min_sec, max_sec, format_duration_range(min_sec, max_sec), details


def estimate_fase2(
    dataset_names: Optional[List[str]] = None,
    model_names: Optional[List[str]] = None,
    n_trials: int = 80,
    resume: bool = False,
) -> Tuple[float, float, str, List[str]]:
    """Estimación para Fase 2: Optimización de hiperparámetros con Optuna."""
    from config import (
        DATASETS, MODELS, UNSUPERVISED_MODELS, SUPERVISED_MODELS,
        RESULTS_DISTANCIAS, optuna_db_path, optuna_study_name,
    )

    datasets = dataset_names or DATASETS
    models = model_names or MODELS

    total_studies = len(datasets) * len(models)
    total_trials = total_studies * n_trials
    pending_trials = 0

    if resume:
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.ERROR)
            for ds in datasets:
                for m in models:
                    db_p = optuna_db_path(ds, m)
                    if db_p.exists():
                        try:
                            st = optuna.load_study(
                                study_name=optuna_study_name(ds, m),
                                storage=f"sqlite:///{db_p}",
                            )
                            done = len(st.trials)
                            pending_trials += max(0, n_trials - done)
                        except Exception:
                            pending_trials += n_trials
                    else:
                        pending_trials += n_trials
        except Exception:
            pending_trials = total_trials
    else:
        pending_trials = total_trials

    unsupervised_count = sum(1 for m in models if m in UNSUPERVISED_MODELS)
    supervised_count = sum(1 for m in models if m in SUPERVISED_MODELS)
    total_m = len(models) if len(models) > 0 else 1

    avg_sec_per_trial = (
        (unsupervised_count * 0.035 + supervised_count * 0.055) / total_m
    )

    overhead = total_studies * 0.4
    calc_seconds = pending_trials * avg_sec_per_trial + overhead

    min_sec = max(2.0, calc_seconds * 0.85)
    max_sec = max(5.0, calc_seconds * 1.25)

    n_cached_matrices = len(list(RESULTS_DISTANCIAS.glob("*.npy"))) if RESULTS_DISTANCIAS.exists() else 0
    matrices_ready = n_cached_matrices >= 100

    details = [
        f"{total_studies} estudios Optuna ({len(datasets)} datasets × {len(models)} modelos)",
        f"{n_trials} trials por estudio ({pending_trials} trials pendientes de evaluar)",
        f"Matrices precomputadas en Fase 1: {'✓ Listas en disco' if matrices_ready else '⚠ Verificar ejecución de Fase 1'}",
    ]
    return min_sec, max_sec, format_duration_range(min_sec, max_sec), details


def estimate_fase3(
    dataset_names: Optional[List[str]] = None,
    model_names: Optional[List[str]] = None,
    n_replicas: int = 10,
    force: bool = False,
) -> Tuple[float, float, str, List[str]]:
    """Estimación para Fase 3: Evaluación final con réplicas."""
    from config import (
        DATASETS, MODELS, UNSUPERVISED_MODELS, SUPERVISED_MODELS,
        evaluation_result_path,
    )

    datasets = dataset_names or DATASETS
    models = model_names or MODELS

    total_evals = len(datasets) * len(models) * n_replicas
    pending_evals = 0
    cached_evals = 0

    for ds in datasets:
        for m in models:
            for r in range(n_replicas):
                res_p = evaluation_result_path(ds, m, r)
                if res_p.exists() and not force:
                    cached_evals += 1
                else:
                    pending_evals += 1

    if pending_evals == 0:
        min_sec = 0.1
        max_sec = 0.5
        details = [
            f"{cached_evals}/{total_evals} evaluaciones ya calculadas en disco (.json)",
            "No se requiere cómputo nuevo (usar --force para forzar recálculo)",
        ]
        return min_sec, max_sec, "< 1 segundo (100% en caché)", details

    unsupervised_count = sum(1 for m in models if m in UNSUPERVISED_MODELS)
    supervised_count = sum(1 for m in models if m in SUPERVISED_MODELS)
    total_m = len(models) if len(models) > 0 else 1

    ratio_unsup = unsupervised_count / total_m
    ratio_sup = supervised_count / total_m

    pending_unsup = pending_evals * ratio_unsup
    pending_sup = pending_evals * ratio_sup

    est_sec = (pending_unsup * 0.04) + (pending_sup * 0.35) + 3.0

    min_sec = max(1.0, est_sec * 0.85)
    max_sec = max(3.0, est_sec * 1.25)

    details = [
        f"{len(datasets) * len(models)} configuraciones ganadoras × {n_replicas} réplicas = {total_evals} evaluaciones",
        f"Evaluaciones pendientes: {pending_evals} ({cached_evals} ya presentes en disco)",
        "Modelos no supervisados: transductivo sobre matriz precomputada",
        "MIKNN: validación cruzada estratificada (5-fold) con recálculo por fold",
    ]
    return min_sec, max_sec, format_duration_range(min_sec, max_sec), details


def estimate_fase4() -> Tuple[float, float, str, List[str]]:
    """Estimación para Fase 4: Análisis estadístico comparativo."""
    min_sec = 3.0
    max_sec = 8.0
    details = [
        "Test de Friedman sobre rankings de modelos (PI1)",
        "Post-hoc Nemenyi / Wilcoxon pareado con corrección de Holm",
        "Comparaciones por distancia (PI2) y escalado (PI3)",
        "Correlación Spearman entre Silhouette y métricas externas (PI4)",
        "Generación de diagramas de diferencia crítica (CD diagrams)",
    ]
    return min_sec, max_sec, format_duration_range(min_sec, max_sec), details


def estimate_fase5() -> Tuple[float, float, str, List[str]]:
    """Estimación para Fase 5: Síntesis y reporte de resultados."""
    min_sec = 5.0
    max_sec = 15.0
    details = [
        "Generación de tablas resumen por modelo y distancia (LaTeX, CSV, Markdown)",
        "Renderizado de 6+ figuras de alta resolución (CD diagrams, Heatmaps, Boxplots)",
        "Documentación de amenazas a la validez (§5.5 del protocolo)",
    ]
    return min_sec, max_sec, format_duration_range(min_sec, max_sec), details



# Impresión estandarizada del encabezado de fase


def print_phase_header(
    phase_title: str,
    estimated_time_str: str,
    details: Optional[List[str]] = None,
    logger: Optional[Any] = None,
) -> None:
    """Imprime el banner visual estándar al inicio de cada fase con la estimación de tiempo.

    Args:
        phase_title: Título de la fase (ej: 'FASE 1 — PRECÓMPUTO DE MATRICES DE DISTANCIA').
        estimated_time_str: Cadena de tiempo estimado (ej: '~2 h 05 min').
        details: Lista opcional de detalles/parámetros clave.
        logger: Logger opcional para registrar también el mensaje.
    """
    line = "═" * 75
    print("\n" + line)
    print(f"  {phase_title}")
    print(f"  ⏱  Tiempo estimado: {estimated_time_str}")
    if details:
        print("  " + "─" * 71)
        for d in details:
            print(f"  • {d}")
    print(line + "\n")

    if logger is not None:
        logger.info(f"Iniciando: {phase_title} | Tiempo estimado: {estimated_time_str}")
