"""
config.py — Configuración central del experimento.

Contiene todas las constantes, rutas, y funciones auxiliares que comparten
las Fases 0-5 del protocolo experimental. Un único punto de verdad para
reproducibilidad.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


# Rutas del proyecto


# Raíz del proyecto (MI-DBSCAN/) y del paquete de experimentos
EXP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parent

# Asegurar acceso al paquete miclustering desde lib-code
SRC_PATH = PROJECT_ROOT / "lib-code" / "MIClustering" / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Directorio de datos (datasets .arff)
DATA_DIR = EXP_DIR / "data"

# Directorio raíz de resultados (todo reproducible desde aquí)
RESULTS_ROOT = EXP_DIR / "resultados"

# Sub-directorios de resultados por fase
RESULTS_DISTANCIAS        = RESULTS_ROOT / "distancias"
RESULTS_ESTUDIOS_OPTUNA   = RESULTS_ROOT / "estudios_optuna"
RESULTS_EVALUACION_FINAL  = RESULTS_ROOT / "evaluacion_final"
RESULTS_ANALISIS_ESTADIST = RESULTS_ROOT / "analisis_estadistico"
RESULTS_REPORTE           = RESULTS_ROOT / "reporte"
RESULTS_LOGS              = RESULTS_ROOT / "logs"

ALL_RESULT_DIRS = [
    RESULTS_ROOT,
    RESULTS_DISTANCIAS,
    RESULTS_ESTUDIOS_OPTUNA,
    RESULTS_EVALUACION_FINAL,
    RESULTS_ANALISIS_ESTADIST,
    RESULTS_REPORTE,
    RESULTS_LOGS,
]



# Semillas y reproducibilidad


MASTER_SEED = 42

def derive_seeds(master_seed: int = MASTER_SEED, n: int = 10) -> List[int]:
    """Deriva n semillas de réplica de forma determinista usando SeedSequence.

    Esto garantiza que todo el experimento sea reproducible desde un único
    valor (MASTER_SEED), como exige la Fase 0 del protocolo.

    Args:
        master_seed: Semilla maestra del experimento.
        n: Número de semillas a derivar (= número de réplicas).

    Returns:
        Lista de n enteros, cada uno una semilla independiente.
    """
    ss = np.random.SeedSequence(master_seed)
    child_seeds = ss.spawn(n)
    # Generar un entero de 32 bits por cada child seed
    return [int(cs.generate_state(1)[0]) for cs in child_seeds]


# Número de réplicas (semillas) para la evaluación final (Fase 3)
N_REPLICAS = 10

# Semillas derivadas (pre-calculadas para uso inmediato)
REPLICA_SEEDS = derive_seeds(MASTER_SEED, N_REPLICAS)



# Diseño factorial


# --- Datasets (10 archivos .arff) ---
DATASETS: List[str] = [
    "musk1",
    "musk2",
    "mutagenesis3_atoms",
    "mutagenesis3_chains",
    "BirdsChestnut-backedChickadee",
    "BirdsHammondsFlycatcher",
    "Harddrive1",
    "ImageElephant",
    "Newsgroups1",
    "Thioredoxin",
]

def dataset_path(name: str) -> Path:
    """Devuelve la ruta al archivo .arff de un dataset."""
    return DATA_DIR / f"{name}.arff"


# --- Escaladores (2 niveles) ---
SCALERS: Dict[str, str] = {
    "MinMaxScaler":   "MinMaxScaler",
    "StandardScaler": "StandardScaler",
}

def get_scaler(name: str):
    """Instancia y devuelve un scaler por nombre.

    Args:
        name: 'MinMaxScaler' o 'StandardScaler'.

    Returns:
        Instancia del scaler correspondiente.
    """
    from miclustering.preprocessing.scaler import MinMaxScaler, StandardScaler
    if name == "MinMaxScaler":
        return MinMaxScaler()
    elif name == "StandardScaler":
        return StandardScaler()
    else:
        raise ValueError(f"Scaler desconocido: {name}. Opciones: {list(SCALERS.keys())}")


# --- Distancias (6 niveles) ---
# Usamos 'hausdorff' como nombre canónico de la variante max (=hausdorff_max).
DISTANCES: List[str] = [
    "hausdorff",        # = hausdorff_max
    "hausdorff_min",
    "hausdorff_avg",
    "cauchy_schwarz",
    "earth_movers",
    "mahalanobis",
]

def get_distance_func(name: str):
    """Devuelve la función de distancia del registro de MIClustering."""
    from miclustering.distances import DISTANCE_REGISTRY
    if name not in DISTANCE_REGISTRY:
        raise ValueError(f"Distancia '{name}' no registrada. Disponibles: {list(DISTANCE_REGISTRY.keys())}")
    return DISTANCE_REGISTRY[name]


# --- Modelos (5 niveles) ---
MODELS: List[str] = [
    "midbscan",
    "cosmic",
    "mikmeans",
    "mikmedoids",
    "miknn",
]

# Modelos no supervisados (evaluación transductiva, dataset completo)
UNSUPERVISED_MODELS = ["midbscan", "cosmic", "mikmeans", "mikmedoids"]

# Modelo supervisado (evaluación con CV estratificada)
SUPERVISED_MODELS = ["miknn"]



# Optuna — Configuración de la búsqueda de hiperparámetros


N_TRIALS_DEFAULT = 80       # Presupuesto idéntico para los 5 modelos (Fase 2, punto 3)
OPTUNA_SAMPLER_SEED = 42    # Mismo sampler/seed para todos (fairness)


def get_hyperparameter_space(
    model_name: str,
    trial,  # optuna.trial.Trial
    n_bags: int,
    dist_percentiles: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Define el espacio de búsqueda de Optuna para cada modelo.

    Los rangos se adaptan al dataset concreto usando n_bags (derivado de analisis_datasets.py)
    y los percentiles reales de la distribución de distancias.

    Args:
        model_name: Nombre del modelo ('midbscan', 'cosmic', etc.).
        trial: Objeto optuna.trial.Trial para suggest_*.
        n_bags: Número de bolsas del dataset (para acotar k y min_pts).
        dist_percentiles: Dict con percentiles ('p5', 'p10', 'p50', 'p60', etc.).
                         Si None, se usan rangos conservadores por defecto.

    Returns:
        Dict con los hiperparámetros sugeridos por el trial.
    """
    import math

    # Rango adaptativo de epsilon basado en percentiles de distancias
    if dist_percentiles is not None and dist_percentiles:
        raw_p5 = dist_percentiles.get("p5", 0.1)
        raw_p60 = dist_percentiles.get("p60", 10.0)

        # Si p5 es 0 (ej. bolsas con instancias idénticas), buscar el primer percentil positivo
        if raw_p5 <= 0.0:
            raw_p5 = dist_percentiles.get("p10", 0.0)
        if raw_p5 <= 0.0:
            raw_p5 = dist_percentiles.get("p25", 0.0)

        # Asegurar suelo estrictamente positivo para log=True
        eps_low = max(float(raw_p5), 1e-4)
        eps_high = max(float(raw_p60), eps_low * 2.0, 0.01)
    else:
        eps_low, eps_high = 0.1, 20.0

    # Rango de k basado en √N (heurística de analisis_datasets.py) acotado por n_bags
    k_max = max(2, min(int(math.sqrt(n_bags)), max(2, n_bags - 1)))
    min_pts_max = min(15, max(2, n_bags - 1))
    k_knn_max = min(15, max(1, n_bags - 2))

    if model_name == "midbscan":
        return {
            "epsilon": trial.suggest_float("epsilon", eps_low, eps_high, log=True),
            "min_pts": trial.suggest_int("min_pts", 2, min_pts_max),
        }

    elif model_name == "cosmic":
        epsilon = trial.suggest_float("epsilon", eps_low, eps_high, log=True)
        # epsilon_prime <= epsilon (Fase 2 del protocolo)
        eps_prime_low = min(eps_low, epsilon * 0.95)
        if eps_prime_low >= epsilon:
            epsilon_prime = epsilon
        else:
            epsilon_prime = trial.suggest_float("epsilon_prime", eps_prime_low, epsilon)
        return {
            "epsilon": epsilon,
            "min_pts": trial.suggest_int("min_pts", 2, min_pts_max),
            "epsilon_prime": epsilon_prime,
        }

    elif model_name == "mikmeans":
        return {
            "k": trial.suggest_int("k", 2, k_max),
        }

    elif model_name == "mikmedoids":
        return {
            "k": trial.suggest_int("k", 2, k_max),
        }

    elif model_name == "miknn":
        return {
            "k": trial.suggest_int("k", 1, k_knn_max),
        }

    else:
        raise ValueError(f"Modelo desconocido: {model_name}")



# Utilidades de naming y persistencia


def distance_matrix_filename(dataset: str, scaler: str, distance: str) -> str:
    """Nombre canónico para una matriz de distancias cacheada (.npy)."""
    return f"{dataset}_{scaler}_{distance}.npy"


def distance_matrix_path(dataset: str, scaler: str, distance: str) -> Path:
    """Ruta completa a una matriz de distancias cacheada."""
    return RESULTS_DISTANCIAS / distance_matrix_filename(dataset, scaler, distance)


def optuna_study_name(dataset: str, model: str) -> str:
    """Nombre del estudio Optuna para un par (dataset, modelo)."""
    return f"{dataset}_{model}"


def optuna_db_path(dataset: str, model: str) -> Path:
    """Ruta al archivo SQLite del estudio Optuna."""
    return RESULTS_ESTUDIOS_OPTUNA / f"{optuna_study_name(dataset, model)}.db"


def evaluation_result_path(dataset: str, model: str, seed_idx: int) -> Path:
    """Ruta al JSON de resultados de evaluación final por réplica."""
    return RESULTS_EVALUACION_FINAL / f"{dataset}_{model}_seed{seed_idx}.json"


def ensure_result_dirs():
    """Crea todos los directorios de resultados si no existen."""
    for d in ALL_RESULT_DIRS:
        d.mkdir(parents=True, exist_ok=True)



# Logging


import logging

def setup_logging(phase_name: str = "experiment", level: int = logging.INFO) -> logging.Logger:
    """Configura logging con salida a consola y archivo en resultados/logs/.

    Args:
        phase_name: Nombre de la fase para el archivo de log.
        level: Nivel de logging.

    Returns:
        Logger configurado.
    """
    ensure_result_dirs()

    logger = logging.getLogger(f"mil_experiment.{phase_name}")
    logger.setLevel(level)

    # Evitar handlers duplicados si se llama varias veces
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s — %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Consola
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Archivo
    log_file = RESULTS_LOGS / f"{phase_name}.log"
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
