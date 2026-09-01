"""
fase0_setup.py — Fase 0: Preparación y control de reproducibilidad.

Antes de ejecutar cualquier fase del protocolo, este script:
  1. Verifica y documenta versiones de dependencias.
  2. Calcula checksums SHA-256 de los 10 datasets .arff.
  3. Crea la estructura de directorios de resultados.
  4. Documenta las semillas derivadas y decisiones de diseño.

Todo se persiste en resultados/logs/ para trazabilidad.

Uso:
    python fase0_setup.py
"""

import hashlib
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict
import pandas as pd

from config import (
    PROJECT_ROOT,
    DATA_DIR,
    MASTER_SEED,
    N_REPLICAS,
    N_TRIALS_DEFAULT,
    REPLICA_SEEDS,
    DATASETS,
    DISTANCES,
    SCALERS,
    MODELS,
    RESULTS_ROOT,
    RESULTS_LOGS,
    dataset_path,
    ensure_result_dirs,
    setup_logging,
)
from time_estimator import estimate_fase0, print_phase_header

logger = setup_logging("fase0")



# 1. Verificar y documentar versiones


def document_versions() -> Dict[str, str]:
    """Recoge las versiones de todas las dependencias relevantes."""
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }

    libs = [
        "numpy", "scipy", "sklearn", "pandas", "optuna",
        "torch", "pot", "joblib", "tqdm",
        "matplotlib", "seaborn",
    ]

    for lib_name in libs:
        try:
            if lib_name == "sklearn":
                import sklearn
                versions["scikit-learn"] = sklearn.__version__
            elif lib_name == "pot":
                import ot
                versions["POT"] = ot.__version__
            else:
                mod = __import__(lib_name)
                versions[lib_name] = mod.__version__
        except ImportError:
            versions[lib_name] = "NO INSTALADO"
        except AttributeError:
            versions[lib_name] = "versión no disponible"

    # Versión de MIClustering (intentar obtener del pyproject.toml o __init__)
    try:
        import miclustering
        versions["miclustering"] = getattr(miclustering, "__version__", "dev")
    except ImportError:
        versions["miclustering"] = "NO IMPORTABLE"

    # Git commit hash
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            versions["git_commit"] = result.stdout.strip()
        else:
            versions["git_commit"] = "no git repo"
    except Exception:
        versions["git_commit"] = "no disponible"

    return versions



# 2. Checksums SHA-256 de datasets


def compute_checksums() -> Dict[str, str]:
    """Calcula SHA-256 de cada archivo .arff para trazabilidad."""
    checksums = {}

    for ds_name in DATASETS:
        path = dataset_path(ds_name)
        if not path.exists():
            logger.warning(f"Dataset no encontrado: {path}")
            checksums[ds_name] = "FILE_NOT_FOUND"
            continue

        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        checksums[ds_name] = sha256.hexdigest()
        logger.info(f"[{ds_name}] SHA-256: {checksums[ds_name][:16]}...")

    return checksums



# 3. Documentar el diseño experimental


def document_design() -> Dict:
    """Genera un documento JSON con todas las decisiones de diseño."""
    return {
        "protocolo": "Comparación de Algoritmos MIL bajo Distintas Métricas de Distancia",
        "timestamp": datetime.now().isoformat(),

        # Reproducibilidad
        "master_seed": MASTER_SEED,
        "n_replicas": N_REPLICAS,
        "replica_seeds": REPLICA_SEEDS,

        # Diseño factorial
        "datasets": DATASETS,
        "scalers": list(SCALERS.keys()),
        "distances": DISTANCES,
        "models": MODELS,
        "n_trials_optuna": N_TRIALS_DEFAULT,

        # Espacio combinatorio
        "n_datasets": len(DATASETS),
        "n_scalers": len(SCALERS),
        "n_distances": len(DISTANCES),
        "n_models": len(MODELS),
        "celdas_experimentales": len(DATASETS) * len(SCALERS) * len(DISTANCES) * len(MODELS),
        "estudios_optuna": len(DATASETS) * len(MODELS),

        # Decisiones de diseño (pre-registradas ANTES de ver resultados)
        "decisiones_preregistradas": {
            "tratamiento_ruido": (
                "Instancias de ruido (label=-1) de COSMIC/MIDBSCAN se EXCLUYEN del cálculo "
                "de métricas externas, pero su porcentaje se reporta como variable diagnóstica."
            ),
            "tratamiento_k": (
                "La asignación húngara admite matrices de confusión rectangulares. "
                "No se fuerza k = número real de clases."
            ),
            "desempate_optuna": (
                "Si dos configuraciones producen el mismo score de Silhouette, "
                "se prioriza la de menor tiempo de cómputo."
            ),
            "escalado_no_supervisados": (
                "Para los 4 modelos no supervisados (evaluación transductiva), "
                "ajustar el escalado sobre el dataset completo es válido y no constituye fuga."
            ),
            "escalado_miknn": (
                "Para MIKNN (supervisado), en la evaluación final el escalado se ajusta "
                "SOLO con las bolsas de entrenamiento de cada fold. En la búsqueda de "
                "hiperparámetros se permite la matriz completa como aproximación."
            ),
        },
    }



# Pipeline principal


def run_fase0():
    """Ejecuta la Fase 0 completa."""
    _, _, time_str, details = estimate_fase0()
    print_phase_header(
        phase_title="FASE 0 — PREPARACIÓN Y CONTROL DE REPRODUCIBILIDAD",
        estimated_time_str=time_str,
        details=details,
        logger=logger,
    )

    # 1. Crear directorios
    print("\n[1/4] Creando estructura de directorios...")
    ensure_result_dirs()
    # Directorio adicional para análisis de datasets
    (RESULTS_ROOT / "analisis_datasets").mkdir(parents=True, exist_ok=True)
    logger.info("Directorios de resultados creados.")

    # 2. Versiones
    print("[2/4] Documentando versiones de dependencias...")
    versions = document_versions()
    versions_path = RESULTS_LOGS / "versiones.json"
    with open(versions_path, "w", encoding="utf-8") as f:
        json.dump(versions, f, indent=2, ensure_ascii=False)
    # Exportar también a CSV
    df_versions = pd.DataFrame(list(versions.items()), columns=["componente", "version"])
    df_versions.to_csv(RESULTS_LOGS / "versiones.csv", index=False)
    logger.info(f"Versiones guardadas en {versions_path} y {RESULTS_LOGS / 'versiones.csv'}")

    print("  Versiones clave:")
    for k, v in versions.items():
        print(f"    {k}: {v}")

    # 3. Checksums
    print("\n[3/4] Calculando checksums SHA-256 de datasets...")
    checksums = compute_checksums()
    checksums_path = RESULTS_LOGS / "checksums.json"
    with open(checksums_path, "w", encoding="utf-8") as f:
        json.dump(checksums, f, indent=2)
    # Exportar también a CSV
    df_checksums = pd.DataFrame(list(checksums.items()), columns=["dataset", "sha256"])
    df_checksums.to_csv(RESULTS_LOGS / "checksums.csv", index=False)
    logger.info(f"Checksums guardados en {checksums_path} y {RESULTS_LOGS / 'checksums.csv'}")

    missing = [k for k, v in checksums.items() if v == "FILE_NOT_FOUND"]
    if missing:
        print(f"  ⚠ Datasets no encontrados: {missing}")
    else:
        print(f"  ✓ Los 10 datasets verificados con SHA-256.")

    # 4. Diseño experimental
    print("\n[4/4] Documentando diseño experimental...")
    design = document_design()
    design_path = RESULTS_LOGS / "diseno_experimental.json"
    with open(design_path, "w", encoding="utf-8") as f:
        json.dump(design, f, indent=2, ensure_ascii=False)
    logger.info(f"Diseño guardado en {design_path}")

    print(f"  Semilla maestra: {MASTER_SEED}")
    print(f"  Réplicas: {N_REPLICAS}")
    print(f"  Semillas derivadas: {REPLICA_SEEDS}")
    print(f"  Celdas experimentales: {design['celdas_experimentales']}")
    print(f"  Estudios Optuna: {design['estudios_optuna']}")

    # Resumen final
    print("\n" + "═" * 75)
    print("  FASE 0 COMPLETADA")
    print(f"  Artefactos en: {RESULTS_LOGS}")
    print("═" * 75 + "\n")


if __name__ == "__main__":
    run_fase0()
