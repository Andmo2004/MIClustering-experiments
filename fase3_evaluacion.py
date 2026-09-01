"""
fase3_evaluacion.py — Fase 3: Evaluación final con réplicas.

Toma la mejor configuración (escalado, distancia, hiperparámetros) hallada en
la Fase 2 para cada (dataset, modelo) y re-ejecuta el modelo con esa
configuración fija bajo las r semillas de réplica.

  - No supervisados (COSMIC, MIDBSCAN, MIKMEANS, MIKMEDOIDS): evaluación
    transductiva sobre el dataset completo, reutilizando la matriz de distancias
    cacheada en la Fase 1.
  - MIKNN (supervisado): validación cruzada estratificada (5-fold) con recálculo
    de submatrices de distancia por fold (protocolo, Fase 1, punto 4).

Las r semillas controlan SOLO la parte estocástica del algoritmo (inicialización
de centroides/medoides, orden de procesamiento). NO se re-optimizan hiperparámetros.

Uso:
    python fase3_evaluacion.py                                  # Completo
    python fase3_evaluacion.py --datasets musk1 --models mikmeans  # Subconjunto
    python fase3_evaluacion.py --n-replicas 3                   # Menos réplicas (test)
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

from config import (
    DATASETS,
    MODELS,
    UNSUPERVISED_MODELS,
    SUPERVISED_MODELS,
    N_REPLICAS,
    REPLICA_SEEDS,
    MASTER_SEED,
    RESULTS_EVALUACION_FINAL,
    RESULTS_ESTUDIOS_OPTUNA,
    RESULTS_LOGS,
    dataset_path,
    distance_matrix_path,
    optuna_study_name,
    optuna_db_path,
    evaluation_result_path,
    get_scaler,
    get_distance_func,
    ensure_result_dirs,
    setup_logging,
    derive_seeds,
)
from time_estimator import estimate_fase3, print_phase_header

from miclustering.data.arff_reader import ArffToMIData
from miclustering.data.midata import MIData
from miclustering.data.utils import parse_label
from miclustering.evaluation.bcm import MILEvaluator
from miclustering.distances.distance_matrix import compute_distance_matrix

logger = setup_logging("fase3")


# ═══════════════════════════════════════════════════════════════════════════
# Carga de mejores configuraciones de la Fase 2
# ═══════════════════════════════════════════════════════════════════════════

def load_best_config(dataset_name: str, model_name: str) -> Dict[str, Any]:
    """Carga la mejor configuración del estudio Optuna de la Fase 2.

    Returns:
        Dict con keys: 'scaler', 'distance', y los hiperparámetros del modelo.
    """
    import optuna

    db_path = optuna_db_path(dataset_name, model_name)
    if not db_path.exists():
        raise FileNotFoundError(
            f"Estudio Optuna no encontrado: {db_path}. "
            f"¿Ejecutaste la Fase 2 primero?"
        )

    storage = f"sqlite:///{db_path}"
    study_name = optuna_study_name(dataset_name, model_name)
    study = optuna.load_study(study_name=study_name, storage=storage)

    best = study.best_trial
    params = dict(best.params)

    # Separar scaler y distance del resto de hiperparámetros
    config = {
        "scaler": params.pop("scaler"),
        "distance": params.pop("distance"),
        "best_score": best.value,
        "best_trial": best.number,
        "model_params": params,
    }

    logger.info(
        f"[{dataset_name}/{model_name}] Mejor config: "
        f"scaler={config['scaler']}, dist={config['distance']}, "
        f"params={config['model_params']}, score={config['best_score']:.4f}"
    )

    return config


# ═══════════════════════════════════════════════════════════════════════════
# Instanciación de modelos
# ═══════════════════════════════════════════════════════════════════════════

def instantiate_model(model_name: str, params: Dict[str, Any], metric: str, seed: int):
    """Instancia un modelo de MIClustering con los hiperparámetros dados."""
    from miclustering.models.midbscan import MIDBSCAN
    from miclustering.models.cosmic import COSMIC
    from miclustering.models.mikmeans import MIKMeans
    from miclustering.models.mikmedoids import MIKMedoids
    from miclustering.models.miknn import MIKnn

    if model_name == "midbscan":
        return MIDBSCAN(
            epsilon=params["epsilon"],
            min_pts=params["min_pts"],
            metric=metric,
            n_jobs=1, device="cpu",
        )
    elif model_name == "cosmic":
        return COSMIC(
            epsilon=params["epsilon"],
            min_pts=params["min_pts"],
            epsilon_prime=params.get("epsilon_prime"),
            metric=metric,
            n_jobs=1, device="cpu",
        )
    elif model_name == "mikmeans":
        return MIKMeans(
            k=params["k"],
            metric=metric,
            random_state=seed,
            n_jobs=1, device="cpu",
        )
    elif model_name == "mikmedoids":
        return MIKMedoids(
            k=params["k"],
            metric=metric,
            random_state=seed,
            n_jobs=1, device="cpu",
        )
    elif model_name == "miknn":
        return MIKnn(
            k=params["k"],
            metric=metric,
            n_jobs=1, device="cpu",
        )
    else:
        raise ValueError(f"Modelo desconocido: {model_name}")


# ═══════════════════════════════════════════════════════════════════════════
# Evaluación no supervisada (transductiva)
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_unsupervised_replica(
    dataset_name: str,
    model_name: str,
    config: Dict[str, Any],
    seed: int,
    seed_idx: int,
) -> Dict[str, Any]:
    """Ejecuta una réplica de un modelo no supervisado sobre el dataset completo.

    Evaluación transductiva: se usa el dataset completo (sin split train/test).
    Se reutiliza la matriz de distancias precomputada de la Fase 1.
    """
    scaler_name = config["scaler"]
    distance_name = config["distance"]
    model_params = config["model_params"]

    # Cargar dataset y escalar
    dataset = ArffToMIData.from_arff(dataset_path(dataset_name))
    scaler = get_scaler(scaler_name)
    scaled_dataset = scaler.fit_transform(dataset)

    # Cargar matriz de distancias precomputada
    dist_matrix = np.load(distance_matrix_path(dataset_name, scaler_name, distance_name))

    # Instanciar y ejecutar modelo
    t0 = time.perf_counter()
    model = instantiate_model(model_name, model_params, metric=distance_name, seed=seed)
    model.fit(scaled_dataset)
    fit_time = time.perf_counter() - t0

    labels = model.labels

    # Métricas externas (usando MILEvaluator con asignación húngara)
    # Silenciar la salida de MILEvaluator.evaluate
    import io
    import contextlib
    f_buf = io.StringIO()
    with contextlib.redirect_stdout(f_buf):
        external_metrics = MILEvaluator.evaluate(
            scaled_dataset, labels,
            title=f"{model_name} ({dataset_name}, seed={seed})"
        )

    # Estadísticas de clustering
    label_arr = np.array([labels.get(bag.bag_id, -1) for bag in scaled_dataset.bags])
    n_clusters = len(np.unique(label_arr[label_arr >= 0]))
    noise_count = int(np.sum(label_arr < 0))
    noise_pct = round(100.0 * noise_count / len(label_arr), 2) if len(label_arr) > 0 else 0.0

    result = {
        "dataset": dataset_name,
        "model": model_name,
        "seed_idx": seed_idx,
        "seed": seed,
        "scaler": scaler_name,
        "distance": distance_name,
        "model_params": model_params,
        "fit_time_sec": round(fit_time, 4),
        "n_clusters": n_clusters,
        "noise_count": noise_count,
        "noise_pct": noise_pct,
        **external_metrics,
    }

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Evaluación supervisada (MIKNN con CV estratificada)
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_supervised_replica(
    dataset_name: str,
    model_name: str,
    config: Dict[str, Any],
    seed: int,
    seed_idx: int,
) -> Dict[str, Any]:
    """Ejecuta una réplica de MIKNN con validación cruzada estratificada.

    En cada fold:
    - El escalador se ajusta SOLO con las bolsas de entrenamiento.
    - Las submatrices de distancia se recalculan por fold.
    - NO se usa la matriz completa de la Fase 1 (protocolo, Fase 3).
    """
    scaler_name = config["scaler"]
    distance_name = config["distance"]
    model_params = config["model_params"]

    # Cargar dataset
    dataset = ArffToMIData.from_arff(dataset_path(dataset_name))
    bags = dataset.bags

    # Extraer labels para estratificación
    labels = []
    for bag in bags:
        lv = parse_label(bag.label) if isinstance(bag.label, (str, float)) else int(bag.label)
        labels.append(lv)
    labels = np.array(labels)

    # CV estratificada (la semilla controla el shuffle de folds)
    min_class_count = int(np.bincount(labels.astype(int)).min())
    n_folds = max(2, min(5, min_class_count))

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_metrics = []
    total_time = 0.0

    bag_ids_list: List[str] = [bag.bag_id for bag in bags]
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(bag_ids_list, labels)):
        train_bags = [bags[i] for i in train_idx]
        test_bags = [bags[i] for i in test_idx]

        train_data = MIData(train_bags, f"{dataset_name}_train_f{fold_idx}")
        test_data = MIData(test_bags, f"{dataset_name}_test_f{fold_idx}")

        # Escalar: fit SOLO con train (protocolo Fase 3)
        scaler = get_scaler(scaler_name)
        train_scaled = scaler.fit_transform(train_data)
        test_scaled = scaler.transform(test_data)

        # Ejecutar modelo
        t0 = time.perf_counter()
        metric_func = get_distance_func(distance_name)
        model = instantiate_model(model_name, model_params, metric=distance_name, seed=seed)
        model.fit(train_scaled)
        preds = model.predict(test_scaled)
        fold_time = time.perf_counter() - t0
        total_time += fold_time

        # Métricas externas
        import io, contextlib
        f_buf = io.StringIO()
        with contextlib.redirect_stdout(f_buf):
            metrics = MILEvaluator.evaluate(
                test_scaled, preds,
                title=f"MIKNN fold {fold_idx}"
            )

        fold_metrics.append(metrics)

    # Promediar métricas sobre folds
    avg_metrics = {}
    if fold_metrics:
        for key in fold_metrics[0]:
            values = [fm[key] for fm in fold_metrics if key in fm]
            if values and isinstance(values[0], (int, float)):
                avg_metrics[key] = float(np.mean(values))
                avg_metrics[f"{key}_std"] = float(np.std(values))

    result = {
        "dataset": dataset_name,
        "model": model_name,
        "seed_idx": seed_idx,
        "seed": seed,
        "scaler": scaler_name,
        "distance": distance_name,
        "model_params": model_params,
        "n_folds": n_folds,
        "fit_time_sec": round(total_time, 4),
        "n_clusters": 2,  # MIKNN es clasificador, siempre 2 clases
        "noise_count": 0,
        "noise_pct": 0.0,
        **avg_metrics,
    }

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline principal
# ═══════════════════════════════════════════════════════════════════════════

def run_fase3(
    dataset_names: Optional[List[str]] = None,
    model_names: Optional[List[str]] = None,
    n_replicas: int = N_REPLICAS,
    force: bool = False,
) -> None:
    """Ejecuta la Fase 3: evaluación final con réplicas.

    Args:
        dataset_names: Datasets a evaluar (None = todos).
        model_names: Modelos a evaluar (None = todos).
        n_replicas: Número de réplicas (semillas).
        force: Recalcular aunque ya existan resultados.
    """
    datasets = dataset_names or DATASETS
    models = model_names or MODELS

    # Derivar semillas si n_replicas difiere del default
    if n_replicas != N_REPLICAS:
        seeds = derive_seeds(MASTER_SEED, n_replicas)
    else:
        seeds = REPLICA_SEEDS[:n_replicas]

    ensure_result_dirs()

    total = len(datasets) * len(models) * n_replicas

    # Estimación de tiempo
    _, _, time_str, details = estimate_fase3(
        dataset_names=datasets,
        model_names=models,
        n_replicas=n_replicas,
        force=force,
    )
    print_phase_header(
        phase_title="FASE 3 — EVALUACIÓN FINAL CON RÉPLICAS",
        estimated_time_str=time_str,
        details=details,
        logger=logger,
    )

    all_results = []
    errors = []

    with tqdm(total=total, desc="Fase 3", ncols=80) as pbar:
        for ds_name in datasets:
            for model_name in models:
                # Cargar mejor configuración de Fase 2
                try:
                    config = load_best_config(ds_name, model_name)
                except FileNotFoundError as e:
                    logger.error(f"Skipping {ds_name}/{model_name}: {e}")
                    errors.append({
                        "dataset": ds_name, "model": model_name,
                        "error": str(e),
                    })
                    pbar.update(n_replicas)
                    continue

                for seed_idx, seed in enumerate(seeds):
                    pbar.set_postfix_str(
                        f"{ds_name[:10]}/{model_name[:8]}/s{seed_idx}"
                    )

                    # ¿Ya calculado?
                    out_path = evaluation_result_path(ds_name, model_name, seed_idx)
                    if out_path.exists() and not force:
                        with open(out_path) as f:
                            result = json.load(f)
                        all_results.append(result)
                        pbar.update(1)
                        continue

                    try:
                        if model_name in UNSUPERVISED_MODELS:
                            result = evaluate_unsupervised_replica(
                                ds_name, model_name, config, seed, seed_idx
                            )
                        else:
                            result = evaluate_supervised_replica(
                                ds_name, model_name, config, seed, seed_idx
                            )

                        # Guardar JSON individual
                        with open(out_path, "w", encoding="utf-8") as f:
                            json.dump(result, f, indent=2, default=str, ensure_ascii=False)

                        all_results.append(result)

                    except Exception as e:
                        logger.error(
                            f"Error en {ds_name}/{model_name}/seed{seed_idx}: {e}",
                            exc_info=True,
                        )
                        errors.append({
                            "dataset": ds_name,
                            "model": model_name,
                            "seed_idx": seed_idx,
                            "error": str(e),
                        })

                    pbar.update(1)

    # Guardar resumen consolidado en CSVs
    if all_results:
        # Aplanar model_params si existe como dict en el dataframe
        rows = []
        for r in all_results:
            row = dict(r)
            if "model_params" in row and isinstance(row["model_params"], dict):
                mp = row.pop("model_params")
                for k, v in mp.items():
                    row[f"param_{k}"] = v
            rows.append(row)

        df = pd.DataFrame(rows)
        csv_path_logs = RESULTS_LOGS / "fase3_resultados_completos.csv"
        csv_path_eval = RESULTS_EVALUACION_FINAL / "fase3_evaluaciones_detalladas.csv"
        df.to_csv(csv_path_logs, index=False)
        df.to_csv(csv_path_eval, index=False)
        logger.info(f"Resultados completos guardados en {csv_path_logs} y {csv_path_eval}")

        # Resumen por (dataset, modelo): media ± std de todas las métricas numéricas disponibles
        metric_cols = [c for c in ["F1-Score", "Precision", "Recall", "Specificity", "fit_time_sec", "noise_pct", "n_clusters"] if c in df.columns]
        if metric_cols:
            agg_dict = {}
            for col in metric_cols:
                agg_dict[f"{col}_mean"] = (col, "mean")
                agg_dict[f"{col}_std"] = (col, "std")
            agg_dict["n_replicas"] = ("seed_idx", "count")

            summary = df.groupby(["dataset", "model", "scaler", "distance"]).agg(**agg_dict).reset_index()

            print("\n  Resumen (media ± std sobre réplicas):")
            print(summary.to_string(index=False))

            summary.to_csv(RESULTS_LOGS / "fase3_resumen.csv", index=False)
            summary.to_csv(RESULTS_EVALUACION_FINAL / "fase3_resumen_dataset_modelo.csv", index=False)

            # Resumen agregado global por modelo
            global_model_summary = df.groupby("model").agg(
                F1_mean=("F1-Score", "mean"),
                F1_std=("F1-Score", "std"),
                Precision_mean=("Precision", "mean") if "Precision" in df.columns else ("model", "count"),
                Recall_mean=("Recall", "mean") if "Recall" in df.columns else ("model", "count"),
                Tiempo_mean=("fit_time_sec", "mean") if "fit_time_sec" in df.columns else ("model", "count"),
                total_evaluaciones=("seed_idx", "count"),
            ).reset_index()
            global_model_summary.to_csv(RESULTS_EVALUACION_FINAL / "fase3_resumen_global_modelos.csv", index=False)

    if errors:
        print(f"\n  ⚠ Errores: {len(errors)}")
        for e in errors[:10]:
            print(f"    - {e.get('dataset')}/{e.get('model')}: {e.get('error')}")

    print("\n" + "═" * 75)
    print("  FASE 3 COMPLETADA")
    print(f"  Resultados en: {RESULTS_EVALUACION_FINAL}")
    print("═" * 75 + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Fase 3: Evaluación final con réplicas.",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help=f"Datasets a evaluar (default: todos). Opciones: {DATASETS}",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help=f"Modelos a evaluar (default: todos). Opciones: {MODELS}",
    )
    parser.add_argument(
        "--n-replicas", type=int, default=N_REPLICAS,
        help=f"Número de réplicas (default: {N_REPLICAS}).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Recalcular aunque ya existan resultados.",
    )

    args = parser.parse_args()

    run_fase3(
        dataset_names=args.datasets,
        model_names=args.models,
        n_replicas=args.n_replicas,
        force=args.force,
    )


if __name__ == "__main__":
    main()
