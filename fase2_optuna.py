"""
fase2_optuna.py — Fase 2: Optimización de hiperparámetros con Optuna.

Para cada par (dataset, modelo) = 10×5 = 50 estudios, ejecuta una búsqueda
bayesiana (TPE) con presupuesto idéntico (N_TRIALS) y misma semilla del
sampler, incluyendo scaler y distancia como hiperparámetros categóricos.

Función objetivo:
  - No supervisados: Silhouette score sobre la matriz de distancias precomputada.
  - MIKNN: F1 macro vía validación cruzada estratificada (usando matriz completa
    de la Fase 1 como aproximación de eficiencia).

Uso:
    python fase2_optuna.py                                # Todos los 50 estudios
    python fase2_optuna.py --datasets musk1 --models mikmeans  # Subconjunto
    python fase2_optuna.py --n-trials 20                  # Menos trials (test)
    python fase2_optuna.py --resume                       # Continuar estudios existentes
"""

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from config import (
    DATASETS,
    DISTANCES,
    SCALERS,
    MODELS,
    UNSUPERVISED_MODELS,
    SUPERVISED_MODELS,
    N_TRIALS_DEFAULT,
    EARLY_STOPPING_DEFAULT,
    OPTUNA_SAMPLER_SEED,
    MASTER_SEED,
    RESULTS_ESTUDIOS_OPTUNA,
    RESULTS_DISTANCIAS,
    RESULTS_LOGS,
    dataset_path,
    distance_matrix_path,
    optuna_study_name,
    optuna_db_path,
    get_scaler,
    get_distance_func,
    get_hyperparameter_space,
    ensure_result_dirs,
    setup_logging,
)
from time_estimator import estimate_fase2, print_phase_header

from miclustering.data.arff_reader import ArffToMIData
from miclustering.data.midata import MIData
from miclustering.data.utils import parse_label
from miclustering.distances.distance_matrix import compute_distance_matrix

# Silhouette precomputed de sklearn (la implementación propia se añadirá después)
from sklearn.metrics import silhouette_score, f1_score
from sklearn.model_selection import StratifiedKFold, KFold

logger = setup_logging("fase2")

# Reducir verbosidad de Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)



# Funciones auxiliares


def load_distance_matrix(dataset_name: str, scaler_name: str, distance_name: str) -> np.ndarray:
    """Carga una matriz de distancias precomputada de la Fase 1."""
    path = distance_matrix_path(dataset_name, scaler_name, distance_name)
    if not path.exists():
        raise FileNotFoundError(
            f"Matriz de distancias no encontrada: {path}. "
            f"¿Ejecutaste la Fase 1 primero?"
        )
    return np.load(path)


def get_distance_percentiles(dataset_name: str, scaler_name: str, distance_name: str) -> Dict[str, float]:
    """Calcula percentiles de la distribución de distancias para informar rangos de epsilon."""
    matrix = load_distance_matrix(dataset_name, scaler_name, distance_name)
    n = matrix.shape[0]
    upper_tri = matrix[np.triu_indices(n, k=1)]

    if len(upper_tri) == 0:
        return {}

    percentiles = {
        "p5": float(np.percentile(upper_tri, 5)),
        "p10": float(np.percentile(upper_tri, 10)),
        "p25": float(np.percentile(upper_tri, 25)),
        "p50": float(np.percentile(upper_tri, 50)),
        "p60": float(np.percentile(upper_tri, 60)),
        "p75": float(np.percentile(upper_tri, 75)),
        "p90": float(np.percentile(upper_tri, 90)),
        "p95": float(np.percentile(upper_tri, 95)),
    }
    return percentiles


def instantiate_model(model_name: str, params: Dict[str, Any], metric: str = "hausdorff", seed: int = 42):
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
            n_jobs=1,
            device="cpu",
        )

    elif model_name == "cosmic":
        return COSMIC(
            epsilon=params["epsilon"],
            min_pts=params["min_pts"],
            epsilon_prime=params.get("epsilon_prime"),
            metric=metric,
            n_jobs=1,
            device="cpu",
        )

    elif model_name == "mikmeans":
        return MIKMeans(
            k=params["k"],
            metric=metric,
            max_iters=30,
            tol=0.03,
            random_state=seed,
            n_jobs=1,
            device="cpu",
        )

    elif model_name == "mikmedoids":
        return MIKMedoids(
            k=params["k"],
            metric=metric,
            random_state=seed,
            n_jobs=1,
            device="cpu",
        )

    elif model_name == "miknn":
        return MIKnn(
            k=params["k"],
            metric=metric,
            n_jobs=1,
            device="cpu",
        )

    else:
        raise ValueError(f"Modelo desconocido: {model_name}")


def fit_unsupervised_with_precomputed(model, dataset: MIData, dist_matrix: np.ndarray) -> Dict[str, int]:
    """Ejecuta un modelo no supervisado usando la matriz precomputada (Fase 1) si es soportada."""
    from miclustering.models.midbscan import MIDBSCAN
    from miclustering.models.cosmic import COSMIC
    from miclustering.models.mikmedoids import MIKMedoids

    if isinstance(model, (MIDBSCAN, COSMIC, MIKMedoids)):
        model.fit(dataset, precomputed_matrix=dist_matrix)
    else:
        model.fit(dataset)
    return model.labels


def evaluate_unsupervised_silhouette(
    dist_matrix: np.ndarray,
    labels_dict: Dict[str, int],
    bag_ids: List[str],
) -> float:
    """Calcula Silhouette score sobre la matriz de distancias precomputada.

    Puntos de ruido (-1) se excluyen del cálculo.

    Returns:
        Silhouette score en [-1, 1]. Si hay < 2 clusters, retorna -1.
    """
    # Alinear labels con el orden de la matriz
    label_arr = np.array([labels_dict.get(bid, -1) for bid in bag_ids])

    # Filtrar ruido
    valid_mask = label_arr >= 0
    n_valid = valid_mask.sum()

    if n_valid < 2:
        return -1.0

    # Clusters únicos (sin ruido)
    unique_labels = np.unique(label_arr[valid_mask])
    if len(unique_labels) < 2:
        return -1.0

    # Submatriz sin ruido
    valid_indices = np.where(valid_mask)[0]
    sub_matrix = dist_matrix[np.ix_(valid_indices, valid_indices)]
    sub_labels = label_arr[valid_indices]

    try:
        from typing import cast
        return float(silhouette_score(cast(Any, sub_matrix), sub_labels, metric="precomputed"))
    except Exception:
        return -1.0



# Función objetivo por tipo de modelo


def create_unsupervised_objective(
    model_name: str,
    dataset_name: str,
    dataset: MIData,
    bag_ids: List[str],
    n_bags: int,
):
    """Crea la función objetivo de Optuna para modelos no supervisados.

    La función objetivo:
    1. Selecciona scaler + distancia como hiperparámetros categóricos.
    2. Carga la matriz de distancias precomputada (Fase 1).
    3. Ajusta el modelo sobre el dataset completo (transductivo).
    4. Calcula Silhouette sobre la matriz precomputada.
    """
    def objective(trial):
        # Hiperparámetros categóricos
        scaler_name = trial.suggest_categorical("scaler", list(SCALERS.keys()))
        distance_name = trial.suggest_categorical("distance", DISTANCES)

        # Cargar matriz precomputada
        try:
            dist_matrix = load_distance_matrix(dataset_name, scaler_name, distance_name)
        except FileNotFoundError as e:
            logger.warning(f"Matriz no encontrada: {e}")
            return -1.0  # Peor score posible

        # Percentiles para rangos adaptativos de epsilon
        dist_percentiles = get_distance_percentiles(dataset_name, scaler_name, distance_name)

        # Hiperparámetros del modelo
        model_params = get_hyperparameter_space(
            model_name, trial, n_bags, dist_percentiles
        )

        # Escalar dataset
        scaler = get_scaler(scaler_name)
        scaled_dataset = scaler.fit_transform(dataset)

        # Ejecutar modelo con 3 sub-semillas para reducir varianza
        scores = []
        sub_seeds = [MASTER_SEED + i for i in range(3)]

        for sub_seed in sub_seeds:
            try:
                model = instantiate_model(model_name, model_params, metric=distance_name, seed=sub_seed)
                labels = fit_unsupervised_with_precomputed(model, scaled_dataset, dist_matrix)
                bag_ids_current = [bag.bag_id for bag in scaled_dataset.bags]

                score = evaluate_unsupervised_silhouette(
                    dist_matrix, labels, bag_ids_current
                )
                scores.append(score)
            except Exception as e:
                logger.debug(f"Error en trial {trial.number} sub-seed {sub_seed}: {e}")
                scores.append(-1.0)

        return float(np.mean(scores))

    return objective


def create_supervised_objective(
    model_name: str,
    dataset_name: str,
    dataset: MIData,
    n_bags: int,
):
    """Crea la función objetivo de Optuna para MIKNN (supervisado).

    Usa F1 macro vía validación cruzada estratificada.
    En esta fase, se usa la matriz completa de la Fase 1 como aproximación
    de eficiencia (ver protocolo, Fase 1, punto 4).
    """
    from miclustering.evaluation.bcm import MILEvaluator

    def objective(trial):
        # Hiperparámetros categóricos
        scaler_name = trial.suggest_categorical("scaler", list(SCALERS.keys()))
        distance_name = trial.suggest_categorical("distance", DISTANCES)

        # Percentiles para rangos adaptativos
        try:
            dist_percentiles = get_distance_percentiles(dataset_name, scaler_name, distance_name)
        except FileNotFoundError:
            return 0.0

        # Hiperparámetros del modelo
        model_params = get_hyperparameter_space(
            model_name, trial, n_bags, dist_percentiles
        )

        # Escalar con dataset completo (aproximación, ver protocolo)
        scaler = get_scaler(scaler_name)
        scaled_dataset = scaler.fit_transform(dataset)

        # Validación cruzada estratificada (5-fold)
        bags = scaled_dataset.bags
        labels = []
        for bag in bags:
            lv = parse_label(bag.label) if isinstance(bag.label, (str, float)) else int(bag.label)
            labels.append(lv)
        labels = np.array(labels)

        min_class_count = int(np.bincount(labels.astype(int)).min())
        bag_ids_list: List[str] = [bag.bag_id for bag in bags]
        if min_class_count >= 2:
            n_folds = min(5, min_class_count)
            splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=MASTER_SEED)
            splits = list(splitter.split(bag_ids_list, labels))
        else:
            logger.warning(
                f"[{dataset_name}] min_class_count={min_class_count} < 2. "
                "Usando KFold no estratificado como fallback para CV (la clase minoritaria puede no estar presente en todos los folds)."
            )
            n_folds = max(2, min(5, len(labels)))
            splitter = KFold(n_splits=n_folds, shuffle=True, random_state=MASTER_SEED)
            splits = list(splitter.split(bag_ids_list))

        fold_scores = []

        for train_idx, test_idx in splits:
            train_bags = [bags[i] for i in train_idx]
            test_bags = [bags[i] for i in test_idx]

            train_data = MIData(train_bags, f"{dataset_name}_train")
            test_data = MIData(test_bags, f"{dataset_name}_test")

            try:
                model = instantiate_model(model_name, model_params, metric=distance_name, seed=MASTER_SEED)
                model.fit(train_data)
                preds = model.predict(test_data)

                # Calcular F1 macro
                y_true = labels[test_idx]
                y_pred = np.array([preds.get(bags[i].bag_id, 0) for i in test_idx])
                f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
                fold_scores.append(f1)
            except Exception as e:
                logger.debug(f"Error en fold: {e}")
                fold_scores.append(0.0)

        return float(np.mean(fold_scores)) if fold_scores else 0.0

    return objective



# Pipeline principal


class EarlyStoppingCallback:
    """Callback de Optuna para early stopping tras N trials consecutivos sin mejora."""

    def __init__(self, patience: int = EARLY_STOPPING_DEFAULT):
        self.patience = patience
        self._best_value: Optional[float] = None
        self._no_improvement_count: int = 0

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if self.patience <= 0:
            return

        # Inicializar el mejor valor histórico del estudio si aún no está fijado
        if self._best_value is None:
            completed_values = [
                t.value
                for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
            ]
            if completed_values:
                self._best_value = max(completed_values)

        # Si el trial actual completó exitosamente y supera el mejor valor histórico
        if (
            trial.state == optuna.trial.TrialState.COMPLETE
            and trial.value is not None
            and (self._best_value is None or trial.value > self._best_value)
        ):
            self._best_value = trial.value
            self._no_improvement_count = 0
        else:
            self._no_improvement_count += 1

        if self._no_improvement_count >= self.patience:
            logger.info(
                f"[{study.study_name}] Early stopping activado: {self._no_improvement_count} "
                f"trials consecutivos sin mejora. Mejor score: {self._best_value:.4f} "
                f"(detenido en trial #{trial.number})."
            )
            study.stop()


def run_single_study(
    dataset_name: str,
    model_name: str,
    n_trials: int = N_TRIALS_DEFAULT,
    early_stopping_patience: int = EARLY_STOPPING_DEFAULT,
    resume: bool = False,
) -> Dict[str, Any]:
    """Ejecuta un estudio Optuna para un par (dataset, modelo).

    Args:
        dataset_name: Nombre del dataset.
        model_name: Nombre del modelo.
        n_trials: Número máximo de trials.
        early_stopping_patience: Corta si transcurren N trials sin mejora (0 = desactiva).
        resume: Si True, continúa un estudio existente.

    Returns:
        Dict con la mejor configuración hallada.
    """
    study_name = optuna_study_name(dataset_name, model_name)
    db_path = optuna_db_path(dataset_name, model_name)
    storage = f"sqlite:///{db_path}"

    logger.info(f"[{study_name}] Iniciando estudio ({n_trials} trials)...")

    # Cargar dataset
    arff_path = dataset_path(dataset_name)
    dataset = ArffToMIData.from_arff(arff_path)
    bag_ids = [bag.bag_id for bag in dataset.bags]
    n_bags = len(bag_ids)

    # Crear sampler con semilla fija (fairness)
    sampler = TPESampler(seed=OPTUNA_SAMPLER_SEED)
    pruner = MedianPruner()

    # Crear o cargar estudio
    load_if_exists = resume
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=load_if_exists,
    )

    # Determinar cuántos trials ya existen
    existing_trials = len(study.trials)
    remaining = max(0, n_trials - existing_trials)

    if remaining == 0:
        logger.info(f"[{study_name}] Ya tiene {existing_trials} trials, nada que añadir.")
    else:
        if existing_trials > 0:
            logger.info(f"[{study_name}] Retomando: {existing_trials} existentes, {remaining} restantes.")

        # Crear función objetivo
        if model_name in UNSUPERVISED_MODELS:
            objective = create_unsupervised_objective(
                model_name, dataset_name, dataset, bag_ids, n_bags
            )
        else:
            objective = create_supervised_objective(
                model_name, dataset_name, dataset, n_bags
            )

        # Ejecutar optimización con blindaje ante trials fallidos y early stopping
        callbacks = (
            [EarlyStoppingCallback(patience=early_stopping_patience)]
            if early_stopping_patience > 0
            else None
        )
        study.optimize(
            objective,
            n_trials=remaining,
            catch=(Exception,),
            callbacks=callbacks,
            show_progress_bar=False,
        )

    # Exportar trials individuales a CSV
    trials_df = study.trials_dataframe()
    trials_csv_path = RESULTS_ESTUDIOS_OPTUNA / f"{study_name}_trials.csv"
    trials_df.to_csv(trials_csv_path, index=False)
    logger.info(f"[{study_name}] Trials exportados a CSV: {trials_csv_path}")

    # Extraer mejor resultado
    best = study.best_trial
    result = {
        "dataset": dataset_name,
        "model": model_name,
        "study_name": study_name,
        "n_trials": len(study.trials),
        "best_trial_number": best.number,
        "best_value": best.value,
        "best_params": best.params,
        "db_path": str(db_path),
        "trials_csv": str(trials_csv_path),
    }

    logger.info(
        f"[{study_name}] Mejor trial #{best.number}: "
        f"score={best.value:.4f} | params={best.params}"
    )

    return result


def run_fase2(
    dataset_names: Optional[List[str]] = None,
    model_names: Optional[List[str]] = None,
    n_trials: int = N_TRIALS_DEFAULT,
    early_stopping_patience: int = EARLY_STOPPING_DEFAULT,
    resume: bool = False,
) -> None:
    """Ejecuta la Fase 2 completa: optimización de hiperparámetros.

    Args:
        dataset_names: Datasets a procesar (None = todos).
        model_names: Modelos a optimizar (None = todos).
        n_trials: Número máximo de trials por estudio.
        early_stopping_patience: Corta si transcurren N trials sin mejora.
        resume: Si True, continúa estudios existentes.
    """
    datasets = dataset_names or DATASETS
    models = model_names or MODELS

    ensure_result_dirs()

    total = len(datasets) * len(models)

    # Estimación de tiempo
    _, _, time_str, details = estimate_fase2(
        dataset_names=datasets,
        model_names=models,
        n_trials=n_trials,
        resume=resume,
    )
    print_phase_header(
        phase_title="FASE 2 — OPTIMIZACIÓN DE HIPERPARÁMETROS (OPTUNA)",
        estimated_time_str=time_str,
        details=details,
        logger=logger,
    )

    all_results = []
    errors = []

    with tqdm(total=total, desc="Fase 2", ncols=80) as pbar:
        for ds_name in datasets:
            for model_name in models:
                pbar.set_postfix_str(f"{ds_name[:12]}/{model_name}")

                try:
                    result = run_single_study(
                        ds_name, model_name,
                        n_trials=n_trials,
                        early_stopping_patience=early_stopping_patience,
                        resume=resume,
                    )
                    all_results.append(result)
                except Exception as e:
                    logger.error(f"Error en {ds_name}/{model_name}: {e}", exc_info=True)
                    errors.append({
                        "dataset": ds_name,
                        "model": model_name,
                        "error": str(e),
                    })

                pbar.update(1)

    # Guardar resumen en múltiples formatos CSV
    if all_results:
        # 1. CSV aplanado con columnas explícitas para cada hiperparámetro
        rows = []
        for r in all_results:
            row = {
                "dataset": r["dataset"],
                "model": r["model"],
                "study_name": r["study_name"],
                "n_trials": r["n_trials"],
                "best_trial_number": r["best_trial_number"],
                "best_score": r["best_value"],
                "best_scaler": r["best_params"].get("scaler", ""),
                "best_distance": r["best_params"].get("distance", ""),
            }
            for k, v in r["best_params"].items():
                if k not in ["scaler", "distance"]:
                    row[f"param_{k}"] = v
            rows.append(row)

        df_flat = pd.DataFrame(rows)
        csv_path_logs = RESULTS_LOGS / "fase2_mejores_configuraciones.csv"
        csv_path_optuna = RESULTS_ESTUDIOS_OPTUNA / "fase2_mejores_configuraciones.csv"
        df_flat.to_csv(csv_path_logs, index=False)
        df_flat.to_csv(csv_path_optuna, index=False)
        logger.info(f"Mejores configuraciones guardadas en {csv_path_logs} y {csv_path_optuna}")

        # 2. Consolidar todos los trials individuales en un único gran CSV
        all_trials_dfs = []
        for r in all_results:
            trials_f = Path(r.get("trials_csv", ""))
            if trials_f.exists():
                try:
                    t_df = pd.read_csv(trials_f)
                    t_df["dataset"] = r["dataset"]
                    t_df["model"] = r["model"]
                    all_trials_dfs.append(t_df)
                except Exception as ex:
                    logger.warning(f"No se pudo leer {trials_f}: {ex}")

        if all_trials_dfs:
            df_all_trials = pd.concat(all_trials_dfs, ignore_index=True)
            all_trials_csv = RESULTS_ESTUDIOS_OPTUNA / "fase2_todos_los_trials.csv"
            df_all_trials.to_csv(all_trials_csv, index=False)
            logger.info(f"Todos los trials consolidados en {all_trials_csv}")

        # 3. JSON detallado
        json_path = RESULTS_LOGS / "fase2_resultados_completos.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, default=str, ensure_ascii=False)

        # Resumen por consola
        print("\n  Mejores configuraciones:")
        for r in all_results:
            print(f"    [{r['dataset']}/{r['model']}] score={r['best_value']:.4f} → {r['best_params']}")

    if errors:
        print(f"\n  ⚠ Errores: {len(errors)}")
        for e in errors:
            print(f"    - {e['dataset']}/{e['model']}: {e['error']}")

    print("\n" + "═" * 75)
    print("  FASE 2 COMPLETADA")
    print(f"  Estudios en: {RESULTS_ESTUDIOS_OPTUNA}")
    print("═" * 75 + "\n")



# CLI


def main():
    parser = argparse.ArgumentParser(
        description="Fase 2: Optimización de hiperparámetros con Optuna.",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help=f"Datasets a procesar (default: todos). Opciones: {DATASETS}",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help=f"Modelos a optimizar (default: todos). Opciones: {MODELS}",
    )
    parser.add_argument(
        "--n-trials", type=int, default=N_TRIALS_DEFAULT,
        help=f"Número máximo de trials por estudio (default: {N_TRIALS_DEFAULT}).",
    )
    parser.add_argument(
        "--early-stopping", type=int, default=EARLY_STOPPING_DEFAULT,
        help=f"Paciencia de early stopping: corta tras N trials sin mejora (default: {EARLY_STOPPING_DEFAULT}, 0 para desactivar).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Continuar estudios existentes en lugar de crear nuevos.",
    )

    args = parser.parse_args()

    run_fase2(
        dataset_names=args.datasets,
        model_names=args.models,
        n_trials=args.n_trials,
        early_stopping_patience=args.early_stopping,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
