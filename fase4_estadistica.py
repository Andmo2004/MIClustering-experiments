"""
fase4_estadistica.py — Fase 4: Análisis estadístico comparativo.

A partir de los resultados de la Fase 3, realiza:
  1. Test de Friedman sobre rankings de modelos (PI1).
  2. Post-hoc Nemenyi o Wilcoxon pareado con corrección de Holm.
  3. Comparaciones por distancia (PI2) y por escalado (PI3).
  4. Diagrama de diferencia crítica (CD diagram).
  5. Correlación Spearman entre Silhouette y métricas externas (PI4).

Uso:
    python fase4_estadistica.py
    python fase4_estadistica.py --metric F1-Score      # Métrica alternativa
    python fase4_estadistica.py --alpha 0.01            # Nivel de significancia
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from config import (
    DATASETS,
    MODELS,
    RESULTS_EVALUACION_FINAL,
    RESULTS_ANALISIS_ESTADIST,
    RESULTS_LOGS,
    ensure_result_dirs,
    setup_logging,
)
from time_estimator import estimate_fase4, print_phase_header

logger = setup_logging("fase4")



# Carga y preparación de datos


def load_fase3_results() -> pd.DataFrame:
    """Carga todos los resultados JSON de la Fase 3 en un DataFrame."""
    results = []

    for json_file in RESULTS_EVALUACION_FINAL.glob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
            results.append(data)
        except Exception as e:
            logger.warning(f"Error cargando {json_file}: {e}")

    if not results:
        # Intentar desde el CSV consolidado
        csv_path = RESULTS_LOGS / "fase3_resultados_completos.csv"
        if csv_path.exists():
            return pd.read_csv(csv_path)
        raise FileNotFoundError(
            "No se encontraron resultados de la Fase 3. "
            "Ejecuta fase3_evaluacion.py primero."
        )

    return pd.DataFrame(results)


def build_performance_matrix(
    df: pd.DataFrame,
    metric: str = "F1-Score",
    agg: str = "mean",
) -> pd.DataFrame:
    """Construye la matriz dataset × modelo (media de réplicas).

    Args:
        df: DataFrame con resultados de Fase 3.
        metric: Nombre de la métrica a usar.
        agg: Función de agregación ('mean', 'median').

    Returns:
        DataFrame pivotado (filas=datasets, columnas=modelos).
    """
    if metric not in df.columns:
        available = [c for c in df.columns if c not in [
            "dataset", "model", "seed_idx", "seed", "scaler", "distance",
            "model_params", "fit_time_sec", "n_clusters", "noise_count",
            "noise_pct", "n_folds",
        ]]
        raise ValueError(f"Métrica '{metric}' no encontrada. Disponibles: {available}")

    # Agregar por (dataset, model)
    grouped = df.groupby(["dataset", "model"])[metric].agg(agg).reset_index()
    matrix = grouped.pivot(index="dataset", columns="model", values=metric)

    return matrix



# Tests estadísticos


def friedman_test(matrix: pd.DataFrame) -> Dict[str, Any]:
    """Test de Friedman sobre la matriz de rendimiento.

    Responde PI1: ¿Existen diferencias significativas entre los modelos/tratamientos?

    Args:
        matrix: DataFrame (datasets × modelos) con rendimiento medio.

    Returns:
        Dict con estadístico, p-valor, y rankings medios.
    """
    # Convertir a array para scipy
    # Cada columna = un tratamiento (modelo), cada fila = un bloque (dataset)
    data = matrix.dropna()

    if data.shape[0] < 3:
        logger.warning("Menos de 3 datasets con datos completos para Friedman.")
        return {"error": "Datos insuficientes para Friedman"}

    groups = [data[col].values for col in data.columns]
    stat, p_value = stats.friedmanchisquare(*groups)

    # Rankings: para cada dataset, rankear los modelos (rank 1 = mejor)
    rankings = data.rank(axis=1, ascending=False, method="average")
    mean_ranks = rankings.mean()

    return {
        "test": "Friedman",
        "statistic": float(stat),
        "p_value": float(p_value),
        "n_datasets": data.shape[0],
        "n_treatments": data.shape[1],
        "mean_ranks": mean_ranks.to_dict(),
        "ranking_order": mean_ranks.sort_values().index.tolist(),
    }


def nemenyi_posthoc(matrix: pd.DataFrame, alpha: float = 0.05) -> Optional[pd.DataFrame]:
    """Post-hoc de Nemenyi tras un test de Friedman significativo.

    Requiere scikit-posthocs. Si no está instalado, usa Wilcoxon pareado.

    Returns:
        DataFrame con p-valores de comparaciones par-a-par, o None.
    """
    try:
        import scikit_posthocs as sp  # type: ignore
    except ImportError:
        logger.warning("scikit-posthocs no instalado. Usando Wilcoxon pareado como alternativa.")
        return wilcoxon_posthoc(matrix, alpha)

    data = matrix.dropna()
    # scikit-posthocs espera formato long
    melted = data.reset_index().melt(
        id_vars="dataset", var_name="model", value_name="score"
    )

    try:
        result = sp.posthoc_nemenyi_friedman(
            melted, y_col="score", group_col="model", block_col="dataset"
        )
        return result
    except Exception as e:
        logger.warning(f"Error en Nemenyi: {e}. Usando Wilcoxon pareado.")
        return wilcoxon_posthoc(matrix, alpha)


def wilcoxon_posthoc(
    matrix: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Post-hoc Wilcoxon pareado con corrección de Holm.

    Alternativa a Nemenyi cuando scikit-posthocs no está disponible.
    """
    data = matrix.dropna()
    models = data.columns.tolist()
    n_models = len(models)

    p_values = np.ones((n_models, n_models))

    for i in range(n_models):
        for j in range(i + 1, n_models):
            try:
                stat, p = stats.wilcoxon(
                    data[models[i]], data[models[j]],
                    alternative="two-sided",
                )
                p_values[i, j] = p
                p_values[j, i] = p
            except Exception:
                p_values[i, j] = 1.0
                p_values[j, i] = 1.0

    # Corrección de Holm
    pairs = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            pairs.append((i, j, p_values[i, j]))
    pairs.sort(key=lambda x: x[2])

    m = len(pairs)
    for rank, (i, j, p) in enumerate(pairs):
        corrected_p = min(p * (m - rank), 1.0)
        p_values[i, j] = corrected_p
        p_values[j, i] = corrected_p

    return pd.DataFrame(p_values, index=models, columns=models)


def cd_diagram_data(
    matrix: pd.DataFrame,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Genera los datos necesarios para un diagrama de diferencia crítica.

    Calcula el CD (critical difference) según Nemenyi:
      CD = q_alpha * sqrt(k*(k+1) / (6*N))

    donde k = nº tratamientos, N = nº bloques (datasets).
    """
    data = matrix.dropna()
    k = data.shape[1]  # tratamientos
    N = data.shape[0]  # bloques

    # Rankings
    rankings = data.rank(axis=1, ascending=False, method="average")
    mean_ranks = rankings.mean().sort_values()

    # Valores críticos de Nemenyi (q_alpha para alpha=0.05)
    # Tabla para k=2..10 (aproximación)
    q_table = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
        6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
    }

    q_alpha = q_table.get(k, 2.728)  # Default a k=5
    cd = q_alpha * np.sqrt(k * (k + 1) / (6.0 * N))

    return {
        "mean_ranks": mean_ranks.to_dict(),
        "critical_difference": float(cd),
        "alpha": alpha,
        "k": k,
        "N": N,
        "q_alpha": q_alpha,
    }



# Correlación interna vs externa (PI4)


def correlation_internal_external(
    df: pd.DataFrame,
    internal_metric: str = "best_score",
    external_metric: str = "F1-Score",
) -> Dict[str, Any]:
    """Correlación Spearman entre métrica interna (Silhouette) y externa.

    Responde PI4: ¿Existe consistencia entre calidad interna y externa?
    """
    # Intentar cargar el best_score de los estudios Optuna
    optuna_csv = RESULTS_LOGS / "fase2_mejores_configuraciones.csv"
    if not optuna_csv.exists():
        logger.warning("No se encontró fase2_mejores_configuraciones.csv para PI4.")
        return {"error": "Datos de Fase 2 no disponibles"}

    df_optuna = pd.read_csv(optuna_csv)
    score_col = "best_score" if "best_score" in df_optuna.columns else ("best_value" if "best_value" in df_optuna.columns else None)
    if score_col is None:
        return {"error": "Columna de score no encontrada en datos de Fase 2"}

    # Merge con resultados de Fase 3
    if external_metric not in df.columns:
        return {"error": f"Métrica '{external_metric}' no disponible en resultados"}

    # Agregar Fase 3 por (dataset, model)
    df_ext = df.groupby(["dataset", "model"])[external_metric].mean().reset_index()
    df_ext.columns = ["dataset", "model", "external_score"]

    # Merge
    df_merged = pd.merge(
        df_optuna[["dataset", "model", score_col]],
        df_ext,
        on=["dataset", "model"],
        how="inner",
    )

    if len(df_merged) < 3:
        return {"error": "Datos insuficientes para correlación (menos de 3 observaciones)"}

    rho, p_value = stats.spearmanr(df_merged[score_col], df_merged["external_score"])

    return {
        "test": "Spearman",
        "internal_metric": "Silhouette (Optuna best)",
        "external_metric": external_metric,
        "rho": float(rho),
        "p_value": float(p_value),
        "n_observations": len(df_merged),
    }



# Pipeline principal


def run_fase4(
    metric: str = "F1-Score",
    alpha: float = 0.05,
) -> None:
    """Ejecuta la Fase 4 completa: análisis estadístico comparativo."""

    ensure_result_dirs()
    RESULTS_ANALISIS_ESTADIST.mkdir(parents=True, exist_ok=True)

    # Estimación de tiempo
    _, _, time_str, details = estimate_fase4()
    details_with_metric = [f"Métrica principal: {metric} | Nivel de significancia: α = {alpha}"] + details
    print_phase_header(
        phase_title="FASE 4 — ANÁLISIS ESTADÍSTICO COMPARATIVO",
        estimated_time_str=time_str,
        details=details_with_metric,
        logger=logger,
    )

    # 1. Cargar datos
    print("\n[1/5] Cargando resultados de la Fase 3...")
    df = load_fase3_results()
    print(f"  {len(df)} evaluaciones cargadas ({df['dataset'].nunique()} datasets, {df['model'].nunique()} modelos)")

    # 2. Construir matriz de rendimiento (PI1: por modelo)
    print("\n[2/5] Construyendo matriz dataset × modelo...")
    matrix_model = build_performance_matrix(df, metric=metric)
    print(matrix_model.to_string())

    csv_path = RESULTS_ANALISIS_ESTADIST / "matriz_rendimiento_modelo.csv"
    matrix_model.to_csv(csv_path)

    # 3. Test de Friedman (PI1)
    print("\n[3/5] Test de Friedman (PI1: ¿Diferencias entre modelos?)...")
    friedman_result = friedman_test(matrix_model)

    if "error" not in friedman_result:
        print(f"  Estadístico: {friedman_result['statistic']:.4f}")
        print(f"  p-valor: {friedman_result['p_value']:.6f}")
        print(f"  Rankings medios: {friedman_result['mean_ranks']}")
        print(f"  Orden (mejor → peor): {friedman_result['ranking_order']}")

        significant = friedman_result["p_value"] < alpha
        print(f"  → {'SIGNIFICATIVO' if significant else 'NO significativo'} (α={alpha})")

        # Post-hoc si significativo
        if significant:
            print("\n  Post-hoc (comparaciones par-a-par):")
            posthoc_result = nemenyi_posthoc(matrix_model, alpha)
            if posthoc_result is not None:
                print(posthoc_result.to_string())
                posthoc_path = RESULTS_ANALISIS_ESTADIST / "posthoc_pvalues_modelo.csv"
                posthoc_result.to_csv(posthoc_path)

        # CD diagram data
        cd_data = cd_diagram_data(matrix_model, alpha)
        print(f"\n  Diferencia Crítica (CD): {cd_data['critical_difference']:.4f}")

        # Exportar rankings y resumen de Friedman a CSV
        df_ranks = pd.DataFrame([
            {"modelo": m, "ranking_medio": r}
            for m, r in friedman_result["mean_ranks"].items()
        ]).sort_values("ranking_medio")
        df_ranks.to_csv(RESULTS_ANALISIS_ESTADIST / "friedman_rankings_modelos.csv", index=False)

        df_friedman_summary = pd.DataFrame([{
            "comparacion": "Modelos (PI1)",
            "estadistico": friedman_result["statistic"],
            "p_valor": friedman_result["p_value"],
            "significativo_alpha_0.05": friedman_result["p_value"] < alpha,
            "diferencia_critica_CD": cd_data.get("critical_difference", None),
            "n_datasets": friedman_result.get("n_datasets"),
            "n_tratamientos": friedman_result.get("n_treatments"),
        }])
        df_friedman_summary.to_csv(RESULTS_ANALISIS_ESTADIST / "friedman_resumen_modelos.csv", index=False)
    else:
        print(f"  {friedman_result['error']}")
        friedman_result = {"error": friedman_result["error"]}
        cd_data = {}

    # 4. Comparaciones por distancia (PI2) y escalado (PI3)
    print("\n[4/5] Comparaciones adicionales (PI2: distancia, PI3: escalado)...")

    additional_tests = {}

    # PI2: por distancia
    if "distance" in df.columns:
        matrix_dist = df.groupby(["dataset", "distance"])[metric].mean().reset_index()
        matrix_dist_pivot = matrix_dist.pivot(index="dataset", columns="distance", values=metric)

        if matrix_dist_pivot.shape[1] >= 2:
            friedman_dist = friedman_test(matrix_dist_pivot)
            additional_tests["PI2_distancia"] = friedman_dist
            if "error" not in friedman_dist:
                print(f"\n  PI2 (Distancia): Friedman p={friedman_dist['p_value']:.6f}")
                print(f"  Rankings: {friedman_dist['mean_ranks']}")
                matrix_dist_pivot.to_csv(
                    RESULTS_ANALISIS_ESTADIST / "matriz_rendimiento_distancia.csv"
                )
                df_ranks_dist = pd.DataFrame([
                    {"distancia": d, "ranking_medio": r}
                    for d, r in friedman_dist["mean_ranks"].items()
                ]).sort_values("ranking_medio")
                df_ranks_dist.to_csv(RESULTS_ANALISIS_ESTADIST / "friedman_rankings_distancia.csv", index=False)

    # PI3: por escalado
    if "scaler" in df.columns:
        matrix_scaler = df.groupby(["dataset", "scaler"])[metric].mean().reset_index()
        matrix_scaler_pivot = matrix_scaler.pivot(index="dataset", columns="scaler", values=metric)

        if matrix_scaler_pivot.shape[1] >= 2:
            friedman_scaler = friedman_test(matrix_scaler_pivot)
            additional_tests["PI3_escalado"] = friedman_scaler
            if "error" not in friedman_scaler:
                print(f"\n  PI3 (Escalado): Friedman p={friedman_scaler['p_value']:.6f}")
                print(f"  Rankings: {friedman_scaler['mean_ranks']}")
                matrix_scaler_pivot.to_csv(
                    RESULTS_ANALISIS_ESTADIST / "matriz_rendimiento_escalado.csv"
                )
                df_ranks_sc = pd.DataFrame([
                    {"scaler": s, "ranking_medio": r}
                    for s, r in friedman_scaler["mean_ranks"].items()
                ]).sort_values("ranking_medio")
                df_ranks_sc.to_csv(RESULTS_ANALISIS_ESTADIST / "friedman_rankings_escalado.csv", index=False)

    # 5. Correlación interna vs externa (PI4)
    print("\n[5/5] Correlación interna-externa (PI4)...")
    corr_result = correlation_internal_external(df, external_metric=metric)
    if "error" not in corr_result:
        print(f"  Spearman ρ = {corr_result['rho']:.4f} (p = {corr_result['p_value']:.6f})")
        print(f"  n = {corr_result['n_observations']} observaciones")
        df_corr = pd.DataFrame([corr_result])
        df_corr.to_csv(RESULTS_ANALISIS_ESTADIST / "correlacion_interna_externa.csv", index=False)
    else:
        print(f"  {corr_result['error']}")

    # Guardar todos los resultados
    all_results = {
        "metric": metric,
        "alpha": alpha,
        "PI1_friedman_modelos": friedman_result,
        "PI1_cd_diagram": cd_data,
        "PI2_PI3": additional_tests,
        "PI4_correlacion": corr_result,
    }

    json_path = RESULTS_ANALISIS_ESTADIST / "fase4_resultados.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str, ensure_ascii=False)

    print("\n" + "═" * 75)
    print("  FASE 4 COMPLETADA")
    print(f"  Resultados en: {RESULTS_ANALISIS_ESTADIST}")
    print("═" * 75 + "\n")



# CLI


def main():
    parser = argparse.ArgumentParser(
        description="Fase 4: Análisis estadístico comparativo.",
    )
    parser.add_argument(
        "--metric", type=str, default="F1-Score",
        help="Métrica externa principal para la comparación (default: F1-Score).",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="Nivel de significancia (default: 0.05).",
    )

    args = parser.parse_args()
    run_fase4(metric=args.metric, alpha=args.alpha)


if __name__ == "__main__":
    main()
