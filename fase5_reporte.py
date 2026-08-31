"""
fase5_reporte.py — Fase 5: Síntesis y reporte de resultados.

Genera las tablas y figuras finales para la tesis:
  1. Tabla resumen por modelo: mejor distancia, escalado, métrica media ± std, ranking.
  2. Tabla resumen por distancia: rendimiento agregado.
  3. CD diagrams de las tres comparaciones (modelo, distancia, escalado).
  4. Heatmaps de rendimiento.
  5. Discusión de amenazas a la validez.

Uso:
    python fase5_reporte.py
    python fase5_reporte.py --format pdf     # Exportar figuras como PDF
    python fase5_reporte.py --metric Recall   # Métrica alternativa
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # Backend no interactivo
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    DATASETS,
    MODELS,
    DISTANCES,
    SCALERS,
    RESULTS_EVALUACION_FINAL,
    RESULTS_ANALISIS_ESTADIST,
    RESULTS_REPORTE,
    RESULTS_LOGS,
    ensure_result_dirs,
    setup_logging,
)

logger = setup_logging("fase5")

# Estilo de gráficos
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


# ═══════════════════════════════════════════════════════════════════════════
# Carga de datos
# ═══════════════════════════════════════════════════════════════════════════

def load_all_data():
    """Carga resultados de Fase 3 y análisis de Fase 4."""
    # Fase 3
    csv_path = RESULTS_LOGS / "fase3_resultados_completos.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        # Cargar desde JSONs individuales
        results = []
        for f in RESULTS_EVALUACION_FINAL.glob("*.json"):
            with open(f) as fh:
                results.append(json.load(fh))
        df = pd.DataFrame(results) if results else pd.DataFrame()

    # Fase 4
    fase4_path = RESULTS_ANALISIS_ESTADIST / "fase4_resultados.json"
    fase4 = {}
    if fase4_path.exists():
        with open(fase4_path) as f:
            fase4 = json.load(f)

    return df, fase4


# ═══════════════════════════════════════════════════════════════════════════
# Tablas resumen
# ═══════════════════════════════════════════════════════════════════════════

def generate_model_summary_table(df: pd.DataFrame, metric: str = "F1-Score") -> pd.DataFrame:
    """Tabla resumen por modelo (§5.1 del protocolo)."""
    if df.empty or metric not in df.columns:
        return pd.DataFrame()

    summary = df.groupby("model").agg(
        best_distance=("distance", lambda x: x.value_counts().idxmax() if not x.empty else "N/A"),
        best_scaler=("scaler", lambda x: x.value_counts().idxmax() if not x.empty else "N/A"),
        metric_mean=(metric, "mean"),
        metric_std=(metric, "std"),
        precision_mean=("Precision", "mean"),
        recall_mean=("Recall", "mean"),
        n_evaluations=("dataset", "count"),
    ).reset_index()

    summary.columns = [
        "Modelo", "Distancia más frecuente", "Scaler más frecuente",
        f"{metric} (media)", f"{metric} (std)",
        "Precision (media)", "Recall (media)", "N evaluaciones",
    ]

    # Ordenar por rendimiento
    summary = summary.sort_values(f"{metric} (media)", ascending=False)

    return summary


def generate_distance_summary_table(df: pd.DataFrame, metric: str = "F1-Score") -> pd.DataFrame:
    """Tabla resumen por distancia (§5.2 del protocolo)."""
    if df.empty or metric not in df.columns:
        return pd.DataFrame()

    summary = df.groupby("distance").agg(
        metric_mean=(metric, "mean"),
        metric_std=(metric, "std"),
        n_models=("model", "nunique"),
        n_datasets=("dataset", "nunique"),
    ).reset_index()

    summary.columns = [
        "Distancia", f"{metric} (media)", f"{metric} (std)",
        "Modelos evaluados", "Datasets evaluados",
    ]

    summary = summary.sort_values(f"{metric} (media)", ascending=False)

    return summary


# ═══════════════════════════════════════════════════════════════════════════
# Figuras
# ═══════════════════════════════════════════════════════════════════════════

def plot_heatmap(df: pd.DataFrame, metric: str = "F1-Score", fmt: str = "png"):
    """Heatmap dataset × modelo con la métrica de rendimiento."""
    if df.empty or metric not in df.columns:
        logger.warning("Datos insuficientes para heatmap.")
        return

    matrix = df.groupby(["dataset", "model"])[metric].mean().reset_index()
    pivot = matrix.pivot(index="dataset", columns="model", values=metric)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="YlOrRd",
        linewidths=0.5, ax=cast(Any, ax),
        cbar_kws={"label": metric},
    )
    ax.set_title(f"Rendimiento por Dataset × Modelo ({metric})", fontsize=14, pad=15)
    ax.set_ylabel("Dataset", fontsize=12)
    ax.set_xlabel("Modelo", fontsize=12)
    plt.tight_layout()

    path = RESULTS_REPORTE / f"heatmap_dataset_modelo.{fmt}"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Heatmap guardado en {path}")


def plot_boxplot_models(df: pd.DataFrame, metric: str = "F1-Score", fmt: str = "png"):
    """Boxplot comparativo de modelos."""
    if df.empty or metric not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    df_plot = df[["model", metric]].dropna()
    order = df_plot.groupby("model")[metric].median().sort_values(ascending=False).index

    sns.boxplot(data=df_plot, x="model", y=metric, order=order, ax=cast(Any, ax), palette="Set2")
    sns.stripplot(
        data=df_plot, x="model", y=metric, order=order, ax=cast(Any, ax),
        color="black", alpha=0.3, size=3, jitter=True,
    )
    ax.set_title(f"Distribución de {metric} por Modelo", fontsize=14, pad=15)
    ax.set_xlabel("Modelo", fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    plt.tight_layout()

    path = RESULTS_REPORTE / f"boxplot_modelos.{fmt}"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Boxplot guardado en {path}")


def plot_boxplot_distances(df: pd.DataFrame, metric: str = "F1-Score", fmt: str = "png"):
    """Boxplot comparativo de distancias."""
    if df.empty or metric not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    df_plot = df[["distance", metric]].dropna()
    order = df_plot.groupby("distance")[metric].median().sort_values(ascending=False).index

    sns.boxplot(data=df_plot, x="distance", y=metric, order=order, ax=cast(Any, ax), palette="Set3")
    ax.set_title(f"Distribución de {metric} por Métrica de Distancia", fontsize=14, pad=15)
    ax.set_xlabel("Distancia", fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    plt.xticks(rotation=15)
    plt.tight_layout()

    path = RESULTS_REPORTE / f"boxplot_distancias.{fmt}"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Boxplot distancias guardado en {path}")


def plot_cd_diagram(fase4_data: dict, comparison: str = "modelos", fmt: str = "png"):
    """Diagrama de diferencia crítica simplificado.

    Para un CD diagram completo con líneas de conectividad, se recomienda
    usar `autorank` o `Orange3`. Esta versión genera la visualización básica.
    """
    cd_key = "PI1_cd_diagram"
    if cd_key not in fase4_data or not fase4_data[cd_key]:
        logger.warning("Datos de CD diagram no disponibles.")
        return

    cd_info = fase4_data[cd_key]
    mean_ranks = cd_info.get("mean_ranks", {})
    cd_value = cd_info.get("critical_difference", 0)

    if not mean_ranks:
        return

    # Ordenar por ranking
    sorted_ranks = sorted(mean_ranks.items(), key=lambda x: x[1])
    names = [r[0] for r in sorted_ranks]
    ranks = [r[1] for r in sorted_ranks]

    fig, ax = plt.subplots(figsize=(10, 3))

    # Eje horizontal = rankings
    ax.set_xlim(0.5, max(ranks) + 0.5)
    ax.set_ylim(0, 1)
    ax.invert_xaxis()

    # Líneas y labels
    for i, (name, rank) in enumerate(zip(names, ranks)):
        y = 0.3 + (i % 2) * 0.3
        ax.plot(rank, 0.5, "o", markersize=8, color="steelblue", zorder=5)
        ax.annotate(
            f"{name}\n({rank:.2f})",
            xy=(rank, 0.5), xytext=(rank, y),
            fontsize=10, ha="center", va="center",
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.8),
        )

    # Línea de CD
    ax.annotate(
        "", xy=(1, 0.85), xytext=(1 + cd_value, 0.85),
        arrowprops=dict(arrowstyle="<->", color="red", lw=2),
    )
    ax.text(
        1 + cd_value / 2, 0.92, f"CD = {cd_value:.3f}",
        ha="center", fontsize=11, color="red", fontweight="bold",
    )

    ax.set_xlabel("Ranking medio (menor = mejor)", fontsize=12)
    ax.set_title(f"Diagrama de Diferencia Crítica — {comparison}", fontsize=14, pad=15)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    path = RESULTS_REPORTE / f"cd_diagram_{comparison}.{fmt}"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"CD diagram guardado en {path}")


def plot_noise_analysis(df: pd.DataFrame, fmt: str = "png"):
    """Análisis del porcentaje de ruido para MIDBSCAN y COSMIC."""
    density_models = df[df["model"].isin(["midbscan", "cosmic"])]
    if density_models.empty or "noise_pct" not in density_models.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=density_models, x="dataset", y="noise_pct",
        hue="model", ax=cast(Any, ax), palette="Set2",
    )
    ax.set_title("Porcentaje de Ruido por Dataset (MIDBSCAN vs COSMIC)", fontsize=14, pad=15)
    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel("Ruido (%)", fontsize=12)
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Modelo")
    plt.tight_layout()

    path = RESULTS_REPORTE / f"noise_analysis.{fmt}"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Análisis de ruido guardado en {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline principal
# ═══════════════════════════════════════════════════════════════════════════

def run_fase5(metric: str = "F1-Score", fig_format: str = "png"):
    """Ejecuta la Fase 5 completa: generación de reporte."""

    ensure_result_dirs()
    RESULTS_REPORTE.mkdir(parents=True, exist_ok=True)

    print("\n" + "═" * 75)
    print("  FASE 5 — SÍNTESIS Y REPORTE DE RESULTADOS")
    print(f"  Métrica principal: {metric} | Formato figuras: {fig_format}")
    print("═" * 75)

    # Cargar datos
    df, fase4_data = load_all_data()

    if df.empty:
        print("  ⚠ No hay resultados disponibles. Ejecuta las Fases 1-4 primero.")
        return

    # 1. Tablas resumen
    print("\n[1/5] Generando tablas resumen en CSV...")

    table_model = generate_model_summary_table(df, metric)
    if not table_model.empty:
        print("\n  Resumen por Modelo:")
        print(table_model.to_string(index=False))
        table_model.to_csv(RESULTS_REPORTE / "tabla_resumen_modelo.csv", index=False)

    table_dist = generate_distance_summary_table(df, metric)
    if not table_dist.empty:
        print("\n  Resumen por Distancia:")
        print(table_dist.to_string(index=False))
        table_dist.to_csv(RESULTS_REPORTE / "tabla_resumen_distancia.csv", index=False)

    if "scaler" in df.columns:
        table_scaler = df.groupby("scaler").agg(
            metric_mean=(metric, "mean"),
            metric_std=(metric, "std"),
            n_evaluaciones=("dataset", "count"),
        ).reset_index()
        table_scaler.columns = ["Scaler", f"{metric} (media)", f"{metric} (std)", "N evaluaciones"]
        table_scaler.to_csv(RESULTS_REPORTE / "tabla_resumen_escalado.csv", index=False)

    # Matriz dataset x modelo a CSV
    if metric in df.columns:
        mat_ds_model = df.groupby(["dataset", "model"])[metric].mean().unstack()
        mat_ds_model.to_csv(RESULTS_REPORTE / "matriz_dataset_modelo_f1.csv")

    # Resumen de ruido a CSV
    density_df = df[df["model"].isin(["midbscan", "cosmic"])]
    if not density_df.empty and "noise_pct" in density_df.columns:
        noise_table = density_df.groupby(["dataset", "model"]).agg(
            noise_pct_mean=("noise_pct", "mean"),
            noise_count_mean=("noise_count", "mean") if "noise_count" in density_df.columns else ("noise_pct", "count"),
        ).reset_index()
        noise_table.to_csv(RESULTS_REPORTE / "analisis_ruido_modelos_densidad.csv", index=False)

    # 2. Heatmap
    print("\n[2/5] Generando heatmap dataset × modelo...")
    plot_heatmap(df, metric, fig_format)

    # 3. Boxplots
    print("[3/5] Generando boxplots comparativos...")
    plot_boxplot_models(df, metric, fig_format)
    plot_boxplot_distances(df, metric, fig_format)

    # 4. CD Diagram
    print("[4/5] Generando CD diagram...")
    if fase4_data:
        plot_cd_diagram(fase4_data, "modelos", fig_format)
    else:
        print("  ⚠ Datos de Fase 4 no disponibles para CD diagram.")

    # 5. Análisis de ruido
    print("[5/5] Generando análisis de ruido...")
    plot_noise_analysis(df, fig_format)

    # Amenazas a la validez
    threats = {
        "tamaño_datasets": (
            "Los 10 datasets varían significativamente en número de bolsas "
            "e instancias por bolsa, lo que puede sesgar los rankings de Friedman "
            "hacia modelos que funcionan bien en datasets grandes."
        ),
        "silhouette_no_euclidea": (
            "El índice de Silhouette está originalmente definido para espacios "
            "euclídeos. Su interpretación con métricas no euclídeas (EMD, Mahalanobis) "
            "puede no ser directamente comparable."
        ),
        "coste_computacional": (
            "EMD y Mahalanobis son significativamente más costosas que Hausdorff, "
            "lo que puede limitar su uso práctico en datasets grandes."
        ),
        "hungarian_assignment": (
            "La asignación húngara para métricas externas puede ser inestable "
            "cuando k difiere sustancialmente del número real de clases."
        ),
    }

    threats_path = RESULTS_REPORTE / "amenazas_validez.json"
    with open(threats_path, "w", encoding="utf-8") as f:
        json.dump(threats, f, indent=2, ensure_ascii=False)

    print("\n" + "═" * 75)
    print("  FASE 5 COMPLETADA")
    print(f"  Tablas y figuras en: {RESULTS_REPORTE}")
    print("═" * 75 + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Fase 5: Síntesis y reporte de resultados.",
    )
    parser.add_argument(
        "--metric", type=str, default="F1-Score",
        help="Métrica principal para las tablas y figuras (default: F1-Score).",
    )
    parser.add_argument(
        "--format", type=str, default="png", choices=["png", "pdf", "svg"],
        help="Formato de las figuras (default: png).",
    )

    args = parser.parse_args()
    run_fase5(metric=args.metric, fig_format=args.format)


if __name__ == "__main__":
    main()
