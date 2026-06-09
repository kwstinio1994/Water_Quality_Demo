import os
import pandas as pd

import pipeline_code.generic.generic_library as gl
from pipeline_code.configurations.config import (
    metric_info, correlation_columns, season_map,
    insight_time_col, insight_lat_col, insight_lon_col,
    insight_wqi_col, insight_result_col, wqi_classes,
)
from pipeline_code.insights import insights_library as il


def run_insights_pipeline(
    config,
    project,
    output_dir="data",
    *,
    skip_if_exists=False,
    produce_plots=True,
):
    base_dir = os.path.join(output_dir, project.name, "wqi")
    insights_dir = os.path.join(output_dir, project.name, "insights")
    figures_dir = os.path.join(insights_dir, "figures")
    os.makedirs(insights_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    metrics = list(metric_info.keys())
    value_cols = [metric_info[m]["value"] for m in metrics]
    flag_cols = [metric_info[m]["flag"] for m in metrics]

    print(f"{'=' * 70}")
    print(f"  Insights pipeline")
    print(f"{'=' * 70}")

    wqi_csv = os.path.join(base_dir, "water_quality_index.csv")
    print(f"\n[1] Loading wqi csv: {wqi_csv}")
    df = pd.read_csv(wqi_csv)
    for c in df.columns:
        if "Unnamed" in c:
            df.drop(columns=[c], inplace=True)
    print(f"  Rows: {len(df)},  Columns: {len(df.columns)}")

    print(f"\n[2] Assigning seasons")
    df = il.assign_seasons(df, insight_time_col, season_map)
    print(f"  Seasons: {sorted(df['season'].unique())}")

    if produce_plots:
        print(f"\n[3] Correlation heatmaps + covariance + pairplots")
        out_dir = os.path.join(figures_dir, "correlations")
        il.plot_correlation_heatmaps(df, insight_result_col, correlation_columns, out_dir)
        il.plot_pairplots(df, insight_result_col, correlation_columns, out_dir)
        print(f"  →  {out_dir}/")

    print(f"\n[4] Seasonal metric breakdown")
    out_csv = os.path.join(insights_dir, "seasonal_metric_breakdown.csv")
    skip = skip_if_exists and os.path.isfile(out_csv)
    if skip:
        print(f"  Skipping (exists)")
    else:
        smb = il.seasonal_metric_breakdown(df, metrics, flag_cols)
        gl.save_csv(smb, out_csv)
        print(f"  Rows: {len(smb)}  →  {out_csv}")

    print(f"\n[5] Spot breakdown (per-location quality distribution)")
    out_csv = os.path.join(insights_dir, "spot_breakdown.csv")
    skip = skip_if_exists and os.path.isfile(out_csv)
    if skip:
        print(f"  Skipping (exists)")
    else:
        sb = il.spot_breakdown(df, insight_lat_col, insight_lon_col, insight_wqi_col, wqi_classes)
        gl.save_csv(sb, out_csv)
        print(f"  Rows: {len(sb)}  →  {out_csv}")

    print(f"\n[6] Spot + season breakdown")
    out_csv = os.path.join(insights_dir, "spot_season_breakdown.csv")
    skip = skip_if_exists and os.path.isfile(out_csv)
    if skip:
        print(f"  Skipping (exists)")
    else:
        ssb = il.spot_season_breakdown(df, insight_lat_col, insight_lon_col, "season",
                                        insight_wqi_col, wqi_classes)
        gl.save_csv(ssb, out_csv)
        print(f"  Rows: {len(ssb)}  →  {out_csv}")

    print(f"\n[7] Holistic percentages")
    out_csv = os.path.join(insights_dir, "holistic_breakdown.csv")
    skip = skip_if_exists and os.path.isfile(out_csv)
    if skip:
        print(f"  Skipping (exists)")
    else:
        hb = il.holistic_breakdown(df, insight_result_col)
        gl.save_csv(hb, out_csv)
        print(f"  Rows: {len(hb)}  →  {out_csv}")

    print(f"\n[8] Per-bin metric flag breakdown")
    out_csv = os.path.join(insights_dir, "per_bin_breakdown.csv")
    skip = skip_if_exists and os.path.isfile(out_csv)
    if skip:
        print(f"  Skipping (exists)")
    else:
        pbb = il.per_bin_breakdown(df, insight_result_col, metrics, flag_cols, wqi_classes)
        gl.save_csv(pbb, out_csv)
        print(f"  Rows: {len(pbb)}  →  {out_csv}")

    if produce_plots:
        print(f"\n[9] Seasonal average bar plots")
        out_dir = os.path.join(figures_dir, "seasonal_averages")
        avg_df = il.seasonal_averages(df, "season", value_cols)
        il.plot_seasonal_bars(avg_df, "season", value_cols, out_dir)
        print(f"  →  {out_dir}/")

    print(f"\n[10] Seasonal index (quality distribution per season)")
    out_csv = os.path.join(insights_dir, "seasonal_index.csv")
    skip = skip_if_exists and os.path.isfile(out_csv)
    if skip:
        print(f"  Skipping (exists)")
    else:
        si = il.seasonal_index(df, "season", insight_result_col)
        gl.save_csv(si, out_csv)
        print(f"  Rows: {len(si)}  →  {out_csv}")

    print(f"\n[11] What excellent means")
    out_csv = os.path.join(insights_dir, "what_excellent_means.csv")
    skip = skip_if_exists and os.path.isfile(out_csv)
    if skip:
        print(f"  Skipping (exists)")
    else:
        wem = il.what_excellent_means(df, insight_result_col, metrics, flag_cols, "Excellent")
        gl.save_csv(wem, out_csv)
        print(f"  Rows: {len(wem)}  →  {out_csv}")

    print(f"\n{'─' * 70}")
    print(f"  Finished insights")
    print(f"  Outputs: {insights_dir}/")
    if produce_plots:
        print(f"  Figures: {figures_dir}/")
    print(f"{'─' * 70}")

    return {"df": df, "insights_dir": insights_dir, "figures_dir": figures_dir}
