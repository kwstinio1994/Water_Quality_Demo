"""
Generic pipeline orchestrator for a single water-quality metric.
"""

import os
import sys

import pandas as pd
import numpy as np

from pipeline_code.generic import generic_library as gl


def _step_dir(base, metric, step):
    d = os.path.join(base, metric.name, step)
    os.makedirs(d, exist_ok=True)
    return d


def run_metric_pipeline(
    config,
    project,
    metric,
    output_dir="data",
    *,
    value_col=None,
    lat_col="latitude",
    lon_col="longitude",
    time_col="time",
    do_clustering=False,
    n_clusters_list=(5,),
    region=None,
    skip_if_exists=False,
):
    if value_col is None:
        value_col = gl.value_column(metric)

    if region is None:
        margin = 0.5
        region = [
            project.lon_min - margin,
            project.lon_max + margin,
            project.lat_min - margin,
            project.lat_max + margin,
        ]

    base = os.path.join(output_dir, project.name)
    nc_path = os.path.join(base, metric.output_nc)
    results = {}

    print(f"{'=' * 70}")
    print(f"  Metric: {metric.name}")
    print(f"  Columns: lat={lat_col}, lon={lon_col}, time={time_col}, value={value_col}")
    print(f"  Transform: {metric.transform}")
    print(f"{'=' * 70}")

    step = "1_raw_csv"
    out_csv = os.path.join(base, metric.output_csv)
    if os.path.isfile(out_csv):
        print(f"[1/13] Loading existing csv (from download_all)  →  {out_csv}")
        df = pd.read_csv(out_csv)
    elif os.path.isfile(nc_path):
        print(f"[1/13] Loading netcdf  →  {nc_path}")
        df = gl.load_netcdf(nc_path, metric)
        gl.save_csv(df, out_csv)
        print(f"  Rows: {len(df)},  Columns: {list(df.columns)}")
    else:
        print(f"  ! No data found at {out_csv} or {nc_path}")
        print(f"  ! Run download_all() first.")
        sys.exit(1)
    results[step] = df

    step = "2_transform"
    step_dir = _step_dir(base, metric, step)
    out = os.path.join(step_dir, "transformed.csv")
    if metric.transform is not None:
        print(f"[2/13] Collapsing depth via groupby({lat_col}, {lon_col}, {time_col})  →  {out}")
        if skip_if_exists and os.path.isfile(out):
            df = pd.read_csv(out)
        else:
            before = len(df)
            df = gl.collapse_depth(df, lat_col, lon_col, time_col, value_col, metric.transform)
            gl.save_csv(df, out)
            after = len(df)
            print(f"  Rows: {before} → {after},  New columns: {[c for c in df.columns if c not in (lat_col, lon_col, time_col, value_col)]}")
    else:
        print(f"[2/13] No transform configured — skipping")
    results[step] = df

    step = "3_date_range"
    step_dir = _step_dir(base, metric, step)
    out = os.path.join(step_dir, "date_range.csv")
    print(f"[3/13] Date range per ({lat_col}, {lon_col})  →  {out}")
    if skip_if_exists and os.path.isfile(out):
        dr = pd.read_csv(out)
    else:
        dr = gl.compute_date_range(df, lat_col, lon_col, time_col)
        gl.save_csv(dr, out)
    print(f"  Locations: {len(dr)}")
    results[step] = dr

    step = "4_coord_range"
    step_dir = _step_dir(base, metric, step)
    out = os.path.join(step_dir, "coord_range.csv")
    print(f"[4/13] Coordinate range per date  →  {out}")
    if skip_if_exists and os.path.isfile(out):
        cr = pd.read_csv(out)
    else:
        cr = gl.compute_coord_range(df, time_col, lat_col, lon_col)
        gl.save_csv(cr, out)
    n_dupes = cr["duplicates"].sum()
    print(f"  Dates: {len(cr)},  Total duplicate rows: {n_dupes}")
    results[step] = cr

    step = "5_unique_coords"
    step_dir = _step_dir(base, metric, step)
    out = os.path.join(step_dir, "unique_coords.csv")
    print(f"[5/13] Unique ({lat_col}, {lon_col}) pairs  →  {out}")
    if skip_if_exists and os.path.isfile(out):
        uc = pd.read_csv(out)
    else:
        uc = gl.unique_coords(df, lat_col, lon_col)
        gl.save_csv(uc, out)
    print(f"  Unique locations: {len(uc)}")
    results[step] = uc

    step = "6_coord_plot"
    step_dir = _step_dir(base, metric, step)
    out_png = os.path.join(step_dir, f"{metric.name}_coords.png")
    print(f"[6/13] Coordinate map  →  {out_png}")
    if not (skip_if_exists and os.path.isfile(out_png)):
        gl.plot_coords_map(
            uc, lat_col, lon_col, region,
            title=f"{metric.name} sampling locations",
            output_path=out_png,
        )
    results[step] = None

    step = "7_nulls"
    step_dir = _step_dir(base, metric, step)
    print(f"[7/13] Null analysis  →  {step_dir}/")
    nulls = gl.analyze_nulls(df, lat_col, lon_col, time_col, value_cols=[value_col])
    for key, tbl in nulls.items():
        out = os.path.join(step_dir, f"{key}.csv")
        if skip_if_exists and os.path.isfile(out):
            continue
        gl.save_csv(tbl, out)
    total_nulls = nulls[f"nulls_by_loc_{value_col}"]["nulls"].sum()
    total_rows = len(df)
    print(f"  Total nulls: {total_nulls} / {total_rows}  ({100 * total_nulls / max(total_rows, 1):.1f}%)")
    results[step] = nulls

    step = "8_null_gaps"
    step_dir = _step_dir(base, metric, step)
    out = os.path.join(step_dir, "null_gaps.csv")
    print(f"[8/13] Null gap analysis  →  {out}")
    if skip_if_exists and os.path.isfile(out):
        ng = pd.read_csv(out)
    else:
        ng = gl.analyze_null_gaps(df, lat_col, lon_col, time_col, value_col)
        gl.save_csv(ng, out)
    max_gap = ng["max_consecutive_nulls"].max()
    print(f"  Max consecutive nulls across all locations: {max_gap}")
    results[step] = ng

    step = "9_clean_series"
    step_dir = _step_dir(base, metric, step)
    out = os.path.join(step_dir, "series_clean.csv")
    print(f"[9/13] Remove fully-null time series  →  {out}")
    if skip_if_exists and os.path.isfile(out):
        df_clean = pd.read_csv(out)
    else:
        before = len(df)
        df_clean = gl.remove_full_null_series(df, lat_col, lon_col, value_col)
        gl.save_csv(df_clean, out)
        after = len(df_clean)
        print(f"  Rows before: {before},  after: {after}  (removed {before - after})")
    results[step] = df_clean

    if do_clustering:
        step = "10a_kmeans"
        step_dir = _step_dir(base, metric, step)
        uc_clean = gl.unique_coords(df_clean, lat_col, lon_col)
        cluster_results = {}
        for n_clusters in n_clusters_list:
            print(f"[10a] KMeans  n={n_clusters}  →  {step_dir}/")
            out_csv = os.path.join(step_dir, f"clusters_{n_clusters}.csv")
            if skip_if_exists and os.path.isfile(out_csv):
                clustered = pd.read_csv(out_csv)
            else:
                clustered, centers = gl.kmeans_clustering(
                    uc_clean, lat_col, lon_col, n_clusters
                )
                gl.save_csv(clustered, out_csv)
                gl.save_csv(centers, os.path.join(step_dir, f"centers_{n_clusters}.csv"))
            cluster_results[n_clusters] = clustered
            print(f"    Clusters: {clustered['cluster'].nunique()}")
        results[step] = cluster_results

        step = "10b_cluster_labels"
        step_dir = _step_dir(base, metric, step)
        best_n = max(n_clusters_list) if n_clusters_list else 5
        clustered = cluster_results[best_n]
        cluster_map = clustered[[lat_col, lon_col, "cluster", "distance_km", "lat_center", "lon_center"]]
        out = os.path.join(step_dir, f"labelled_{best_n}clusters.csv")
        print(f"[10b] Merge cluster labels (n={best_n})  →  {out}")
        if skip_if_exists and os.path.isfile(out):
            df_labelled = pd.read_csv(out)
        else:
            df_labelled = df_clean.merge(cluster_map, on=[lat_col, lon_col], how="left")
            gl.save_csv(df_labelled, out)
        results[step] = df_labelled

        group_cols = ["cluster"]
        step = "11a_weekly_cluster"
        step_dir = _step_dir(base, metric, step)
        out = os.path.join(step_dir, "weekly_cluster.csv")
        print(f"[11a] Weekly aggregation per cluster  →  {out}")
        if skip_if_exists and os.path.isfile(out):
            weekly = pd.read_csv(out)
        else:
            weekly = gl.weekly_aggregation(
                df_labelled, group_cols, value_col, time_col,
                extra_cols=['lat_center', 'lon_center']
            )
            gl.save_csv(weekly, out)
        print(f"    Weeks: {weekly[time_col].nunique()}")
        results[step] = weekly

        step = "12a_fill_weekly_cluster"
        step_dir = _step_dir(base, metric, step)
        out = os.path.join(step_dir, "filled_weekly_cluster.csv")
        print(f"[12a] Fill null weekly averages (per cluster)  →  {out}")
        if skip_if_exists and os.path.isfile(out):
            filled = pd.read_csv(out)
        else:
            filled = gl.fill_weekly_averages(
                weekly, time_col, "avg", group_cols
            )
            gl.save_csv(filled, out)
        nulls_before = weekly["avg"].isna().sum()
        nulls_after = filled["avg"].isna().sum()
        print(f"    Null avgs: {nulls_before} → {nulls_after}")
        results[step] = filled

    else:
        group_cols = [lat_col, lon_col]

        step = "11b_weekly_location"
        step_dir = _step_dir(base, metric, step)
        out = os.path.join(step_dir, "weekly_location.csv")
        print(f"[11b] Weekly aggregation per ({lat_col}, {lon_col})  →  {out}")
        if skip_if_exists and os.path.isfile(out):
            weekly = pd.read_csv(out)
        else:
            weekly = gl.weekly_aggregation(
                df_clean, group_cols, value_col, time_col
            )
            gl.save_csv(weekly, out)
        print(f"    Weeks: {weekly[time_col].nunique()},  Locations: {weekly.groupby(group_cols).ngroups}")
        results[step] = weekly

        step = "12b_fill_weekly_location"
        step_dir = _step_dir(base, metric, step)
        out = os.path.join(step_dir, "filled_weekly_location.csv")
        print(f"[12b] Fill null weekly averages (per location)  →  {out}")
        if skip_if_exists and os.path.isfile(out):
            filled = pd.read_csv(out)
        else:
            filled = gl.fill_weekly_averages(
                weekly, time_col, "avg", group_cols
            )
            gl.save_csv(filled, out)
        nulls_before = weekly["avg"].isna().sum()
        nulls_after = filled["avg"].isna().sum()
        print(f"    Null avgs: {nulls_before} → {nulls_after}")
        results[step] = filled

    print(f"{'─' * 70}")
    print(f"  Finished  {metric.name}")
    print(f"  Final output:  {len(filled)} rows,  columns: {list(filled.columns)}")
    print(f"{'─' * 70}")

    results["final"] = filled
    return results
