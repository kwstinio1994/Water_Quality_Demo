import os
import pandas as pd

import pipeline_code.generic.generic_library as gl
from pipeline_code.configurations.config import (
    Config, Project, attributes,
    weekly_coords, wqi_weights, wqi_ranges, wqi_classes,
)
from pipeline_code.scoring import scoring_library as sl


def run_combine_pipeline(
    config,
    project,
    output_dir="data",
    *,
    base_metric="oxygen",
    value_prefix="avg",
    time_col="time",
    skip_if_exists=False,
):
    base_dir = os.path.join(output_dir, project.name)
    combined_dir = os.path.join(base_dir, "combined")
    wqi_dir = os.path.join(base_dir, "wqi")
    os.makedirs(combined_dir, exist_ok=True)
    os.makedirs(wqi_dir, exist_ok=True)

    target_metrics = [m.name for m in attributes if m.name != base_metric]

    print(f"{'=' * 70}")
    print(f"  Combining metrics — base: {base_metric}")
    print(f"  Targets: {', '.join(target_metrics)}")
    print(f"{'=' * 70}")

    print(f"\n[1/6] Loading base metric: {base_metric}")
    base_df = sl.load_filled_weekly(base_dir, project.name, base_metric)
    base_coords = sl.unique_coords_from_weekly(base_df, weekly_coords[base_metric])
    base_cols = weekly_coords[base_metric]
    print(f"  Rows: {len(base_df)},  Unique coords: {len(base_coords)}")

    pairwise_results = {}

    for i, tgt in enumerate(target_metrics, 1):
        print(f"\n[2/6] Pairwise: {base_metric} ← {tgt}  ({i}/{len(target_metrics)})")

        out_csv = os.path.join(combined_dir, f"combined_{base_metric}_{tgt}.csv")
        if skip_if_exists and os.path.isfile(out_csv):
            print(f"  Loading existing  →  {out_csv}")
            pairwise_results[tgt] = pd.read_csv(out_csv)
            continue

        tgt_df = sl.load_filled_weekly(base_dir, project.name, tgt)
        tgt_coords = sl.unique_coords_from_weekly(tgt_df, weekly_coords[tgt])
        tgt_cols = weekly_coords[tgt]
        print(f"  Target rows: {len(tgt_df)},  Unique coords: {len(tgt_coords)}")

        coord_map = sl.build_coord_map(
            base_coords, tgt_coords,
            base_cols, tgt_cols,
            base_metric, tgt,
        )
        mean_dist = coord_map["distance_km"].mean()
        print(f"  Mean match distance: {mean_dist:.2f} km")

        result = sl.join_weekly_data(base_df, tgt_df, coord_map, base_metric, tgt, time_col)
        gl.save_csv(result, out_csv)
        print(f"  Combined rows: {len(result)}  →  {out_csv}")
        pairwise_results[tgt] = result

    print(f"\n[3/6] Master merge — combining all pairs")

    out_csv = os.path.join(combined_dir, "combined_metrics.csv")
    skip = skip_if_exists and os.path.isfile(out_csv)

    if skip:
        master = pd.read_csv(out_csv)
    else:
        base_keep = [
            f"latitude_{base_metric}",
            f"longitude_{base_metric}",
            time_col,
        ]
        master = pairwise_results[target_metrics[0]].copy()
        for tgt in target_metrics[1:]:
            keep_cols = base_keep + [c for c in pairwise_results[tgt].columns
                                      if c not in master.columns]
            master = master.merge(pairwise_results[tgt][keep_cols], on=base_keep, how="left")

        for c in master.columns:
            if "Unnamed" in c:
                master.drop(columns=[c], inplace=True)
        gl.save_csv(master, out_csv)

    print(f"  Master rows: {len(master)},  Columns: {len(master.columns)}  →  {out_csv}")

    print(f"\n[4/6] Rating indices and flags")

    out_ri = os.path.join(wqi_dir, "rating_indicator_metrics.csv")
    skip = skip_if_exists and os.path.isfile(out_ri)

    if skip:
        ri_df = pd.read_csv(out_ri)
    else:
        ri_df = sl.compute_wqi(master.copy(), wqi_weights, wqi_ranges, value_prefix)
        gl.save_csv(ri_df, out_ri)

    print(f"  Rows: {len(ri_df)}  →  {out_ri}")

    print(f"\n[5/6] Classifying water quality index")

    out_wqi = os.path.join(wqi_dir, "water_quality_index.csv")
    skip = skip_if_exists and os.path.isfile(out_wqi)

    if skip:
        wqi_df = pd.read_csv(out_wqi)
    else:
        wqi_df = sl.classify_wqi(ri_df, wqi_classes)
        gl.save_csv(wqi_df, out_wqi)

    dist = wqi_df["water_result"].value_counts()
    print(f"  Distribution: {dict(dist)}  →  {out_wqi}")

    print(f"\n{'─' * 70}")
    print(f"  Finished combining + wqi")
    print(f"  Combined:  {combined_dir}/")
    print(f"  Wqi:       {wqi_dir}/")
    print(f"{'─' * 70}")

    return {
        "pairwise": pairwise_results,
        "master": master,
        "ri": ri_df,
        "wqi": wqi_df,
    }
