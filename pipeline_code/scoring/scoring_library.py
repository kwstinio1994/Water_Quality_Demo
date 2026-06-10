import os
import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree, BallTree
from numpy import cos, sin, arcsin, sqrt
from math import radians

from pipeline_code.configurations.config import weekly_coords


def _filled_step_dir(base_dir, metric_name):
    coord_cols = weekly_coords[metric_name]
    if coord_cols == ["lat_center", "lon_center"]:
        return os.path.join(base_dir, metric_name, "12a_fill_weekly_cluster")
    return os.path.join(base_dir, metric_name, "12b_fill_weekly_location")


def load_filled_weekly(base_dir, project_name, metric_name):
    step_dir = _filled_step_dir(base_dir, metric_name)
    path = os.path.join(step_dir, "filled_weekly_cluster.csv" if "cluster" in step_dir else "filled_weekly_location.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"filled weekly not found at {path}")
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)
    return df


def unique_coords_from_weekly(df, coord_cols):
    return df[coord_cols].drop_duplicates().reset_index(drop=True)


def match_coords_kdtree(base_coords, target_coords):
    kd = KDTree(target_coords.values, metric="euclidean")
    distances, indices = kd.query(base_coords.values, k=1)
    return distances.flatten(), indices.flatten()


def match_coords_balltree(base_coords, target_coords):
    base_rad = np.radians(base_coords.values)
    ball = BallTree(base_rad, metric="haversine")
    target_rad = np.radians(target_coords.values)
    distances, indices = ball.query(target_rad, k=1)
    return distances.flatten(), indices.flatten()


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * arcsin(sqrt(a))
    return 6367 * c


def build_coord_map(base_coords, target_coords, base_cols, target_cols, base_name, target_name):
    distances, indices = match_coords_kdtree(base_coords, target_coords)

    base_label = [f"latitude_{base_name}", f"longitude_{base_name}"]
    target_label = [f"latitude_{target_name}", f"longitude_{target_name}"]

    result = base_coords.copy()
    result.columns = base_label

    matched = target_coords.iloc[indices].reset_index(drop=True)
    matched.columns = target_label
    result = pd.concat([result, matched], axis=1)

    result["distance_km"] = result.apply(
        lambda r: haversine(r[base_label[0]], r[base_label[1]], r[target_label[0]], r[target_label[1]]),
        axis=1,
    )
    return result


def join_weekly_data(base_df, target_df, coord_map, base_name, target_name, time_col="time"):
    base_label = [f"latitude_{base_name}", f"longitude_{base_name}"]
    target_label = [f"latitude_{target_name}", f"longitude_{target_name}"]
    base_coord_cols = weekly_coords[base_name]
    target_coord_cols = weekly_coords[target_name]

    base_renamed = base_df.rename(
        columns={base_coord_cols[0]: base_label[0], base_coord_cols[1]: base_label[1]}
    )
    rename_map = {c: f"{c}_{base_name}" for c in base_renamed.columns
                  if c not in base_label + [time_col]}
    base_renamed.rename(columns=rename_map, inplace=True)

    target_renamed = target_df.rename(
        columns={target_coord_cols[0]: target_label[0], target_coord_cols[1]: target_label[1]}
    )
    rename_map = {c: f"{c}_{target_name}" for c in target_renamed.columns
                  if c not in target_label + [time_col]}
    target_renamed.rename(columns=rename_map, inplace=True)

    result = coord_map.merge(base_renamed, on=base_label, how="left")
    result = result.merge(target_renamed, on=target_label + [time_col], how="left")
    return result


def ri_optimal(value, ideal_low, ideal_high, accept_low, accept_high):
    ri = np.ones_like(value, dtype=float)
    below = (value >= accept_low) & (value < ideal_low)
    above = (value > ideal_high) & (value <= accept_high)
    outside = (value < accept_low) | (value > accept_high)
    ri[below] = (value[below] - accept_low) / (ideal_low - accept_low)
    ri[above] = (accept_high - value[above]) / (accept_high - ideal_high)
    ri[outside] = 0.0
    return ri


def ri_lower(value, ideal_high, accept_high):
    ri = np.ones_like(value, dtype=float)
    acc = (value > ideal_high) & (value <= accept_high)
    out = value > accept_high
    ri[acc] = (accept_high - value[acc]) / (accept_high - ideal_high)
    ri[out] = 0.0
    return ri


def ri_higher(value, ideal_low, accept_low):
    ri = np.ones_like(value, dtype=float)
    acc = (value >= accept_low) & (value < ideal_low)
    out = value < accept_low
    ri[acc] = (value[acc] - accept_low) / (ideal_low - accept_low)
    ri[out] = 0.0
    return ri


def flag_optimal(value, ideal_low, ideal_high, accept_low, accept_high):
    flag = np.full_like(value, 2, dtype=int)
    flag[(value >= ideal_low) & (value <= ideal_high)] = 0
    flag[((value >= accept_low) & (value < ideal_low)) | ((value > ideal_high) & (value <= accept_high))] = 1
    return flag


def flag_lower(value, ideal_high, accept_high):
    flag = np.full_like(value, 2, dtype=int)
    flag[value <= ideal_high] = 0
    flag[(value > ideal_high) & (value <= accept_high)] = 1
    return flag


def flag_higher(value, ideal_low, accept_low):
    flag = np.full_like(value, 2, dtype=int)
    flag[value >= ideal_low] = 0
    flag[(value >= accept_low) & (value < ideal_low)] = 1
    return flag


def compute_ri_flag(series, range_def):
    t = range_def["type"]
    if t == "optimal":
        ri = ri_optimal(series.values, *range_def["ideal"], *range_def["acceptable"])
        fl = flag_optimal(series.values, *range_def["ideal"], *range_def["acceptable"])
    elif t == "lower_better":
        ri = ri_lower(series.values, range_def["ideal"], range_def["acceptable"])
        fl = flag_lower(series.values, range_def["ideal"], range_def["acceptable"])
    elif t == "higher_better":
        ri = ri_higher(series.values, range_def["ideal"], range_def["acceptable"])
        fl = flag_higher(series.values, range_def["ideal"], range_def["acceptable"])
    else:
        raise ValueError(f"unknown range type: {t}")
    return ri, fl


def compute_wqi(df, weights, ranges, value_prefix="avg"):
    wqi = np.zeros(len(df))
    for name, weight in weights.items():
        col = f"{value_prefix}_{name}"
        if col not in df.columns:
            continue
        ri, fl = compute_ri_flag(df[col], ranges[name])
        df[f"ri_{name}"] = ri
        df[f"flag_{name}"] = fl
        wqi += weight * ri

    df["water_quality_index"] = (wqi * 10).round(decimals=2)
    return df


def classify_wqi(df, classes):
    bins = [c[0] for c in classes] + [classes[-1][1]]
    labels = [c[2] for c in classes]
    df["water_result"] = pd.cut(
        df["water_quality_index"],
        bins=bins,
        labels=labels,
        right=False,
    )
    return df
