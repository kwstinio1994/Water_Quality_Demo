import os
import pandas as pd
import numpy as np
import xarray as xr
from sklearn.cluster import KMeans
from numpy import cos, sin, arcsin, sqrt
from math import radians


def load_netcdf(nc_path, metric):
    ds = xr.open_dataset(nc_path)
    df = ds.to_dataframe().reset_index()
    if metric.drop_cols:
        df = df.drop(columns=metric.drop_cols, errors='ignore')
    if metric.rename_cols:
        df.rename(columns=metric.rename_cols, inplace=True)
    return df


def save_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def collapse_depth(df, lat_col, lon_col, time_col, value_col, agg_map=None):
    if agg_map is None:
        agg_map = ["mean", "min", "max"]
    if isinstance(agg_map, dict):
        agg_map = list(agg_map.values())
    result = df.groupby([lat_col, lon_col, time_col], observed=True)[value_col].agg(
        agg_map
    ).reset_index()
    names = {a: f"{value_col}_{a}" for a in agg_map}
    names["mean"] = value_col
    result.rename(columns=names, inplace=True)
    if f"{value_col}_min" in result.columns and f"{value_col}_max" in result.columns:
        result[f"{value_col}_diff"] = result[f"{value_col}_max"] - result[f"{value_col}_min"]
    return result


def value_column(metric):
    if metric.value_col:
        return metric.value_col
    if metric.rename_cols and metric.variable in metric.rename_cols:
        return metric.rename_cols[metric.variable]
    return metric.variable


def compute_date_range(df, lat_col, lon_col, time_col='time'):
    return df.groupby([lat_col, lon_col], observed=True)[time_col].agg(['min', 'max']).reset_index()


def compute_coord_range(df, time_col='time', lat_col='latitude', lon_col='longitude'):
    coords_per_date = df.groupby(time_col, observed=True)[[lat_col, lon_col]].nunique().reset_index()
    # count duplicate (lat, lon) rows within each date
    dupes_per_date = (
        df.groupby(time_col, observed=True)
        .apply(lambda g: g.duplicated(subset=[lat_col, lon_col]).sum(), include_groups=False)
        .reset_index(name='duplicates')
    )
    result = coords_per_date.merge(dupes_per_date, on=time_col)
    result.rename(
        columns={lat_col: 'n_latitudes', lon_col: 'n_longitudes'}, inplace=True
    )
    return result


def unique_coords(df, lat_col='latitude', lon_col='longitude'):
    return df[[lat_col, lon_col]].drop_duplicates().reset_index(drop=True)


def plot_coords_map(coords, lat_col, lon_col, region, title, output_path):
    try:
        import pygmt
    except ImportError:
        print("  pygmt not installed — skipping coordinate map")
        return
    fig = pygmt.Figure()
    fig.coast(
        region=region, projection='M6i',
        frame='afg', shorelines=True, land='gray', water='lightblue',
    )
    fig.plot(
        x=coords[lon_col], y=coords[lat_col],
        style='c0.05c', color='red', label='Sampling points',
    )
    fig.basemap(frame=['afg', f'+t{title}'])
    fig.legend()
    fig.savefig(output_path)


def analyze_nulls(df, lat_col, lon_col, time_col='time', value_cols=None):
    if value_cols is None:
        value_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    results = {}
    for vc in value_cols:
        by_loc = (
            df.groupby([lat_col, lon_col], observed=True)[vc]
            .apply(lambda x: x.isnull().sum())
            .reset_index(name='nulls')
        )
        results[f'nulls_by_loc_{vc}'] = by_loc
    for vc in value_cols:
        by_date = (
            df.groupby(time_col, observed=True)[vc]
            .apply(lambda x: x.isnull().sum())
            .reset_index(name='nulls')
        )
        results[f'nulls_by_date_{vc}'] = by_date
    return results


def analyze_null_gaps(df, lat_col, lon_col, time_col='time', value_col=None):
    if value_col is None:
        value_col = df.select_dtypes(include=[np.number]).columns[0]

    # longest run of consecutive nulls per location
    def _max_gap(series):
        is_null = series.isnull().astype(int)
        runs = (is_null != is_null.shift()).cumsum()
        gaps = is_null.groupby(runs).sum()
        return gaps.max() if len(gaps) > 0 else 0

    return (
        df.sort_values([lat_col, lon_col, time_col])
        .groupby([lat_col, lon_col], observed=True)[value_col]
        .apply(_max_gap)
        .reset_index(name='max_consecutive_nulls')
    )


def remove_full_null_series(df, lat_col, lon_col, value_col):
    null_cnt = (
        df.groupby([lat_col, lon_col], observed=True)[value_col]
        .apply(lambda x: x.isnull().sum())
        .reset_index(name='null_count')
    )
    total = (
        df.groupby([lat_col, lon_col], observed=True)[value_col]
        .size()
        .reset_index(name='total')
    )
    merged = null_cnt.merge(total, on=[lat_col, lon_col])
    merged['flag'] = (merged['null_count'] == merged['total']).astype(int)
    valid = merged[merged['flag'] == 0][[lat_col, lon_col]]
    return df.merge(valid, on=[lat_col, lon_col], how='inner')


def kmeans_clustering(coords, lat_col, lon_col, n_clusters, random_state=0):
    X = coords[[lat_col, lon_col]].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    kmeans.fit(X)
    result = coords.copy()
    result['cluster'] = kmeans.labels_
    centers = pd.DataFrame(
        kmeans.cluster_centers_, columns=['lat_center', 'lon_center'],
    )
    centers['cluster'] = range(n_clusters)
    result = result.merge(centers, on='cluster', how='left')
    result['distance_km'] = result.apply(
        lambda r: haversine_dist(r[lat_col], r[lon_col], r['lat_center'], r['lon_center']),
        axis=1,
    )
    return result, centers


def haversine_dist(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * arcsin(sqrt(a))
    return 6367 * c


def weekly_aggregation(df, group_cols, value_col, time_col='time', week_start='W-MON', extra_cols=None):
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    all_group_cols = group_cols + (extra_cols or [])
    grouper = all_group_cols + [pd.Grouper(key=time_col, freq=week_start)]

    mns = df.groupby(grouper, observed=True)[value_col].min().reset_index().rename(
        columns={value_col: 'min'}
    )
    mxs = df.groupby(grouper, observed=True)[value_col].max().reset_index().rename(
        columns={value_col: 'max'}
    )
    avs = df.groupby(grouper, observed=True)[value_col].mean().reset_index().rename(
        columns={value_col: 'avg'}
    )
    mds = df.groupby(grouper, observed=True)[value_col].median().reset_index().rename(
        columns={value_col: 'median'}
    )
    sds = df.groupby(grouper, observed=True)[value_col].std().reset_index().rename(
        columns={value_col: 'std'}
    )
    cnt = df.groupby(grouper, observed=True)[value_col].count().reset_index().rename(
        columns={value_col: 'count'}
    )

    merge_on = all_group_cols + [time_col]
    result = mns.merge(mxs, on=merge_on, how='left')
    result = result.merge(avs, on=merge_on, how='left')
    result = result.merge(mds, on=merge_on, how='left')
    result = result.merge(sds, on=merge_on, how='left')
    result = result.merge(cnt, on=merge_on, how='left')

    result['flag_avg'] = result['avg'].notna().astype(int)
    result['max_diff'] = result['max'] - result['min']
    return result


def fill_weekly_averages(df, time_col='time', value_col='avg', group_cols=None):
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    first = df[time_col].min()
    last = df[time_col].max()

    if group_cols:
        idx = (df[time_col] != first) & (df[time_col] != last)
        df.loc[idx, value_col] = df.groupby(group_cols, observed=True)[value_col].transform(
            lambda s: s.interpolate(limit=6)
        )
        df.loc[df[time_col] == first, value_col] = df.groupby(
            group_cols, observed=True
        )[value_col].transform(lambda s: s.bfill())
        df.loc[df[time_col] == last, value_col] = df.groupby(
            group_cols, observed=True
        )[value_col].transform(lambda s: s.ffill())
    else:
        idx = (df[time_col] != first) & (df[time_col] != last)
        df.loc[idx, value_col] = df[value_col].interpolate(limit=6)
        df.loc[df[time_col] == first, value_col] = df[value_col].bfill()
        df.loc[df[time_col] == last, value_col] = df[value_col].ffill()
    return df
