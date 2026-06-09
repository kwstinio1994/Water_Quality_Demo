import os
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
#  data preparation
# ---------------------------------------------------------------------------

def assign_seasons(df, time_col="time", season_map=None):
    if season_map is None:
        season_map = {1: "spring", 2: "spring", 3: "spring",
                      4: "summer", 5: "summer", 6: "summer",
                      7: "autumn", 8: "autumn", 9: "autumn",
                      10: "winter", 11: "winter", 12: "winter"}
    df["season"] = pd.to_datetime(df[time_col]).dt.month.map(season_map)
    return df


# ---------------------------------------------------------------------------
#  analysis functions
# ---------------------------------------------------------------------------

def seasonal_metric_breakdown(df, metrics, flag_cols, season_col="season"):
    rows = []
    for season in df[season_col].unique():
        season_df = df[df[season_col] == season]
        total = len(season_df)
        row = {"season": season, "total_count": total}
        for name, flag_col in zip(metrics, flag_cols):
            ideal = (season_df[flag_col] == 0).sum()
            acceptable = (season_df[flag_col] == 1).sum()
            non_acceptable = (season_df[flag_col] == 2).sum()
            row[f"ideal_{name}"] = ideal
            row[f"acceptable_{name}"] = acceptable
            row[f"non_acceptable_{name}"] = non_acceptable
            row[f"ideal_pct_{name}"] = round(ideal / total * 100, 2) if total else 0.0
            row[f"acceptable_pct_{name}"] = round(acceptable / total * 100, 2) if total else 0.0
            row[f"non_acceptable_pct_{name}"] = round(non_acceptable / total * 100, 2) if total else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def spot_breakdown(df, lat_col, lon_col, wqi_col, classes):
    classes = list(classes)
    result = df.groupby([lat_col, lon_col], as_index=False)[wqi_col].count()
    result.rename(columns={wqi_col: "total_count"}, inplace=True)
    for (lo, hi, label) in classes:
        short = label.lower().replace(" ", "_")
        count_col = f"count_{short}"
        if lo == -1e9:
            mask = df[wqi_col] < hi
        elif hi == 1e9:
            mask = df[wqi_col] >= lo
        else:
            mask = (df[wqi_col] >= lo) & (df[wqi_col] < hi)
        counts = df[mask].groupby([lat_col, lon_col], as_index=False)[wqi_col].count()
        counts.rename(columns={wqi_col: count_col}, inplace=True)
        result = result.merge(counts, on=[lat_col, lon_col], how="left")
    result.fillna(0, inplace=True)
    count_cols = [c for c in result.columns if c.startswith("count_")]
    result["sum_total"] = result[count_cols].sum(axis=1)
    for (lo, hi, label) in classes:
        short = label.lower().replace(" ", "_")
        result[f"{short}_pct"] = round(
            result[f"count_{short}"] / result["total_count"] * 100, 2
        )
    return result


def spot_season_breakdown(df, lat_col, lon_col, season_col, wqi_col, classes):
    groups = df.groupby([season_col, lat_col, lon_col])
    total = groups[wqi_col].count().reset_index(name="total_count")
    for (lo, hi, label) in classes:
        short = label.lower().replace(" ", "_")
        mask = ((df[wqi_col] >= lo) & (df[wqi_col] < hi))
        if lo == -1e9:
            mask = df[wqi_col] < hi
        elif hi == 1e9:
            mask = df[wqi_col] >= lo
        counts = df[mask].groupby([season_col, lat_col, lon_col]).size().reset_index(name=f"count_{short}")
        total = total.merge(counts, on=[season_col, lat_col, lon_col], how="left")
    total.fillna(0, inplace=True)
    count_cols = [c for c in total.columns if c.startswith("count_")]
    total["sum_total"] = total[count_cols].sum(axis=1)
    for (lo, hi, label) in classes:
        short = label.lower().replace(" ", "_")
        total[f"{short}_pct"] = round(
            total[f"count_{short}"] / total["total_count"] * 100, 2
        )
    return total


def holistic_breakdown(df, result_col):
    counts = df[result_col].value_counts().reset_index()
    counts.columns = [result_col, "count"]
    total = counts["count"].sum()
    counts["percentage"] = round(counts["count"] / total * 100, 2)
    return counts


def per_bin_breakdown(df, result_col, metrics, flag_cols, classes):
    labels = [label for (_, _, label) in classes]
    rows = []
    for label in labels:
        bin_df = df[df[result_col] == label]
        total = len(bin_df)
        row = {"result": label, "total_count": total}
        for name, flag_col in zip(metrics, flag_cols):
            ideal = (bin_df[flag_col] == 0).sum()
            acceptable = (bin_df[flag_col] == 1).sum()
            non_acceptable = (bin_df[flag_col] == 2).sum()
            row[f"ideal_{name}"] = ideal
            row[f"acceptable_{name}"] = acceptable
            row[f"non_acceptable_{name}"] = non_acceptable
            row[f"ideal_pct_{name}"] = round(ideal / total * 100, 2) if total else 0.0
            row[f"acceptable_pct_{name}"] = round(acceptable / total * 100, 2) if total else 0.0
            row[f"non_acceptable_pct_{name}"] = round(non_acceptable / total * 100, 2) if total else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def seasonal_index(df, season_col, result_col):
    groups = df.groupby([season_col, result_col]).size().reset_index(name="count")
    totals = df.groupby(season_col).size().reset_index(name="total_count")
    result = groups.merge(totals, on=season_col)
    result["percentage"] = round(result["count"] / result["total_count"] * 100, 2)
    return result


def what_excellent_means(df, result_col, metrics, flag_cols, excellent_label="Excellent"):
    excellent = df[df[result_col] == excellent_label]
    total = len(excellent)
    row = {"result": excellent_label, "total_count": total}
    for name, flag_col in zip(metrics, flag_cols):
        ideal = (excellent[flag_col] == 0).sum()
        acceptable = (excellent[flag_col] == 1).sum()
        non_acceptable = (excellent[flag_col] == 2).sum()
        row[f"ideal_{name}"] = ideal
        row[f"acceptable_{name}"] = acceptable
        row[f"non_acceptable_{name}"] = non_acceptable
        row[f"ideal_pct_{name}"] = round(ideal / total * 100, 2) if total else 0.0
        row[f"acceptable_pct_{name}"] = round(acceptable / total * 100, 2) if total else 0.0
        row[f"non_acceptable_pct_{name}"] = round(non_acceptable / total * 100, 2) if total else 0.0
    return pd.DataFrame([row])


def seasonal_averages(df, season_col, value_cols):
    return df.groupby(season_col)[value_cols].mean().round(2).reset_index()


# ---------------------------------------------------------------------------
#  plotting functions
# ---------------------------------------------------------------------------

def plot_correlation_heatmaps(df, group_col, value_cols, output_dir, fmt=".2g"):
    import matplotlib.pyplot as plt
    import seaborn as sns
    os.makedirs(output_dir, exist_ok=True)
    for label in df[group_col].unique():
        subset = df[df[group_col] == label][value_cols].dropna()
        if subset.empty:
            continue
        corr = subset.corr()
        cov = subset.cov()
        short = label.lower().replace(" ", "_")

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, vmin=-1, vmax=1, fmt=fmt, ax=ax)
        ax.set_title(f"Correlation — {label}", size=18)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"correlation_{short}.png"))
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cov, annot=True, cmap="coolwarm", center=0, fmt=fmt, ax=ax)
        ax.set_title(f"Covariance — {label}", size=18)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"covariance_{short}.png"))
        plt.close(fig)


def plot_pairplots(df, group_col, value_cols, output_dir):
    import matplotlib.pyplot as plt
    import seaborn as sns
    os.makedirs(output_dir, exist_ok=True)
    for label in df[group_col].unique():
        subset = df[df[group_col] == label][value_cols].dropna()
        if subset.empty:
            continue
        short = label.lower().replace(" ", "_")
        g = sns.pairplot(subset, height=2.0, diag_kind="kde", corner=True)
        g.map_lower(sns.kdeplot, levels=4, color=".2")
        g.fig.suptitle(f"Pairwise relationships — {label}", y=1.05, size=18)
        g.savefig(os.path.join(output_dir, f"pairplot_{short}.png"))
        plt.close(g.fig)


def plot_seasonal_bars(avg_df, season_col, value_cols, output_dir):
    import matplotlib.pyplot as plt
    import seaborn as sns
    os.makedirs(output_dir, exist_ok=True)
    for col in value_cols:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=avg_df, x=season_col, y=col, ax=ax)
        ax.set_title(f"Average {col} by season", size=14)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"seasonal_{col}.png"))
        plt.close(fig)
