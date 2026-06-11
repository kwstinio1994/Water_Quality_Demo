import os
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
#  data preparation
# ---------------------------------------------------------------------------

def parse_time(df, time_col="time"):
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    return df


def aggregate_weekly(df, time_col="time", value_col="water_quality_index"):
    df = parse_time(df, time_col)
    weekly = df.set_index(time_col).resample("W-MON")[value_col].mean().dropna().reset_index()
    return weekly


def train_test_split(series, test_size=0.2):
    n = len(series)
    split = int(n * (1 - test_size))
    train = series.iloc[:split]
    test = series.iloc[split:]
    return train, test


# ---------------------------------------------------------------------------
#  evaluation metrics
# ---------------------------------------------------------------------------

def eval_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mse = np.mean((actual - predicted) ** 2)
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(mse)
    nonzero = np.abs(actual) > 1e-12
    mape = np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100 if nonzero.any() else 0.0
    return {
        "mse": round(mse, 4),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape": round(mape, 2),
    }


# ---------------------------------------------------------------------------
#  seasonal decomposition
# ---------------------------------------------------------------------------

def seasonal_decomposition(series, period=52, model="additive"):
    from statsmodels.tsa.seasonal import seasonal_decompose
    series = series.set_index("time") if "time" in series.columns else series
    series = series.iloc[:, 0] if series.ndim == 2 else series
    series = series.dropna().astype(float)
    effective = min(period, len(series) // 2)
    if effective < 2:
        return None
    result = seasonal_decompose(series, model=model, period=effective, extrapolate_trend="freq")
    return result


def decomp_to_df(result):
    df = pd.DataFrame({
        "trend": result.trend,
        "seasonal": result.seasonal,
        "residual": result.resid,
    })
    df.index.name = "time"
    return df.reset_index()


def plot_decomposition(result, title, output_path):
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    observed = result.observed
    trend = result.trend
    seasonal = result.seasonal
    resid = result.resid
    axes[0].plot(observed.index, observed.values, color="black")
    axes[0].set_ylabel("observed")
    axes[0].set_title(title)
    axes[1].plot(trend.index, trend.values, color="tab:blue")
    axes[1].set_ylabel("trend")
    axes[2].plot(seasonal.index, seasonal.values, color="tab:green")
    axes[2].set_ylabel("seasonal")
    axes[3].plot(resid.index, resid.values, color="tab:red", marker="o", linestyle="", markersize=2)
    axes[3].set_ylabel("residual")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
#  per-location trend
# ---------------------------------------------------------------------------

def per_location_trend(df, time_col="time", lat_col="latitude_oxygen",
                       lon_col="longitude_oxygen", value_col="water_quality_index"):
    from scipy.stats import linregress
    df = parse_time(df, time_col)
    df["_numtime"] = df[time_col].astype(np.int64) // 10 ** 9
    rows = []
    for (lat, lon), group in df.groupby([lat_col, lon_col]):
        group = group.dropna(subset=[value_col])
        if len(group) < 3:
            continue
        slope, intercept, rvalue, pvalue, stderr = linregress(
            group["_numtime"].values, group[value_col].values
        )
        rows.append({
            lat_col: lat, lon_col: lon,
            "slope": slope,
            "intercept": intercept,
            "rvalue": rvalue,
            "pvalue": pvalue,
            "stderr": stderr,
        })
    result = pd.DataFrame(rows)
    result["direction"] = result["slope"].apply(
        lambda s: "improving" if s > 0 else ("declining" if s < 0 else "stable")
    )
    return result


def plot_trend_map(trend_df, lat_col, lon_col, output_path):
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = trend_df["slope"].values
    sc = ax.scatter(trend_df[lon_col], trend_df[lat_col], c=colors,
                    cmap="RdYlGn", s=60, edgecolors="black", linewidth=0.5)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("per-location WQI trend (slope)")
    fig.colorbar(sc, ax=ax, label="slope")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
#  ARIMA
# ---------------------------------------------------------------------------

def arima_forecast(series, n_predictions=12, seasonal=True, period=52):
    series = series.copy()
    series = series.set_index("time") if "time" in series.columns else series
    series = series.iloc[:, 0] if series.ndim == 2 else series
    series = series.dropna().astype(float)

    try:
        import pmdarima as pmd
        auto_model = pmd.auto_arima(
            series, seasonal=seasonal, m=period if seasonal else 1,
            start_p=1, start_q=1, max_p=3, max_q=3,
            start_P=0, start_Q=0, max_P=2, max_Q=2,
            trace=False, error_action="ignore", stepwise=True,
        )
        order = auto_model.order
        seas_order = auto_model.seasonal_order
    except ImportError:
        from statsmodels.tsa.arima.model import ARIMA
        auto_model = ARIMA(series, order=(1, 0, 1))
        auto_model = auto_model.fit()
        order = auto_model.model.order
        seas_order = None

    from statsmodels.tsa.arima.model import ARIMA
    if seasonal and seas_order and seas_order[1] > 0:
        model = ARIMA(series, order=order, seasonal_order=seas_order)
    else:
        model = ARIMA(series, order=order)
    fitted = model.fit()

    forecast_result = fitted.get_forecast(steps=n_predictions)
    forecast_values = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int(alpha=0.05)
    forecast_df = pd.DataFrame({
        "forecast": forecast_values.values,
        "lower_ci": conf_int.iloc[:, 0].values if conf_int.ndim == 2 else conf_int.values[:, 0],
        "upper_ci": conf_int.iloc[:, 1].values if conf_int.ndim == 2 else conf_int.values[:, 1],
    })
    last_time = pd.to_datetime(series.index[-1]) if hasattr(series.index, '__array__') else pd.to_datetime(list(series.index)[-1])
    forecast_df.index = pd.date_range(start=last_time + pd.Timedelta(weeks=1), periods=n_predictions, freq="W-MON")
    forecast_df.index.name = "time"

    return fitted, forecast_df.reset_index()


def arima_evaluate(series, test_size=0.2, seasonal=True, period=52):
    series = series.copy()
    series = series.set_index("time") if "time" in series.columns else series
    series = series.iloc[:, 0] if series.ndim == 2 else series
    series = series.dropna().astype(float)

    train, test = train_test_split(series, test_size)
    if len(test) < 2:
        return None, None, {"error": "test set too small"}

    try:
        import pmdarima as pmd
        auto_model = pmd.auto_arima(
            train, seasonal=seasonal, m=period if seasonal else 1,
            start_p=1, start_q=1, max_p=3, max_q=3,
            trace=False, error_action="ignore", stepwise=True,
        )
        order = auto_model.order
        seas_order = auto_model.seasonal_order
    except ImportError:
        from statsmodels.tsa.arima.model import ARIMA
        auto_model = ARIMA(train, order=(1, 0, 1)).fit()
        order = auto_model.model.order
        seas_order = None

    from statsmodels.tsa.arima.model import ARIMA
    if seasonal and seas_order and seas_order[1] > 0:
        model = ARIMA(train, order=order, seasonal_order=seas_order)
    else:
        model = ARIMA(train, order=order)
    fitted = model.fit()

    forecast = fitted.forecast(steps=len(test))
    metrics = eval_metrics(test.values, forecast.values)
    metrics["order"] = order
    if seas_order:
        metrics["seasonal_order"] = seas_order
    metrics["aic"] = round(fitted.aic, 2)

    return fitted, forecast, metrics


# ---------------------------------------------------------------------------
#  Prophet
# ---------------------------------------------------------------------------

def prophet_forecast(series, n_predictions=12, seasonality_mode="additive", period=52):
    try:
        from prophet import Prophet
    except ImportError:
        return None, None, {"error": "prophet not installed"}

    series = series.copy()
    series = series.set_index("time") if "time" in series.columns else series
    series = series.iloc[:, 0] if series.ndim == 2 else series
    series = series.dropna().astype(float).reset_index()
    series.columns = ["ds", "y"]

    model = Prophet(
        seasonality_mode=seasonality_mode,
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
    )
    model.fit(series)

    future = model.make_future_dataframe(periods=n_predictions, freq="W-MON")
    forecast = model.predict(future)
    return model, forecast


def prophet_evaluate(series, test_size=0.2, seasonality_mode="additive"):
    try:
        from prophet import Prophet
    except ImportError:
        return None, None, {"error": "prophet not installed"}

    series = series.copy()
    series = series.set_index("time") if "time" in series.columns else series
    series = series.iloc[:, 0] if series.ndim == 2 else series
    series = series.dropna().astype(float).reset_index()
    series.columns = ["ds", "y"]

    n = len(series)
    split = int(n * (1 - test_size))
    train = series.iloc[:split]
    test = series.iloc[split:]

    if len(test) < 2:
        return None, None, {"error": "test set too small"}

    model = Prophet(
        seasonality_mode=seasonality_mode,
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
    )
    model.fit(train)

    future = test[["ds"]].copy()
    forecast = model.predict(future)
    merged = test.merge(forecast[["ds", "yhat"]], on="ds", how="left")
    metrics = eval_metrics(merged["y"].values, merged["yhat"].values)
    metrics["n_train"] = len(train)
    metrics["n_test"] = len(test)

    return model, forecast, metrics
