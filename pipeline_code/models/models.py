import os
import pandas as pd

import pipeline_code.generic.generic_library as gl
from pipeline_code.configurations.config import (
    models_config,
    model_time_col, model_value_col, model_lat_col, model_lon_col, model_test_size,
)
from pipeline_code.models import models_library as ml


def run_models_pipeline(
    config,
    project,
    output_dir="data",
    *,
    skip_if_exists=False,
    produce_plots=True,
):
    base_dir = os.path.join(output_dir, project.name, "wqi")
    models_dir = os.path.join(output_dir, project.name, "models")
    figures_dir = os.path.join(models_dir, "figures")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print(f"{'=' * 70}")
    print(f"  Machine-learning models pipeline")
    print(f"{'=' * 70}")

    wqi_csv = os.path.join(base_dir, "water_quality_index.csv")
    print(f"\n[1] Loading wqi csv: {wqi_csv}")
    df = pd.read_csv(wqi_csv)
    for c in df.columns:
        if "Unnamed" in c:
            df.drop(columns=[c], inplace=True)
    print(f"  Rows: {len(df)},  Columns: {len(df.columns)}")

    cfg = models_config

    if cfg["seasonal_decomp"]["enabled"]:
        print(f"\n[2] Seasonal decomposition")
        out_dir = os.path.join(figures_dir, "seasonal_decomp")
        os.makedirs(out_dir, exist_ok=True)
        out_csv = os.path.join(models_dir, "decomposition_components.csv")
        skip = skip_if_exists and os.path.isfile(out_csv)
        if skip:
            print(f"  Skipping (exists)")
        else:
            weekly = ml.aggregate_weekly(df, model_time_col, model_value_col)
            print(f"  Weekly series: {len(weekly)} points")
            if len(weekly) < cfg["seasonal_decomp"]["period"] * 2:
                print(f"  Warning: series shorter than 2 full periods, results may be unreliable")
            decomp = ml.seasonal_decomposition(
                weekly, period=cfg["seasonal_decomp"]["period"],
                model=cfg["seasonal_decomp"]["model"],
            )
            if decomp is None:
                print(f"  Not enough data for decomposition (need 2× period)")
            else:
                decomp_df = ml.decomp_to_df(decomp)
                gl.save_csv(decomp_df, out_csv)
                print(f"  Components saved  →  {out_csv}")
                if produce_plots:
                    plot_path = os.path.join(out_dir, "decomposition.png")
                    ml.plot_decomposition(decomp, f"WQI seasonal decomposition", plot_path)
                    print(f"  Plot  →  {plot_path}")

    if cfg["per_location_trend"]["enabled"]:
        print(f"\n[3] Per-location trend analysis")
        out_dir = os.path.join(figures_dir, "trend")
        os.makedirs(out_dir, exist_ok=True)
        out_csv = os.path.join(models_dir, "per_location_trend.csv")
        skip = skip_if_exists and os.path.isfile(out_csv)
        if skip:
            trend_df = pd.read_csv(out_csv)
            print(f"  Skipping (exists)")
        else:
            trend_df = ml.per_location_trend(
                df, model_time_col, model_lat_col, model_lon_col, model_value_col,
            )
            gl.save_csv(trend_df, out_csv)
            print(f"  Locations: {len(trend_df)}")
            improving = (trend_df["direction"] == "improving").sum()
            declining = (trend_df["direction"] == "declining").sum()
            print(f"  Improving: {improving},  Declining: {declining},  Stable: {len(trend_df) - improving - declining}")
            print(f"  →  {out_csv}")
            if produce_plots and len(trend_df) > 0:
                plot_path = os.path.join(out_dir, "trend_map.png")
                ml.plot_trend_map(trend_df, model_lat_col, model_lon_col, plot_path)
                print(f"  Map  →  {plot_path}")

    if cfg["arima"]["enabled"]:
        print(f"\n[4] ARIMA forecast")
        out_dir = os.path.join(figures_dir, "arima")
        os.makedirs(out_dir, exist_ok=True)
        out_fc = os.path.join(models_dir, "arima_forecast.csv")
        out_ev = os.path.join(models_dir, "arima_evaluation.csv")
        skip = skip_if_exists and os.path.isfile(out_fc)
        if skip:
            print(f"  Skipping (exists)")
        else:
            weekly = ml.aggregate_weekly(df, model_time_col, model_value_col)
            n_pred = cfg["arima"]["n_predictions"]
            print(f"  Series: {len(weekly)} weeks,  Forecast: {n_pred} weeks ahead")
            fitted, forecast_df, metrics = ml.arima_evaluate(
                weekly, model_test_size,
                seasonal=cfg["arima"]["seasonal"],
                period=cfg["arima"]["period"],
            )
            if metrics and "error" not in metrics:
                print(f"  Order: {metrics.get('order')},  AIC: {metrics.get('aic')}")
                print(f"  Test RMSE: {metrics.get('rmse')},  MAE: {metrics.get('mae')}")
                gl.save_csv(pd.DataFrame([metrics]), out_ev)
            else:
                print(f"  Evaluation skipped: {metrics.get('error', 'unknown')}")

            fitted2, fc_df, _ = ml.arima_evaluate(
                weekly, 0.0,
                seasonal=cfg["arima"]["seasonal"],
                period=cfg["arima"]["period"],
            )
            fc2, full_fc = ml.arima_forecast(
                weekly, n_pred,
                seasonal=cfg["arima"]["seasonal"],
                period=cfg["arima"]["period"],
            )
            gl.save_csv(full_fc, out_fc)
            print(f"  Forecast  →  {out_fc}")

    if cfg["prophet"]["enabled"]:
        print(f"\n[5] Prophet forecast")
        out_dir = os.path.join(figures_dir, "prophet")
        os.makedirs(out_dir, exist_ok=True)
        out_fc = os.path.join(models_dir, "prophet_forecast.csv")
        out_ev = os.path.join(models_dir, "prophet_evaluation.csv")
        skip = skip_if_exists and os.path.isfile(out_fc)
        if skip:
            print(f"  Skipping (exists)")
        else:
            weekly = ml.aggregate_weekly(df, model_time_col, model_value_col)
            n_pred = cfg["prophet"]["n_predictions"]
            print(f"  Series: {len(weekly)} weeks,  Forecast: {n_pred} weeks ahead")

            model, forecast, metrics = ml.prophet_evaluate(
                weekly, model_test_size,
                seasonality_mode=cfg["prophet"]["seasonality_mode"],
            )
            if metrics and "error" not in metrics:
                print(f"  Train: {metrics.get('n_train')},  Test: {metrics.get('n_test')}")
                print(f"  Test RMSE: {metrics.get('rmse')},  MAE: {metrics.get('mae')}")
                gl.save_csv(pd.DataFrame([metrics]), out_ev)
            else:
                print(f"  Evaluation: {metrics.get('error', 'unknown')}")

            model2, fc_full, _ = ml.prophet_forecast(
                weekly, n_pred,
                seasonality_mode=cfg["prophet"]["seasonality_mode"],
            )
            if fc_full is not None:
                fc_out = fc_full[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(n_pred).reset_index(drop=True)
                fc_out.columns = ["time", "forecast", "lower_ci", "upper_ci"]
                gl.save_csv(fc_out, out_fc)
                print(f"  Forecast  →  {out_fc}")

    print(f"\n{'─' * 70}")
    print(f"  Finished models pipeline")
    print(f"  Outputs: {models_dir}/")
    if produce_plots:
        print(f"  Figures: {figures_dir}/")
    print(f"{'─' * 70}")

    return {"models_dir": models_dir, "figures_dir": figures_dir}
