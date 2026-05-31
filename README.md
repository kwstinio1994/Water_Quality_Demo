# Water Quality Assessment Pipeline

> **WIP** — Work in Progress

A Python pipeline for assessing coastal marine water quality using satellite and model data from the **Copernicus Marine Service (CMEMS)**.

## Purpose

Monitor and evaluate coastal water quality in the **Pagasetic Gulf / Thermaic Gulf** region of Greece by computing **Water Quality Indices (WQI)** tailored to different use cases — general environmental quality, swimming suitability, and fishing/aquaculture suitability.

## Workflow

1. **Data acquisition** — Download 5 water quality metrics from CMEMS (chlorophyll, turbidity, pH, temperature, dissolved oxygen) via `motuclient`
2. **Spatial joining** — Combine metrics by nearest-neighbor matching on coordinates
3. **WQI computation** — Normalize each metric into a Rating Indicator (0–1), compute weighted WQI (0–10 scale), and classify results (Excellent / Good / Medium / Bad / Very Bad)
4. **Forecasting** — ARIMA time-series forecasting of WQI values
5. **Analysis** — Seasonal breakdowns, per-spot insights, correlation analysis

## Status

Currently refactoring the original exploratory scripts into a modular pipeline:

```
pipeline_code/
  configurations/   — Config, Project definition, metric metadata
  data_download/    — Concurrent downloading of all metrics
  generic_cleanup/  — (planned)
  ...
```

## Project structure

The original work is in `water_quality_demo/` with numbered directories (`1.Metrics/` through `7.References/`). The pipeline code is being organized under `pipeline_code/`.

## Requirements

- Python 3.10+
- `motuclient` (Copernicus Marine Service client)
- `xarray`, `netCDF4`, `pandas`, `numpy`
- `scikit-learn`, `statsmodels`, `pmdarima`
- Copernicus Marine credentials (set in `.env`)
