# Water Quality Demo

> **WIP** — Work in Progress

This directory contains the original exploratory work and the ongoing pipeline refactor for assessing coastal marine water quality using Copernicus Marine Service data.

## Contents

- **`1.Metrics/`** — Raw data download scripts + initial & further analysis for each of the 5 water quality metrics (chlorophyll, turbidity, pH, temperature, dissolved oxygen)
- **`2.Combine Metrics/`** — Spatial joining of the 5 metrics via nearest-neighbor matching (KDTree / BallTree)
- **`3.Water Quality Index/`** — Rating Indicator computation, weighted WQI calculation, correlation analysis, seasonal breakdowns, per-spot insights, ARIMA forecasting
- **`4.Swimming Index/`** — Re-weighted WQI for swimming suitability
- **`5.Fishing Index/`** — Re-weighted WQI for fishing/aquaculture suitability
- **`6.Presentations/`** — Progress presentations
- **`7.References/`** — Academic papers, water quality standards, reference data
- **`pipeline_code/`** — Modular pipeline being refactored from the original scripts
  - `configurations/config.py` — Config, Project, MetricDef definitions
  - `data_download/download.py` — Concurrent download module

## Pipeline usage

```python
from pipeline_code.configurations.config import Config, Project
from pipeline_code.data_download.download import download_all

cfg = Config()
project = Project(
    name="my_site",
    lon_min=25.0, lon_max=25.5,
    lat_min=39.7, lat_max=40.1,
    date_start="2020-01-01",
    date_end="2022-09-30"
)

results = download_all(cfg, project)
```
