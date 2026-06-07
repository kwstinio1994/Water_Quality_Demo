import os


class Project:
    def __init__(self, name, lon_min, lon_max, lat_min, lat_max, date_start, date_end):
        self.name = name
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.date_start = date_start
        self.date_end = date_end


class MetricDef:
    def __init__(self, name, variable, service_id, product_id, motu_url,
                 output_nc, output_csv, depth_min=0.0, depth_max=0.0,
                 rename_cols=None, drop_cols=None,
                 value_col=None, transform=None):
        self.name = name
        self.variable = variable
        self.service_id = service_id
        self.product_id = product_id
        self.motu_url = motu_url
        self.output_nc = output_nc
        self.output_csv = output_csv
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.rename_cols = rename_cols
        self.drop_cols = drop_cols
        self.value_col = value_col
        self.transform = transform

attributes = [
    MetricDef(
        name="chlorophyll",
        variable="CHL",
        service_id="OCEANCOLOUR_MED_BGC_HR_L3_NRT_009_205-TDS",
        product_id="cmems_obs_oc_med_bgc_tur-spm-chl_nrt_l3-hr-mosaic_P1D-m",
        motu_url="https://nrt.cmems-du.eu/motu-web/Motu",
        output_nc="Chl.nc",
        output_csv="Chl.csv",
        rename_cols={"CHL": "chl", "lat": "latitude", "lon": "longitude"},
        value_col="chl",
    ),
    MetricDef(
        name="turbidity",
        variable="TUR",
        service_id="OCEANCOLOUR_MED_BGC_HR_L3_NRT_009_205-TDS",
        product_id="cmems_obs_oc_med_bgc_tur-spm-chl_nrt_l3-hr-mosaic_P1D-m",
        motu_url="https://nrt.cmems-du.eu/motu-web/Motu",
        output_nc="Turbidity.nc",
        output_csv="Turbidity.csv",
        rename_cols={"lat": "latitude", "lon": "longitude"},
    ),
    MetricDef(
        name="pH",
        variable="ph",
        service_id="GLOBAL_ANALYSIS_FORECAST_BIO_001_028-TDS",
        product_id="global-analysis-forecast-bio-001-028-daily",
        motu_url="https://nrt.cmems-du.eu/motu-web/Motu",
        output_nc="pH_initial.nc",
        output_csv="pH_initial.csv",
        depth_min=0.495,
        depth_max=22.0,
        transform={"mean": "mean", "min": "min", "max": "max"},
    ),
    MetricDef(
        name="temperature",
        variable="thetao",
        service_id="GLOBAL_ANALYSIS_FORECAST_PHY_001_024-TDS",
        product_id="global-analysis-forecast-phy-001-024",
        motu_url="https://nrt.cmems-du.eu/motu-web/Motu",
        output_nc="Temperature.nc",
        output_csv="Temperature.csv",
        depth_min=0.49402499198913574,
        depth_max=0.49402499198913574,
        rename_cols={"thetao": "Temperature"},
        value_col="Temperature",
        drop_cols=["depth"],
    ),
    MetricDef(
        name="oxygen",
        variable="o2",
        service_id="",
        product_id="",
        motu_url="",
        output_nc="Oxygen_initial.nc",
        output_csv="Oxygen_initial.csv",
        value_col="Oxygen",
        transform={"mean": "mean", "min": "min", "max": "max"},
    ),
]


class Config:
    def __init__(self, env_path=None):
        if env_path is None:
            env_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
        self._data = {}
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, _, value = line.partition('=')
                self._data[key.strip()] = value.strip()

    @property
    def username(self):
        return self._data.get('USERNAME', '')

    @property
    def password(self):
        return self._data.get('PASSWORD', '')


# coordinate columns in the filled-weekly output for the combine step
weekly_coords = {
    "chlorophyll": ["lat_center", "lon_center"],
    "turbidity": ["lat_center", "lon_center"],
    "oxygen": ["latitude", "longitude"],
    "pH": ["latitude", "longitude"],
    "temperature": ["latitude", "longitude"],
}

# wqi weights — sum must be 1.0
wqi_weights = {
    "chlorophyll": 0.30,
    "oxygen": 0.25,
    "turbidity": 0.20,
    "temperature": 0.15,
    "pH": 0.10,
}

# wqi ideal / acceptable ranges per metric
wqi_ranges = {
    "pH": {"type": "optimal", "ideal": (8.1, 8.2), "acceptable": (6.5, 9.0)},
    "temperature": {"type": "optimal", "ideal": (12.0, 16.0), "acceptable": (-1.0, 30.0)},
    "turbidity": {"type": "lower_better", "ideal": 0.1, "acceptable": 5.0},
    "oxygen": {"type": "higher_better", "ideal": 70.0, "acceptable": 35.0},
    "chlorophyll": {"type": "optimal", "ideal": (0.3, 10.0), "acceptable": (0.0, 15.0)},
}

# wqi class boundaries (left-inclusive, right-exclusive)
wqi_classes = [
    (-1e9, 2.5, "Very Bad"),
    (2.5, 5.0, "Bad"),
    (5.0, 7.0, "Medium"),
    (7.0, 9.0, "Good"),
    (9.0, 1e9, "Excellent"),
]

# season definitions (month → name)
season_map = {
    1: "spring", 2: "spring", 3: "spring",
    4: "summer", 5: "summer", 6: "summer",
    7: "autumn", 8: "autumn", 9: "autumn",
    10: "winter", 11: "winter", 12: "winter",
}

# columns to use in correlation analysis
correlation_columns = [
    *(f"avg_{name}" for name in wqi_weights),
    "water_quality_index",
]

# metric value and flag columns (keys match wqi_weights)
metric_info = {
    name: {"value": f"avg_{name}", "flag": f"flag_{name}"}
    for name in wqi_weights
}

# insight pipeline column names
insight_time_col = "time"
insight_lat_col = "latitude_oxygen"
insight_lon_col = "longitude_oxygen"
insight_wqi_col = "water_quality_index"
insight_result_col = "water_result"

# machine learning models configuration
models_config = {
    "seasonal_decomp": {
        "enabled": True,
        "period": 52,
        "model": "additive",
    },
    "per_location_trend": {
        "enabled": True,
    },
    "arima": {
        "enabled": False,
        "n_predictions": 12,
        "seasonal": True,
        "period": 52,
    },
    "prophet": {
        "enabled": False,
        "n_predictions": 12,
        "seasonality_mode": "additive",
    },
}

# modeling column names
model_time_col = "time"
model_value_col = "water_quality_index"
model_lat_col = "latitude_oxygen"
model_lon_col = "longitude_oxygen"
model_test_size = 0.2
