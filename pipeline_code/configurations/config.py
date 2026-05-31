# TODO: add all the imports in one module
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
                 rename_cols=None, drop_cols=None):
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

# TODO: check if we can avoid the hardcoded naming and depth
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
