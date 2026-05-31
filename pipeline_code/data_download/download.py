import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import motuclient
import pandas as pd
import xarray as xr

from pipeline_code.configurations.config import Config, Project, attributes


class MotuOptions:
    def __init__(self, attrs):
        super(MotuOptions, self).__setattr__("attrs", attrs)

    def __setattr__(self, k, v):
        self.attrs[k] = v

    def __getattr__(self, k):
        try:
            return self.attrs[k]
        except KeyError:
            return None


def _build_request(config, project, metric, nc_path):
    req = {
        "date_min": f"{project.date_start} 00:00:00",
        "date_max": f"{project.date_end} 23:59:59",
        "longitude_min": project.lon_min,
        "longitude_max": project.lon_max,
        "latitude_min": project.lat_min,
        "latitude_max": project.lat_max,
        "variable": [metric.variable],
        "depth_min": metric.depth_min,
        "depth_max": metric.depth_max,
        "out_dir": os.path.dirname(nc_path),
        "out_name": nc_path,
        "auth_mode": "cas",
        "user": config.username,
        "pwd": config.password,
    }
    return req


def _process_netcdf(metric, output_dir):
    nc_path = os.path.join(output_dir, metric.output_nc)
    ds = xr.open_dataset(nc_path)
    df = ds.to_dataframe().reset_index()
    if metric.drop_cols:
        df = df.drop(columns=metric.drop_cols, errors='ignore')
    if metric.rename_cols:
        df.rename(columns=metric.rename_cols, inplace=True)
    csv_path = os.path.join(output_dir, metric.output_csv)
    df.to_csv(csv_path)
    return df


def _download_and_process(config, project, metric, output_dir):
    nc_path = os.path.join(output_dir, metric.output_nc)
    req = _build_request(config, project, metric, nc_path)
    req["service_id"] = metric.service_id
    req["product_id"] = metric.product_id
    req["motu"] = metric.motu_url
    motuclient.motu_api.execute_request(MotuOptions(req))
    return _process_netcdf(metric, output_dir)


def _download_oxygen(config, project, output_dir):
    nc1 = os.path.join(output_dir, "Oxygen_initial1.nc")
    nc2 = os.path.join(output_dir, "Oxygen_initial2.nc")

    req1 = {
        "service_id": "MEDSEA_MULTIYEAR_BGC_006_008-TDS",
        "product_id": "med-ogs-bio-rean-d",
        "date_min": f"{project.date_start} 00:00:00",
        "date_max": "2020-05-31 23:59:59",
        "longitude_min": project.lon_min,
        "longitude_max": project.lon_max,
        "latitude_min": project.lat_min,
        "latitude_max": project.lat_max,
        "depth_min": 1.01824,
        "depth_max": 22.1,
        "variable": ["o2"],
        "motu": "https://my.cmems-du.eu/motu-web/Motu",
        "out_dir": output_dir,
        "out_name": nc1,
        "auth_mode": "cas",
        "user": config.username,
        "pwd": config.password,
    }
    motuclient.motu_api.execute_request(MotuOptions(req1))
    ds1 = xr.open_dataset(nc1)
    df1 = ds1.to_dataframe().reset_index()
    df1.rename(columns={'o2': 'Oxygen'}, inplace=True)
    df1.to_csv(os.path.join(output_dir, "Oxygen_initial1.csv"))

    req2 = {
        "service_id": "MEDSEA_ANALYSISFORECAST_BGC_006_014-TDS",
        "product_id": "med-ogs-bio-an-fc-d",
        "date_min": "2020-06-01 00:00:00",
        "date_max": f"{project.date_end} 23:59:59",
        "longitude_min": project.lon_min,
        "longitude_max": project.lon_max,
        "latitude_min": project.lat_min,
        "latitude_max": project.lat_max,
        "depth_min": 1.01824,
        "depth_max": 22.1,
        "variable": ["o2"],
        "motu": "https://nrt.cmems-du.eu/motu-web/Motu",
        "out_dir": output_dir,
        "out_name": nc2,
        "auth_mode": "cas",
        "user": config.username,
        "pwd": config.password,
    }
    motuclient.motu_api.execute_request(MotuOptions(req2))
    ds2 = xr.open_dataset(nc2)
    df2 = ds2.to_dataframe().reset_index()
    df2.rename(columns={'o2': 'Oxygen'}, inplace=True)
    df2.to_csv(os.path.join(output_dir, "Oxygen_initial2.csv"))

    df = pd.concat([df1, df2], ignore_index=True)
    df.to_csv(os.path.join(output_dir, "Oxygen_initial.csv"))
    return df


def download_metric(config, project, metric, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    if metric.name == "oxygen":
        return _download_oxygen(config, project, output_dir)

    return _download_and_process(config, project, metric, output_dir)


def download_all(config, project, output_dir="data", max_workers=5):
    output_dir = os.path.join(output_dir, project.name)
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_metric, config, project, m, output_dir): m.name
            for m in attributes
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
                print(f"  download {name}")
            except Exception as e:
                results[name] = None
                print(f"  download {name} failed: {e}")

    return results
