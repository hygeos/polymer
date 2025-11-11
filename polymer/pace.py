from pathlib import Path
from typing import Dict

import xarray as xr
from core.download import download_url
from core.env import getdir
from core.tools import raiseflag
from dateutil.parser import parse

from polymer.common import L2FLAGS


def Level1B_PACE_OCI(product_pace_oci: Path) -> xr.Dataset:
    """
    Read PACE OCI Level 1B products
    """
    tree = xr.open_datatree(product_pace_oci, chunks={"scans": 300, "pixels": 200})

    geo = tree["geolocation_data"].to_dataset().reset_coords(["latitude", "longitude"])
    bdata = tree["sensor_band_parameters"].to_dataset()
    obs = tree["observation_data"].to_dataset()

    ds = xr.Dataset()
    ds["latitude"] = geo["latitude"]
    ds["longitude"] = geo["longitude"]

    ds["vaa"] = geo["sensor_azimuth"].astype("float32")
    ds["vza"] = geo["sensor_zenith"].astype("float32")
    ds["saa"] = geo["solar_azimuth"].astype("float32")
    ds["sza"] = geo["solar_zenith"].astype("float32")

    # TOA reflectance
    ds["Rtoa"] = xr.concat(
        [
            obs["rhot_blue"].rename(blue_bands="bands"),
            obs["rhot_red"].rename(red_bands="bands"),
            obs["rhot_SWIR"].rename(SWIR_bands="bands"),
        ],
        dim="bands",
    )

    ds["wav"] = xr.concat(
        [
            bdata["blue_wavelength"].rename(blue_bands="bands"),
            bdata["red_wavelength"].rename(red_bands="bands"),
            bdata["SWIR_wavelength"].rename(SWIR_bands="bands"),
        ],
        dim="bands",
    )

    # Flags
    ds["flags"] = xr.zeros_like(ds.sza, dtype="uint16")
    raiseflag(
        ds["flags"],
        "LAND",
        L2FLAGS["LAND"],
        geo["watermask"] == 0,
    )

    # Attributes
    ds.attrs.update(sensor="OCI")
    ds.attrs.update(product_name=product_pace_oci.name)
    time_start = parse(tree.attrs["time_coverage_start"])
    time_end = parse(tree.attrs["time_coverage_end"])
    ds.attrs.update(datetime=(time_start + (time_end - time_start) / 2).isoformat())

    ds = ds.assign_coords(bands=ds.wav.astype(int)).chunk(bands=-1)

    # x/y dimensions
    ds = ds.rename(scans="y", pixels="x")

    return ds


def get_sample(sample: int = 1) -> Dict:
    """
    Return sample PACE Level-1B products

    Returns: a dict with keys:
        path: path to the product
        roi: region of interest within the full product
        px: sample pixel coordinates within the roi
    """
    if sample == 1:
        # Freely accessible on NASA website
        return {
            "path": download_url(
                url="https://oceancolor.gsfc.nasa.gov/images/data/pace/sample_data/PACE_OCI.20250101T000738.L1B.V3.nc",
                dirname=getdir("DIR_SAMPLES") / "PACE_OCI",
            ),
            "roi": {"y": slice(1680, None), "x": slice(700, 900)},
            "px": {"y": 15, "x": 150},
        }
    elif sample == 2:
        pace_level1 = getdir("DIR_PACE_LEVEL1B") / "PACE_OCI.20250430T130447.L1B.V3.nc"
        assert pace_level1.exists()
        return {
            "path": pace_level1,
            "roi": {"y": slice(750, 900), "x": slice(950, 1050)},
            "px": {"y": 100, "x": 20},
        }
    else:
        raise ValueError
