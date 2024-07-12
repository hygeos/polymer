#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from tempfile import TemporaryDirectory
from polymer.copernicus_dem import CopernicusDEM


def test_download():
    with TemporaryDirectory() as tmpdir:
        cop_dem = CopernicusDEM(directory=tmpdir, verbose=True)
        local_path = cop_dem._download_tile(-1, 34)
        assert local_path.endswith("Copernicus_DSM_COG_30_S01_00_E034_00_DEM.tif")
        local_path = cop_dem._download_tile(-1, 34)

def test_get():
    lat = np.tile(np.arange(0, -1.05, -0.1), (11, 1)).T
    lon = np.tile(np.arange(34.0, 35.05, 0.1), (11, 1))
    with TemporaryDirectory() as tmpdir:
        cop_dem = CopernicusDEM(directory=tmpdir)
        alt = cop_dem.get(lat, lon)
    assert np.abs(lat[5, 5] - (-0.5)) < 0.0001
    assert np.abs(lon[5, 5] - 34.5) < 0.0001
    assert np.abs(alt[5, 5] - 1175.619384765625) < 0.0001

def test_get_55():
    lat = np.tile(np.arange(56, 54.95, -0.1), (11, 1)).T
    lon = np.tile(np.arange(11.0, 12.05, 0.1), (11, 1))
    with TemporaryDirectory() as tmpdir:
        cop_dem = CopernicusDEM(directory=tmpdir)
        alt = cop_dem.get(lat, lon)
    assert np.abs(lat[5, 5] - (55.5)) < 0.0001
    assert np.abs(lon[5, 5] - 11.5) < 0.0001
    assert np.abs(alt[5, 5] - 30.285972595214844) < 0.0001

