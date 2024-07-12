#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import
import ssl
import certifi
from urllib.request import urlopen
import numpy as np
import os
import os.path
from glob import glob
import sys
import xarray as xr

copernicus_30m_dem_url_prefix = "https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com"
copernicus_90m_dem_url_prefix = "https://copernicus-dem-90m.s3.eu-central-1.amazonaws.com"

class CopernicusDEM(object):
    """
    Copernicus digital elevation model, 90m or 30m, with functions
    to get the altitude above the geoid for a lat-lon grid and
    to download on demand DEM tiles via internet.

    CopernicusDEM().get(lat, lon)
    where lat and lon are array-like returns an array-like alt in m

    The DEM is a grid of tie points, not of pixels. The altitude is given for
    the degree line. There are 1200 rows per degree. The bottom row is missing
    as it is part of the next tile. The number of colums depends on the
    latitude. There are 1200 columns per tile at the equator. From 50 degrees
    onwards it is 800 columns, and so on.

      Longitude spacing  Latitude spacing  width
      0 - 50 	         1x 	           1200
      50 - 60 	         1.5x 	           800
      60 - 70 	         2x 	           600
      70 - 80 	         3x 	           400
      80 - 85 	         5x 	           240
      85 - 90 	         10x 	           120

    DEM rows are ordered from North to South. The first line is the North-most line.
    DEM columns are ordered from West to East.

    Naming convention for the 1 degree tiles in COP format, example:

      Copernicus_DSM_COG_30_N54_00_E009_00_DEM.tif

      Upper Left  (   8.9993750,  55.0004167) (  8d59'57.75"E, 55d 0' 1.50"N)
      Lower Left  (   8.9993750,  54.0004167) (  8d59'57.75"E, 54d 0' 1.50"N)
      Upper Right (   9.9993750,  55.0004167) (  9d59'57.75"E, 55d 0' 1.50"N)
      Lower Right (   9.9993750,  54.0004167) (  9d59'57.75"E, 54d 0' 1.50"N)

      The name denotes the lower left corner. S and W are like minus signs.
    """
    def __init__(self,
                 directory='auxdata-Copernicus-90m-Global-DEM',
                 resolution=90,
                 missing=0.0,
                 verbose=False,
                 with_download=True):
        """
        Memorises DEM parameters

        :param directory: where to find and store downloaded tiles, must exist
        :param resolution: 90 or 30, default 90
        :param missing: what to provide in case of missing value
          * a float, e.g. 0.0 (default)
          * None : raise an error
        :param verbose: with trace output
        :param with_download: whether to allow download of missing tiles, default True
        """
        self.cache_directory = directory
        self.resolution = resolution
        self.verbose = verbose
        self.missing = missing
        if resolution == 90:
            self.url_prefix = copernicus_90m_dem_url_prefix
        elif resolution == 30:
            self.url_prefix = copernicus_30m_dem_url_prefix
        else:
            raise ValueError("resolution " + str(resolution) + " does not match Copernicus DEM. 90 or 30 expected.")
        if not os.path.exists(directory):
            raise IOError('Directory "{}" does not exist'.format(directory))
        self.with_download = with_download
        if self.with_download:
            self.available = self._list_available_tiles()
            if self.verbose:
                print(str(len(self.available)-1) + " remote DEM tiles existing")
        if self.verbose:
            local = glob(os.path.join(self.cache_directory, "*.tif"))
            print(str(len(local)) + " local DEM tiles available")

    def get(self, lat, lon):
        """
        Reads Copernicus DEM data for an array (lat, lon), downloads tiles on demand
        """
        # initialise altitude
        altitude = np.empty(lat.shape, np.float32)
        altitude[:] = np.nan
        # round lat and lon, encode as bin index per pixel of 360 cols and 180 rows starting at -90/-180
        # (A pixel center less than half a pixel above the degree line is covered by this tile)
        # (A pixel center less than half a pixel left of the degree line is covered by this tile)
        tile_height = 1200 * 90 / self.resolution
        half_pixel_height = 0.5 / tile_height
        rows = np.floor(np.array(lat) - half_pixel_height).astype(np.int32)
        tile_width = np.empty(lat.shape, dtype=np.int16)
        tile_width[:] = tile_height
        tile_width[(rows>50)|(rows<-50)] = tile_height * 2 // 3
        tile_width[(rows>60)|(rows<-60)] = tile_height // 2
        tile_width[(rows>70)|(rows<-70)] = tile_height // 3
        tile_width[(rows>80)|(rows<-80)] = tile_height // 5
        tile_width[(rows>85)|(rows<-85)] = tile_height // 10
        half_pixel_width = 0.5 / tile_width
        cols = np.floor(np.array(lon) + half_pixel_width).astype(np.int32)
        bin_index = (rows + 90) * 360 + cols + 180
        del rows, cols
        # determine set of different bin cells
        bin_set = np.unique(bin_index)
        # loop over 1-deg bin cells required to cover grid
        for bin in bin_set:
            # determine lat and lon of lower left corner of bin cell
            row = bin // 360 - 90
            col = bin % 360 - 180
            # ensure DEM tile is available locally
            local_path = self._download_tile(row, col)
            if local_path is None:
                continue
            # read file content
            if self.verbose:
                print("reading DEM tile " + local_path)
            with xr.open_dataset(local_path, engine="rasterio") as ds:
                dem = ds.band_data.values[0]
            # transfer content subset into target altitude
            is_inside_tile = (bin_index == bin)
            # (A pixel center less than half a pixel above the degree line is covered by this tile)
            # (A pixel center less than half a pixel left of the degree line is covered by this tile)
            dem_row = (((row + 1) - np.array(lat)[is_inside_tile] + half_pixel_height) * tile_height).astype(np.int32)
            dem_col = ((np.array(lon)[is_inside_tile] - col + half_pixel_width[is_inside_tile]) * tile_width[is_inside_tile]).astype(np.int32)
            altitude[is_inside_tile] = dem[(dem_row, dem_col)]
            del dem_row, dem_col, is_inside_tile, dem
        del bin_index, half_pixel_width, tile_width
        # fill in missing values
        if self.missing is not None: # assuming float
            altitude[np.isnan(altitude)] = self.missing
        assert not np.isnan(altitude).any()
        #self._write_dem_tile_to_file(alt, lat, lon)
        return altitude

    def fetch_all(self):
        lat, lon = np.meshgrid(np.linspace(-90, 90, 180), np.linspace(-180, 180, 360))
        self.get(lat, lon)


    def _list_available_tiles(self):
        list_filename = "tileList.txt"
        local_list_path = os.path.join(self.cache_directory, list_filename)
        if not os.path.exists(local_list_path):
            url = "{}/{}".format(self.url_prefix, list_filename)
            if self.verbose:
                print('downloading... ', end='')
                sys.stdout.flush()
            with urlopen(url, context=ssl.create_default_context(cafile=certifi.where())) as response:
                content = response.read()
            tmp_path = local_list_path + '.tmp'
            with open(tmp_path, 'wb') as fp:
                fp.write(content)
            os.rename(tmp_path, local_list_path)
            if self.verbose:
                print(list_filename)
        with open(local_list_path) as fp:
            return fp.read().split("\n")

    def _download_tile(self, row, col):
        """
        Checks availability and downloads one DEM tile and stores it
        in directory specified as constructor parameter.

        :param row: degrees north, -90..89
        :param col: degrees east, -180..179
        :return local path to DEM file downloaded or existing, None if there is no such tile
        """
        # Copernicus_DSM_COG_30_N00_00_E006_00_DEM.tif
        dem_filename = "Copernicus_DSM_COG_{}_{}{:02d}_00_{}{:03d}_00_DEM.tif".format(
            self.resolution // 3,  # arc seconds
            "N" if row >= 0 else "S",
            np.abs(row),
            "E" if col >= 0 else "W",
            np.abs(col)
        )
        local_path = os.path.join(self.cache_directory, dem_filename)
        # shortcuts
        if os.path.exists(local_path):
            return local_path
        if not self.with_download or dem_filename[:-len(".tif")] not in self.available:
            return None
        # download
        # https://.../Copernicus_DSM_COG_30_N00_00_E006_00_DEM/Copernicus_DSM_COG_30_N00_00_E006_00_DEM.tif
        url = "{}/{}/{}".format(self.url_prefix, dem_filename[:-len(".tif")], dem_filename)
        if self.verbose:
            print('downloading... ', end='')
            sys.stdout.flush()
        with urlopen(url, context=ssl.create_default_context(cafile=certifi.where())) as response:
            content = response.read()
        tmp_path = local_path + '.tmp'
        with open(tmp_path, 'wb') as fp:
            fp.write(content)
        os.rename(tmp_path, local_path)
        if self.verbose:
            print(dem_filename)
        return local_path

    def _write_dem_tile_to_file(self, alt, lat, lon):
        # test output
        alt_da = xr.DataArray(alt, dims=['y', 'x'], attrs={"long_name": "altitude above geoid in meters"})
        lat_da = xr.DataArray(lat, dims=['y', 'x'], attrs={"standard_name": "latitude"})
        lon_da = xr.DataArray(lon, dims=['y', 'x'], attrs={"standard_name": "longitude"})
        ds = xr.Dataset({"alt": alt_da},
                        coords={"lat": lat_da, "lon": lon_da},
                        attrs={"description": "tile with nearest-interpolated DEM data"})
        filename = "interpolated_dem_" + str(lat[0,0]) + "_" + str(lon[0,0]) + ".nc"
        ds.to_netcdf(filename)

if __name__ == "__main__":
    # download all to the directory provided in argument
    directory = sys.argv[1]
    CopernicusDEM(directory=directory, verbose=True).fetch_all()
