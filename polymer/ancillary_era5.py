#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
ERA5 Ancillary data provider for Polymer
'''

import argparse
from datetime import datetime, timedelta
import os
import numpy as np
import cdsapi
import xarray as xr
from polymer.ancillary import LUT_LatLon


class Ancillary_ERA5(object):
    '''
    Ancillary data provider using ERA5
    https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5

    Arguments:
    * directory: base directory for storing the ERA5 files
    * pattern: pattern for storing the ERA5 files in NetCDF format
    * offline (bool, default False).
        If true, sets offline mode: don't download anything and fail upon missing file.
    '''
    def __init__(self,
                 directory='ANCILLARY/ERA5/',
                 pattern='%Y/%m/%d/era5_%Y%m%d_%H%M%S.nc',
                 time_resolution=1,
                 offline=False,
                 ):
        self.time_resolution = time_resolution
        self.directory = directory
        self.pattern = pattern
        self.ERA5 = ERA5(directory=directory, pattern=pattern, offline=offline)

    def get(self, param, date):
        '''
        Retrieve ancillary parameter at given date

        param:
            * 'wind_speed': surface wind speed in m/s
            * 'surf_press': sea-level pressure in HPa
            * 'ozone': ozone total column in Dobson Units
        '''
        # calculate bracketing times
        day = datetime(date.year, date.month, date.day)
        delta_hours = (date - day).seconds/3600
        hour = self.time_resolution * int(delta_hours/self.time_resolution)

        t1 = day + timedelta(hours=hour)
        t2 = day + timedelta(hours=hour+1)

        file1 = self.ERA5.download_era5(t1)
        file2 = self.ERA5.download_era5(t2)

        ERA1 = xr.open_dataset(file1).isel(time=0)
        ERA2 = xr.open_dataset(file2).isel(time=0)

        x = (date - t1).total_seconds()/(t2 - t1).total_seconds()
        
        if param == 'ozone':
            assert ERA1.tco3.units == 'kg m**-2'
            D1 = ERA1.tco3.values / 2.144e-5   # convert from kg/m2 to DU
            D2 = ERA2.tco3.values / 2.144e-5

        elif param == 'surf_press':
            assert ERA1.sp.units == 'Pa'
            D1 = ERA1.sp.values / 100.   # convert to HPa
            D2 = ERA2.sp.values / 100.

        elif param == 'wind_speed':
            assert ERA1.u10.units == 'm s**-1'
            WU1 = ERA1.u10.values
            WV1 = ERA1.v10.values
            D1 = np.sqrt(WU1*WU1 + WV1*WV1)

            WU2 = ERA2.u10.values
            WV2 = ERA2.v10.values
            D2 = np.sqrt(WU2*WU2 + WV2*WV2)

        else:
            raise Exception('Invalid parameter "{}"'.format(param))

        # roll => lons from -180..180 instead of 0..360
        D1 = np.roll(D1, D1.shape[1]//2, axis=1)
        D2 = np.roll(D2, D2.shape[1]//2, axis=1)

        # Temporal interpolation
        D = LUT_LatLon((1-x)*D1 + x*D2)
        D.date = date
        D.filename = {'ERA5_1': file1,
                      'ERA5_2': file2,
                     }

        return D


class ERA5(object):
    '''
    A class to download ERA5 files for use by Polymer

    Arguments:
    * offline: don't download anything, use existing files
    '''
    def __init__(self,
                 directory='ANCILLARY/ERA5/',
                 pattern='%Y/%m/%d/era5_%Y%m%d_%H%M%S.nc',
                 offline=False,
                 ):
        self.client = cdsapi.Client()
        self.directory = directory
        self.pattern = pattern
        self.offline = offline
        if not os.path.exists(directory):
            raise Exception(f'Directory "{directory}" does not exist.'
                            'Please create it for hosting ERA5 files.')

    def download_era5(self, dt):
        '''
        Download a single ERA5 file for a given datetime `dt`.

        Returns the file name
        '''
        assert dt.minute == 0
        assert dt.second == 0

        target = os.path.join(self.directory, dt.strftime(self.pattern))

        if not os.path.exists(target):

            if self.offline:
                raise Exception(
                    f'ERA5: File {target} is missing and offline'
                    ' mode has been set')

            target_tmp = target + '.tmp'
            directory = os.path.dirname(target)
            print(f'Download {dt} -> {target}')
            if not os.path.exists(directory):
                os.makedirs(directory)

            self.client.retrieve(
                'reanalysis-era5-single-levels',
                {
                    'product_type': 'reanalysis',
                    'variable': ['10m_u_component_of_wind',
                                 '10m_v_component_of_wind',
                                 'surface_pressure',
                                 'total_column_ozone',
                                 'total_column_water_vapour'
                                 ],
                    'year':[
                        f'{dt.year}',
                    ],
                    'month':[
                        f'{dt.month:02}',
                    ],
                    'day':[
                        f'{dt.day:02}',
                    ],
                    'time': f'{dt.hour:02}:00',
                    'format':'netcdf'  # grib, netcdf
                },
                target_tmp)
            os.rename(target_tmp, target)

        return target

    def download_range(self, d0, d1, time_resolution=1):
        t = d0
        while t <= d1:
            self.download_era5(t)
            t += timedelta(hours=time_resolution)


def parse_date(dstring):
    return datetime.strptime(dstring, '%Y-%m-%d')


if __name__ == "__main__":
    # command line mode: download all ERA5 files
    # for a given time range d0 to d1
    parser = argparse.ArgumentParser(
        description='Download all ERA5 files for a given time range')
    parser.add_argument('d0', type=parse_date,
                        help='start date (YYYY-MM-DD)')
    parser.add_argument('d1', type=parse_date,
                        help='stop date (YYYY-MM-DD)')
    parser.add_argument('--time_resolution', type=int,
                        default=1, help='time resolution in hours')
    args = parser.parse_args()

    ERA5().download_range(args.d0, args.d1, args.time_resolution)
