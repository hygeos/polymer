#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function, division, absolute_import
import numpy as np
from datetime import datetime, timedelta
from polymer.ancillary import LUT_LatLon, LockFile
from os.path import join, exists, dirname
from os import makedirs


def download_era(date, filename):
    '''
    Download ERA-INTERIM data

    requires an access to ECMWF server in GRIB format
    see "Installing your API key":
    https://software.ecmwf.int/wiki/display/WEBAPI/Accessing+ECMWF+data+servers+in+batch

    This also requires the client libraries.
    '''
    from ecmwfapi import ECMWFDataServer

    server = ECMWFDataServer()

    server.retrieve({
        'stream'  : 'oper',
        'levtype' : 'sfc',
        'param'   :  '151.128'   # mean sea level pressure
                    '/165.128'   # 10 metre U wind component
                    '/166.128'   # 10 metre V wind component
                    '/206.128',  # total column ozone
        'dataset' : 'interim',
        'step'    : '0',
        'grid'    : '0.75/0.75',
        'time'    : '00/06/12/18',
        'date'    : date.strftime('%Y-%m-%d/to/%Y-%m-%d'),
        'type'    : 'an',
        'class'   : 'ei',
        'target'  : filename,
        })

class Ancillary_ERA(object):
    '''
    Ancillary data provider using ERA-Interim
    http://www.ecmwf.int/en/research/climate-reanalysis/era-interim

    Arguments:
    * directory: base directory for storing the ERA-Interim files
    * pattern: pattern for storing the GRIB files (%Y, %m, %d represent the
               year, month and day)
    * offline: offline mode, does not try to access online data
    '''

    def __init__(self,
                 directory='ANCILLARY/ERA-Interim/',
                 pattern='%Y/era_interim_%Y%m%d.grib',
                 offline=False):
        self.directory = directory
        if not exists(directory):
            raise Exception('Directory {} does not exist. Please create it, by default it will be '
                            'automatically populated with ancillary data. '
                            'Please see help for class Ancillary_ERA for more details.'.format(directory))
        self.pattern = pattern
        self.offline = offline


    def download(self, date, target):

        if exists(target):
            return

        # create directory if needed
        if not exists(dirname(target)):
            makedirs(dirname(target))

        # lock file
        lock = target + '.lock'
        if exists(lock):
            raise Exception('lock file "{}" is present'.format(lock))

        with LockFile(lock):
            download_era(date, target)


    def read(self, param, unit, filename, hour):

        import pygrib

        grbs = pygrib.open(filename)
        grbs.seek(0)

        D = None
        for x in grbs.select(name=param):
            if x.hour == hour:
                D = x.values
                assert x.units == unit, 'unit problem: {}, {}'.format(x.units, unit)
                assert x.latlons()[1][0,0] == 0.
        assert D is not None

        return D


    def get(self, param, date):
        '''
        Retrieve ancillary parameter at given date

        param:
            * 'wind_speed': surface wind speed in m/s
            * 'surf_press': sea-level pressure in HPa
            * 'ozone': ozone total column in Dobson Units
        '''
        # determine the bracketing files and hours

        day = datetime(date.year, date.month, date.day)
        t1 = day + timedelta(hours=6*(date.hour//6))
        t2 = day + timedelta(hours=6*(date.hour//6 + 1))

        file1 = join(self.directory, t1.strftime(self.pattern))
        file2 = join(self.directory, t2.strftime(self.pattern))

        if not self.offline:
            self.download(t1, file1)
            self.download(t2, file2)

        x = (date - t1).total_seconds()/(t2 - t1).total_seconds()

        if param == 'ozone':
            D1 = self.read('Total column ozone', 'kg m**-2', file1, t1.hour)
            D2 = self.read('Total column ozone', 'kg m**-2', file2, t2.hour)

            # convert from kg/m2 to DU
            D1 /= 2.144e-5
            D2 /= 2.144e-5

        elif param == 'surf_press':
            D1 = self.read('Mean sea level pressure', 'Pa', file1, t1.hour)
            D2 = self.read('Mean sea level pressure', 'Pa', file2, t2.hour)

            # convert to HPa
            D1 /= 100.
            D2 /= 100.

        elif param == 'wind_speed':
            WU1 = self.read('10 metre U wind component', 'm s**-1', file1, t1.hour)
            WV1 = self.read('10 metre V wind component', 'm s**-1', file1, t1.hour)
            D1 = np.sqrt(WU1*WU1 + WV1*WV1)

            WU2 = self.read('10 metre U wind component', 'm s**-1', file2, t2.hour)
            WV2 = self.read('10 metre V wind component', 'm s**-1', file2, t2.hour)
            D2 = np.sqrt(WU2*WU2 + WV2*WV2)
        else:
            raise Exception('Invalid parameter "{}"'.format(param))

        # roll => lons from -180..180 instead of 0..360
        D1 = np.roll(D1, D1.shape[1]//2, axis=1)
        D2 = np.roll(D2, D2.shape[1]//2, axis=1)

        # Temporal interpolation
        D = LUT_LatLon((1-x)*D1 + x*D2)
        D.date = date
        D.filename = {'ERA_INTERIM_1': file1,
                      'ERA_INTERIM_2': file2,
                     }

        return D

