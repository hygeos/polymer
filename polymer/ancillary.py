#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import
from os.path import isdir
import requests
import sys
from datetime import datetime, timedelta
from pyhdf.SD import SD
import numpy as np
from os import makedirs, system, remove
from os.path import join, exists, dirname, basename
from polymer.luts import LUT, Idx
from warnings import warn


class LUT_LatLon(object):
    '''
    wrapper around a 2D LUT Lat/Lon
    for reprojection on a custom (lat, lon) grid

    Exemple:
    Ancillary(wind_speed)[lat, lon]
    reprojects wind_speed over grid (lat, lon)
    '''
    def __init__(self, A):

        h, w = A.shape
        assert w/h > 1
        data = np.append(A, A[:, 0, None], axis=1)

        self.data = LUT(data, names=['latitude', 'longitude'],
                axes=[np.linspace(90, -90, h),
                      np.linspace(-180, 180, w+1)]
                )

    def __getitem__(self, coords):
        '''
        Bi-directional linear interpolation over latitude and longitude
        '''

        lat, lon = coords

        return self.data[Idx(lat), Idx(lon)]

class LockFile(object):
    '''
    Create a context with a lock file
    '''
    def __init__(self, filename):
        self.filename = filename

    def __enter__(self):
        system('touch {}'.format(self.filename))

    def __exit__(self, type, value, traceback):
        remove(self.filename)


class Ancillary_NASA(object):
    '''
    Ancillary data provider using NASA data

    Arguments:
    * meteo: NCEP filename              (without interpolation)
             or tuple (meteo1, meteo1)  (with interpolation)
             if None, search for the two closest and activate interpolation
    * ozone: ozone filename             (without interpolation)
             or tuple (ozone1, ozone2)  (with interpolation)
             if None, search for the two closest (first offline, then online)
             and activate interpolation
    * directory: local directory for ancillary data storage
    * offline: boolean. If offline, does not try to download
    '''
    def __init__(self, meteo=None, ozone=None,
                 directory='ANCILLARY/METEO/', offline=False):
        self.meteo = meteo
        self.ozone = ozone
        self.directory = directory
        self.offline = offline
        self.url = 'https://oceandata.sci.gsfc.nasa.gov/cgi/getfile/'

        assert isdir(directory), '{} does not exist'.format(directory)


    def read(self, param, filename):
        '''
        Read ancillary data from filename

        returns LUT_LatLon and date
        '''
        hdf = SD(filename)

        assert isinstance(filename, str)
        if param == 'wind_speed':
            # read m_wind and z_wind
            zwind = hdf.select('z_wind').get()
            mwind = hdf.select('m_wind').get()
            wind = np.sqrt(zwind*zwind + mwind*mwind)
            D = LUT_LatLon(wind)

        elif param == 'surf_press':
            press = hdf.select('press').get()
            D = LUT_LatLon(press)

        elif param == 'ozone':
            sds = hdf.select('ozone')
            assert 'Dobson units' in sds.attributes()['units']
            ozone = sds.get()
            D = LUT_LatLon(ozone)

        else:
            raise Exception('Invalid parameter "{}"'.format(param))

        D.date = datetime.strptime(hdf.attributes()['Start Time'][:13],
                                   '%Y%j%H%M%S')

        return D


    def get(self, param, date):
        '''
        Retrieve ancillary parameter at given date

        param:
            'wind_speed': surface wind speed in m/s
            'surf_press': sea-level pressure in HPa
            'ozone': ozone total column in Dobson Units
        '''
        if param in ['wind_speed', 'surf_press']:
            if self.meteo is None:
                self.meteo = self.find_meteo(date)
            res = self.meteo

        if param in ['ozone']:
            if self.ozone is None:
                self.ozone = self.find_ozone(date)
            res = self.ozone

        if isinstance(res, tuple):
            # interpolation
            D1 = self.read(param, res[0])
            D2 = self.read(param, res[1])

            x = (date - D1.date).total_seconds()/(D2.date - D1.date).total_seconds()

            if D1.data.shape == D2.data.shape:
                D = LUT_LatLon((1-x)*D1.data[:,:] + x*D2.data[:,:])
                D.date = date
                return D
            else:
                warn('Incompatible auxiliary data "{}" {} and "{}" {}'.format(
                    basename(res[0]), D1.data.shape, basename(res[1]), D2.data.shape))
                if x < 0.5:
                    warn('Using "{}"'.format(basename(res[0])))
                    return D1
                else:
                    warn('Using "{}"'.format(basename(res[1])))
                    return D2
        else:
            # res is a single file name (string)
            # disactivate interpolation
            return self.read(param, res)


    def download(self, url, target):

        # check that remote file exists
        r = requests.head(url)
        if r.status_code != requests.codes.ok:
            return 1

        # download that file
        if not exists(dirname(target)):
            makedirs(dirname(target))

        lock = target + '.lock'
        if exists(lock):
            raise Exception('lock file "{}" is present'.format(lock))

        assert basename(url) == basename(target)

        with LockFile(lock), open(target, 'wb') as t:

            content = requests.get(url).content
            t.write(content)

        return 0


    def try_resources(self, patterns, date):

        # first, try local files
        for pattern in patterns:
            target = date.strftime(join(self.directory, '%Y/%j/'+pattern))
            if exists(target):
                return target

        # then try to download if requested
        if not self.offline:  # If offline flag set, don't download
            for pattern in patterns:
                url = date.strftime(self.url+pattern)
                target = date.strftime(join(self.directory, '%Y/%j/'+pattern))

                print('Trying to download', url, '... ', end='')
                sys.stdout.flush()
                if self.download(url, target) == 0:
                    print('success!')
                    return target
                else:
                    print('failure')
        elif self.offline:
            print('Offline ancillary data requested but not available in {}'.format(self.directory))
        return None

    def find_meteo(self, date):

        # bracketing dates
        day = datetime(date.year, date.month, date.day)
        d0 = day + timedelta(hours=6*int(date.hour/6.))
        d1 = day + timedelta(hours=6*(int(date.hour/6.)+1))
        patterns = ['N%Y%j%H_MET_NCEP_6h.hdf',
                    'S%Y%j%H_NCEP.MET']
        f1 = self.try_resources(patterns, d0)
        f2 = self.try_resources(patterns, d1)

        if None in [f1, f2]:
            raise Exception('Could not find meteo files for {}'.format(date))

        return (f1, f2)

    def find_ozone(self, date):

        # bracketing dates
        d0 = datetime(date.year, date.month, date.day)
        d1 = d0 + timedelta(days=1)
        patterns = ['N%Y%j00_O3_AURAOMI_24h.hdf',
                    'S%Y%j00%j23_TOAST.OZONE',
                    'N%Y%j00_O3_EPTOMS_24h.hdf']
        f1 = self.try_resources(patterns, d0)
        f2 = self.try_resources(patterns, d1)
        if None in [f1, f2]:
            raise Exception('Could not find any valid ozone file for {}'.format(date))

        return (f1, f2)


