#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import
from os.path import isdir
import requests
from datetime import datetime, timedelta
from pyhdf.SD import SD
import numpy as np
from os import makedirs, system, remove
from os.path import join, exists, dirname, basename
from polymer.luts import LUT, Idx


class LUT_LatLon(object):
    '''
    wrapper around a 2D LUT Lat/Lon
    for reprojection on a custom (lat, lon) grid

    Exemple:
    Ancillary(wind_speed)[lat, lon]
    reprojects wind_wind speed over grid (lat, lon)
    '''
    def __init__(self, A):

        h, w = A.shape
        assert w/h > 1
        data = np.append(A, A[:, 0, None], axis=1)

        self.A = LUT(data, names=['latitude', 'longitude'],
                axes=[np.linspace(90, -90, h),
                      np.linspace(-180, 180, w+1)]
                )

    def __getitem__(self, coords):
        '''
        Bi-directional linear interpolation over latitude and longitude
        '''

        lat, lon = coords

        return self.A[Idx(lat), Idx(lon)]

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
    meteo: NCEP filename
    ozone: ozone filename
    if any filename is None, finds the closest data online or in
    directory
    '''
    def __init__(self, meteo=None, ozone=None,
                 directory='ANCILLARY/METEO/', offline=False):
        self.meteo = meteo
        self.ozone = ozone
        self.directory = directory
        self.offline = offline
        self.url = 'http://oceandata.sci.gsfc.nasa.gov/cgi/getfile/'

        assert isdir(directory), '{} does not exist'.format(directory)

    def get(self, param, date):
        # TODO
        # interpolate between 2 bracketing datasets

        if param == 'wind_speed':
            if self.meteo is None:
                self.meteo = self.find_meteo(date)
            # read m_wind and z_wind
            zwind = SD(self.meteo).select('z_wind').get()
            mwind = SD(self.meteo).select('m_wind').get()
            wind = np.sqrt(zwind*zwind + mwind*mwind)
            return LUT_LatLon(wind)

        elif param == 'surf_press':
            if self.meteo is None:
                self.meteo = self.find_meteo(date)
            press = SD(self.meteo).select('press').get()
            return LUT_LatLon(press)

        elif param == 'ozone':
            if self.ozone is None:
                self.ozone = self.find_ozone(date)

            sds = SD(self.ozone).select('ozone')
            assert 'Dobson units' in sds.attributes()['units']
            ozone = sds.get()
            return LUT_LatLon(ozone)

        else:
            raise Exception('Invalid parameter "{}"'.format(param))


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

                print('Trying to download', url)
                if self.download(url, target) == 0:
                    print('...success!')
                    return target
                else:
                    print('...failure')
        elif self.offline:
            print('Offline ancillary data requested but not available in {}'.format(self.directory))
        return None

    def find_meteo(self, date):

        d0 = datetime(date.year, date.month, date.day,
                       6*int(date.hour/6.))
        filename = self.try_resources([
                'N%Y%j%H_MET_NCEP_6h.hdf',
                'S%Y%j%H_NCEP.MET',
                ], d0)

        if filename is None:
            raise Exception('Could not find any valid meteo file for {}'.format(d0))

        return filename

    def find_ozone(self, date):

        d0 = datetime(date.year, date.month, date.day)
        for i in range(10):
            filename = self.try_resources([
                'N%Y%j00_O3_AURAOMI_24h.hdf',
                'S%Y%j00%j23_TOAST.OZONE',
                'N%Y%j00_O3_EPTOMS_24h.hdf',
                ], d0 - timedelta(days=i))
            if filename is not None:
                break
        if filename is None:
            raise Exception('Could not find any valid ozone file')

        return filename


