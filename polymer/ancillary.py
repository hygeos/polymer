#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import
from os.path import isdir
from datetime import datetime, timedelta
from pyhdf.SD import SD
import numpy as np
from os import makedirs, system, remove
from os.path import join, exists, dirname, basename
from polymer.luts import LUT, Idx
from warnings import warn
import sys
import bz2
import tempfile


default_met_patterns = ['N%Y%j%H_MET_NCEPR2_6h.hdf.bz2', # reanalysis 2 (best)
                        'N%Y%j%H_MET_NCEP_6h.hdf.bz2', # NRT
                        'N%Y%j%H_MET_NCEP_6h.hdf', # NRT
                        'N%Y%j%H_MET_NCEP_1440x720_f12.hdf', # 12hr forecast
                         ]

default_oz_patterns = ['N%Y%j00_O3_AURAOMI_24h.hdf',
                       'N%Y%j00_O3_TOMSOMI_24h.hdf',
                       'S%Y%j00%j23_TOAST.OZONE',
                       'S%Y%j00%j23_TOVS.OZONE',
                       ]


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


def rolling(t0, deltamax,  delta):
    '''
    returns a list departing from t0 by delta increments

    [t0, t0+delta, t0-delta, t0+2*delta, ...]
    '''
    L = []
    i = 1
    curr = 0*delta
    while abs(curr) <= deltamax:
        L.append(t0+curr)
        sign = ((i%2)*2) - 1
        curr += sign*i*delta
        i+= 1
    return L


def perdelta(start, end, delta):
    '''
    An equivalent of range, working for dates
    '''
    curr = start
    if delta > timedelta(hours=0):
        until = lambda x, y: x <= y
    else:
        until = lambda x, y: x >= y

    L = []
    while until(curr, end):
        L.append(curr)
        curr += delta

    return L


def verify(filename):
    '''
    Fix files with wrong extension from NASA
    -> HDF files with bz2 extension
    '''
    if filename.endswith('.bz2') and system('bzip2 -t '+filename):
        target = filename[:-4]
        system('mv -v {} {}'.format(filename, target))
        filename = target

    return filename


class Ancillary_NASA(object):
    '''
    Ancillary data provider using NASA data.
    See https://oceancolor.gsfc.nasa.gov/cms/ancillary for details

    Arguments:

    * meteo: NCEP filename              (without interpolation)
             or tuple (meteo1, meteo1)  (with interpolation)
             if None, search for the two closest and activate interpolation
    * ozone: ozone filename
             if None, search for the closest file (first offline, then online)
             (don't activate interpolation)
    * directory: local directory for ancillary data storage
    * offline (bool):  If offline, does not try to download
    * delta (float): number of acceptable days before and after scene date, for
                     ancillary data searching.
    * met_patterns (list): patterns for meteorological data.  Will be checked
                           in order. strftime compatible placeholders will be
                           substituted.
                           if None (default), use default patterns
    * ozone_patterns (list): patterns for meteorological data.  Will be checked
                             in order. strftime compatible placeholders will be
                             substituted
                           if None (default), use default patterns
    '''
    def __init__(self, meteo=None, ozone=None,
                 directory='ANCILLARY/METEO/', offline=False, delta=0.,
                 met_patterns=None, ozone_patterns=None):
        self.meteo = meteo
        self.met_patterns = default_met_patterns if (met_patterns is None) else met_patterns
        self.ozone = ozone
        self.ozone_patterns = default_oz_patterns if (ozone_patterns is None) else ozone_patterns
        self.directory = directory
        self.offline = offline
        self.delta = timedelta(days=delta)
        self.url = 'https://oceandata.sci.gsfc.nasa.gov/cgi/getfile/'

        assert isdir(directory), 'Directory {} does not exist. Please create it, by default it will be automatically populated with ancillary data. Please see help for class Ancillary_NASA for more details.'.format(directory)


    def read(self, param, filename,
             uncompress=None, orig_filename=None):
        '''
        Read ancillary data from filename

        returns LUT_LatLon object
        '''
        if uncompress is None:
            uncompress = filename.endswith(".bz2")
        if uncompress:
            with tempfile.NamedTemporaryFile() as decomp_file, open(filename, 'rb') as fp:
                compdata = fp.read()
                try:
                    decompdata = bz2.decompress(compdata)
                except OSError:
                    raise Exception('Error decompressing {}'.format(filename))
                decomp_file.write(decompdata)
                decomp_file.flush()

                D = self.read(param, decomp_file.name,
                              uncompress=False, orig_filename=filename)

                return D

        if orig_filename is None:
            orig_filename = filename

        hdf = SD(filename)

        assert isinstance(filename, str)
        if param == 'wind_speed':
            # read m_wind and z_wind
            zwind = hdf.select('z_wind').get()
            mwind = hdf.select('m_wind').get()
            wind = np.sqrt(zwind*zwind + mwind*mwind)
            D = LUT_LatLon(wind)
            D.filename = {'meteo': orig_filename}

        elif param == 'surf_press':
            press = hdf.select('press').get()
            D = LUT_LatLon(press)
            D.filename = {'meteo': orig_filename}

        elif param == 'ozone':
            sds = hdf.select('ozone')
            assert 'Dobson units' in sds.attributes()['units']
            ozone = sds.get().astype('float')
            D = LUT_LatLon(ozone)
            D.data.data[D.data.data == 0] = np.NaN
            D.filename = {'ozone': orig_filename}

        else:
            raise Exception('Invalid parameter "{}"'.format(param))

        D.date = datetime.strptime(hdf.attributes()['Start Time'][:13],
                                   '%Y%j%H%M%S')

        hdf.end()

        return D


    def get(self, param, date):
        '''
        Retrieve ancillary parameter at given date

        param:
            * 'wind_speed': surface wind speed in m/s
            * 'surf_press': sea-level pressure in HPa
            * 'ozone': ozone total column in Dobson Units
        '''
        if param in ['wind_speed', 'surf_press']:
            if self.meteo is None:
                res = self.find_meteo(date)
            else:
                res = self.meteo

        elif param in ['ozone']:
            if self.ozone is None:
                res = self.find_ozone(date)
            else:
                res = self.ozone

        else:
            raise Exception('Invalid parameter "{}"'.format(param))

        if isinstance(res, tuple):
            # interpolation
            D1 = self.read(param, res[0])
            D2 = self.read(param, res[1])

            if D1.date == D2.date:
                return D1

            x = (date - D1.date).total_seconds()/(D2.date - D1.date).total_seconds()

            if D1.data.shape == D2.data.shape:
                D = LUT_LatLon((1-x)*D1.data[:,:] + x*D2.data[:,:])
                D.date = date
                D.filename = {list(D1.filename.keys())[0]+'1': list(D1.filename.values())[0],
                              list(D2.filename.keys())[0]+'2': list(D2.filename.values())[0],
                             }
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
            # deactivate interpolation
            return self.read(param, res)


    def download(self, url, target):

        # download that file
        if not exists(dirname(target)):
            makedirs(dirname(target))

        lock = target + '.lock'
        if exists(lock):
            raise Exception('lock file "{}" is present'.format(lock))

        assert basename(url) == basename(target)

        with LockFile(lock):

            cmd = 'wget -nv {} -O {}'.format(url, target+'.tmp')
            ret = system(cmd)
            if ret == 0:
                cmd = 'mv {} {}'.format(target+'.tmp', target)
                system(cmd)
            else:
                if exists(target+'.tmp'):
                    system('rm {}'.format(target+'.tmp'))

        return ret


    def try_resources(self, patterns, dates):
        '''
        Try to access offline or online resource defined by patterns (list of
        strings), in 'dates' (list of dates).
        '''

        # first, try local files
        for date in dates:
            for pattern in patterns:
                target = date.strftime(join(self.directory, '%Y/%j/'+pattern))
                if exists(target):
                    return target

        # then try to download if requested
        if not self.offline:  # If offline flag set, don't download
            for date in dates:
                for pattern in patterns:
                    url = date.strftime(self.url+pattern)
                    target = date.strftime(join(self.directory, '%Y/%j/'+pattern))

                    print('Trying to download', url, '... ')
                    sys.stdout.flush()
                    ret = self.download(url, target)
                    if ret == 0:
                        target = verify(target)
                        return target
                    else:
                        print('failure ({})'.format(ret))

        return None

    def find_meteo(self, date):
        # find closest files before and after
        day = datetime(date.year, date.month, date.day)
        t0 = day + timedelta(hours=6*int(date.hour/6.))
        t1 = day + timedelta(hours=6*(int(date.hour/6.)+1))
        f1 = self.try_resources(self.met_patterns, perdelta(t0, t0-self.delta, -timedelta(hours=6)))
        f2 = self.try_resources(self.met_patterns, perdelta(t1, t1+self.delta,  timedelta(hours=6)))

        # if file after acquisition is not present, use only file before the acquisition
        if (f2 is None) and (f1 is not None):
            f2 = f1
        if None in [f1, f2]:
            raise Exception('Could not find meteo files for {}'.format(date))

        return (f1, f2)

    def find_ozone(self, date):
        # find the matching date or the closest possible, before or after

        t0 = datetime(date.year, date.month, date.day)
        f1 = self.try_resources(self.ozone_patterns, rolling(t0, self.delta, timedelta(days=1)))
        if f1 is None:
            raise Exception('Could not find any valid ozone file for {}'.format(date))

        return f1

