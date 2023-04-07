#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os.path import isdir
import xarray as xr
from datetime import datetime, timedelta
import numpy as np
from os import makedirs, system, remove
from os.path import join, exists, dirname, basename
from pyhdf.SD import SD
from polymer.luts import LUT, Idx
from warnings import warn
import sys
import bz2
import tempfile
from dateutil.parser import parse
from .utils import closest, round_date


# resources are a list of functions taking the date, and returning a list of
# closest or bracketing resources, defined by (pattern, date)
forecast_resources = [
    lambda date: [('N%Y%j%H_MET_NCEP_1440x0721_f{}.hdf'.format(
                   '012' if (d.hour % 2 == 0) else '015'), d)
                  for d in round_date(date, 3)]
]

default_met_resources = [
    lambda date: [('GMAO_FP.%Y%m%dT%H0000.MET.NRT.nc', d)
                  for d in round_date(date, 3)],
    lambda date: [('N%Y%j%H_MET_NCEPR2_6h.hdf.bz2', d) for d in round_date(date, 6)],
    lambda date: [('N%Y%j%H_MET_NCEP_6h.hdf.bz2', d) for d in round_date(date, 6)],
    lambda date: [('N%Y%j%H_MET_NCEP_6h.hdf', d) for d in round_date(date, 6)],
]

default_oz_resources = [
    lambda date: [('GMAO_FP.%Y%m%dT%H0000.MET.NRT.nc', d)
                  for d in round_date(date, 3)],
    lambda date: [('N%Y%j00_O3_AURAOMI_24h.hdf', closest(date, 24))],
    lambda date: [('N%Y%j00_O3_TOMSOMI_24h.hdf', closest(date, 24))],
    lambda date: [('S%Y%j00%j23_TOAST.OZONE', closest(date, 24))],
    lambda date: [('S%Y%j00%j23_TOVS.OZONE', closest(date, 24))],
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
        self.dtype = A.dtype

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
    * allow_forecast: bool, default True
        allow to use NCEP forecast data for NRT production
    * allow_standard bool, default True
        allow to use other non-forecast data
    '''
    def __init__(self, meteo=None, ozone=None,
                 directory='ANCILLARY/METEO/',
                 offline=False,
                 allow_standard=True,
                 allow_forecast=True,
                 delta=None,
                 met_patterns=None, ozone_patterns=None):
        self.met_resources = []
        self.ozone_resources = []
        if allow_standard:
            self.met_resources += default_met_resources
            self.ozone_resources += default_oz_resources
        if allow_forecast:
            self.met_resources += forecast_resources
            self.ozone_resources += forecast_resources

        self.meteo = meteo
        self.ozone = ozone
        self.directory = directory
        self.offline = offline

        self.url = 'https://oceandata.sci.gsfc.nasa.gov/cgi/getfile/'

        assert isdir(directory), 'Directory {} does not exist. Please create it, by default it will be automatically populated with ancillary data. Please see help for class Ancillary_NASA for more details.'.format(directory)

        # arguments to be deprecated
        if delta is not None:
            raise DeprecationWarning('The `delta` parameter has been deprecated.')
        if met_patterns is not None:
            raise DeprecationWarning('The `met_patterns` parameter has been deprecated.')
        if ozone_patterns is not None:
            raise DeprecationWarning('The `ozone_patterns` parameter has been deprecated.')


    def read(self, param, filename,
             uncompress=None, orig_filename=None):
        if filename.endswith('.nc'):
            return self.read_nc(param, filename,
                                orig_filename=orig_filename)
        else:
            return self.read_hdf(
                param, filename,
                uncompress=uncompress, orig_filename=orig_filename)

    def read_nc(self, param, filename, orig_filename=None):
        if orig_filename is None:
            orig_filename = filename
        ds = xr.open_dataset(filename)
        if param == 'wind_speed':
            # read m_wind and z_wind
            
            uwind = ds['U10M'].values
            vwind = ds['V10M'].values
            wind = np.sqrt(uwind*uwind + vwind*vwind)
            D = LUT_LatLon(wind[::-1,:])
            D.filename = {'meteo': orig_filename}

        elif param == 'surf_press':
            press = ds['PS'].values
            if 'units' in ds['PS'].attrs.keys() and ds['PS'].units == 'Pa':
                press /= 100.
            D = LUT_LatLon(press[::-1,:])
            D.filename = {'meteo': orig_filename}

        elif param == 'ozone':
            assert 'dobson' in ds['TO3'].units.lower()
            ozone = ds['TO3'].values
            D = LUT_LatLon(ozone[::-1,:])
            D.data.data[D.data.data == 0] = np.NaN
            D.filename = {'ozone': orig_filename}

        D.date = parse(ds.time_coverage_start).replace(tzinfo=None)
        
        return D

    def read_hdf(self, param, filename,
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
                res = self.find(date, self.met_resources)
            else:
                res = [self.meteo]

        elif param in ['ozone']:
            if self.ozone is None:
                res = self.find(date, self.ozone_resources)
            else:
                res = [self.ozone]

        else:
            raise Exception('Invalid parameter "{}"'.format(param))

        if len(res) > 1:
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
            return self.read(param, res[0])


    def download(self, url, target):

        # download that file
        if not exists(dirname(target)):
            makedirs(dirname(target))

        lock = target + '.lock'
        if exists(lock):
            raise Exception('lock file "{}" is present'.format(lock))

        assert basename(url) == basename(target)

        with LockFile(lock):

            # follows https://support.earthdata.nasa.gov/index.php?/Knowledgebase/Article/View/43/21/how-to-access-urs-gated-data-with-curl-and-wget
            cmd = 'wget -nv --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --keep-session-cookies --auth-no-challenge {} -O {}'.format(url, target+'.tmp')
            ret = system(cmd)
            if ret == 0:
                # sanity check
                # raise an error in case of authentication error
                # check that downloaded file is not HTML
                with open(target+'.tmp', 'rb') as fp:
                    errormsg = 'Error authenticating to NASA EarthData for downloading ancillary data. ' \
                    'Please provide authentication through .netrc. See more information on ' \
                    'https://support.earthdata.nasa.gov/index.php?/Knowledgebase/Article/View/43/21/how-to-access-urs-gated-data-with-curl-and-wget'
                    assert not fp.read(100).startswith(b'<!DOCTYPE html>'), errormsg

                cmd = 'mv {} {}'.format(target+'.tmp', target)
                system(cmd)

            else:
                if exists(target+'.tmp'):
                    system('rm {}'.format(target+'.tmp'))

        return ret


    
    def try_resource(self, pattern, date):
        """
        Try to access pattern (string, like 'N%Y%j%H_MET_NCEP_1440x0721_f015.hdf')
        at a given date
        """
        target = date.strftime(join(self.directory, '%Y/%j/'+pattern))
        if exists(target):
            return target
        url = date.strftime(self.url+pattern)

        if not self.offline:
            print('Trying to download', url, '... ')
            sys.stdout.flush()
            ret = self.download(url, target)
            if ret == 0:
                target = verify(target)
                return target
            else:
                print('failure ({})'.format(ret))

        return None


    def find(self, date, patterns):
        '''
        Try to access offline or online resource defined by patterns,
        at `date`
        '''
        for pattern in patterns:
            res = [self.try_resource(pat, d) for pat, d in pattern(date)]

            if None not in res:
                return res

