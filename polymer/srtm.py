#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function, division, absolute_import
import zipfile
try:
    from urllib.request import urlopen
    from urllib.error import HTTPError
except ImportError:  # python2
    from urllib2 import urlopen, HTTPError
import numpy as np
import io
from geoutils.misc import ListOnDisk
from os.path import exists, join
from os import makedirs, rename
import sys



def read_hgt(filename):
    """
    Reads a compressed SRTM file (binary) to a numpy array
    """
    assert filename.endswith('.zip')
    with open(filename, 'rb') as fp:
        zipped = fp.read()

    # uncompress
    zf = zipfile.ZipFile(io.BytesIO(zipped))

    # removes files starting with a dot
    ls = [f for f in zf.filelist if not f.filename.startswith('.')]

    assert len(ls) == 1, 'Invalid archive {} (contains {})'.format(filename, [f.filename for f in zf.filelist])

    data = zf.open(ls[0].filename).read()

    # convert to numpy array
    data = np.fromstring(data, dtype='>i2')  # big endian int16
    N = int(np.sqrt(data.size))
    return data.reshape((N, N))


class SRTM3(object):
    """
    SRTM3 digital elevation model, version 2.1

    3 arc-second (~90m) - Between 56S and 60N

    SRTM3().get(lat, lon)    # where lat and lon are array-like
    returns an array-like in m

    missing: what to provide in case of missing value
        * a float
        * None : raise an error
        * a DEM object to use as a backup, when SRTM provides no data
          Example: backup=GLOBE(...)
                   (see getasse30.py)

    Full initialization with SRTM3(directory=<directory>).fetch_all()
    """
    def __init__(self, directory='data_srtm3', missing=None, verbose=False):
        self.directory = directory
        self.verbose = verbose
        self.missing = missing
        self.folders = ['Africa', 'Australia', 'Eurasia', 'Islands',
                        'North_America', 'South_America']

        if not exists(directory):
            raise IOError('Directory "{}" does not exist'.format(directory))

        # list of available tiles
        self.file_available_tiles = join(directory, 'available_tiles.txt')
        if not exists(self.file_available_tiles):
            print('Listing all available tiles...')
            tiles = self.list_srtm3_tiles()
            print('Listed {} tiles.'.format(len(tiles)))
            with open(self.file_available_tiles, 'w') as fp:
                fp.write('\n'.join(tiles))

        # read the available tiles
        # (as a dictionary tile -> region)
        with open(self.file_available_tiles) as fp:
            avail = fp.read().split('\n')
            self.available = dict([a.split('/')[::-1] for a in avail])


    def list_srtm3_tiles(self):
        """
        List available SRTM3 tiles
        """

        list_tiles = []

        for f in self.folders:
            print('Fetching the list of available tiles for {}...'.format(f))
            url = 'https://dds.cr.usgs.gov/srtm/version2_1/SRTM3/{}/'.format(f)
            with urlopen(url) as response:
                html = response.read()

            # extract all available tiles from the given html page
            tiles = [f+'/'+a[a.index(b'"')+1:a.rindex(b'"')].decode('ascii').replace('.hgt.zip', '')
                     for a in html.split(b'\n') if b'zip' in a]
            list_tiles += tiles

        return list_tiles


    def fetch_all(self):
        lat, lon = np.meshgrid(
                        np.linspace(-90, 90, 1000),
                        np.linspace(-180, 180, 1000),
                        )
        self.get(lat, lon)


    def get(self, lat, lon):
        """
        Reads SRTM3 data from (lat, lon) in m

        data voids are assigned the value -32768
        """
        url_base = 'https://dds.cr.usgs.gov/srtm/version2_1/SRTM3/{}/{}'

        alt = np.zeros(lat.shape, dtype='float32') + np.NaN

        # determine the list of SRTM tiles
        ilatlon = 1000*(90+np.floor(np.array(lat)).astype('int')) + 180 + np.floor(np.array(lon)).astype('int')

        tids = set(ilatlon.ravel())
        N = len(tids)
        for i, tid in enumerate(tids):
            ilat = tid // 1000 - 90
            ilon = tid % 1000 - 180

            tile_name = '{}{:02d}{}{:03d}'.format(
                                    {True: 'N', False: 'S'}[ilat>=0],
                                    abs(ilat),
                                    {True: 'E', False: 'W'}[ilon>=0],
                                    abs(ilon))

            file_name = tile_name+'.hgt.zip'

            # check whether file is available
            if not tile_name in self.available:
                continue

            # download tile if necessary
            srtm_data = join(self.directory, file_name)
            if self.verbose:
                print('[{}/{}] {}: '.format(i, N, srtm_data), end='')
                sys.stdout.flush()
            if not exists(srtm_data):
                folder = self.available[tile_name]
                url = url_base.format(folder, file_name)

                if self.verbose:
                    print('downloading... ', end='')
                    sys.stdout.flush()

                # download
                with urlopen(url) as response:
                    zipped = response.read()

                # write (safely) to local directory
                with open(srtm_data+'.tmp', 'wb') as fp:
                    fp.write(zipped)
                rename(srtm_data+'.tmp', srtm_data)

            if self.verbose:
                print('reading...')
                sys.stdout.flush()
            data = read_hgt(srtm_data)

            n = data.shape[0]
            assert data.shape == (n, n)

            # fills altitude for the current tile
            ok = (ilatlon == tid)

            ilat_ = n-1 - ((np.array(lat)[ok] - ilat)*n).astype('int')
            ilon_ = ((np.array(lon)[ok] - ilon)*n).astype('int')

            alt[ok] = data[(ilat_, ilon_)]

        alt[alt == -32768] = np.NaN

        if self.missing is None:
            assert not np.isnan(alt).any(), 'There are invalid data in SRTM'
        elif hasattr(self.missing, 'get'):
            alt = self.missing.get(lat, lon, altitude=alt)
        else: # assuming float
            alt[np.isnan(alt)] = self.missing

        assert not np.isnan(alt).any()

        return alt



if __name__ == "__main__":
    # download all to the directory provided in argument
    directory = sys.argv[1]
    SRTM3(directory=directory, verbose=True).fetch_all()
