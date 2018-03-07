#!/usr/bin/env python
# -*- coding: utf-8 -*-


try:
    from urllib.request import urlopen
except ImportError:  # python2
    from urllib import urlopen
from urllib.error import HTTPError
import numpy as np
from geoutils.misc import ListOnDisk
from os.path import exists, join
from os import makedirs, rename
from osgeo import gdal, gdal_array


def download_SW(url):
    """
    Downloads one JRC Global Surface Water file at a given url (to memory)

    https://storage.googleapis.com/global-surface-water/downloads/occurrence/occurrence_180W_80N.tif
    """
    try:
        with urlopen(url) as response:
            data = response.read()
    except HTTPError:
        return None
    return data


def read_tif(filename):
    """
    Open geotiff and return a numpy array
    """
    geotiff = gdal.Open(filename)
    data = geotiff.ReadAsArray()
    return data

def get_sw(lat, lon, cache_dir='cache_sw'):
    """
    Reads JRC Global Surface Water data from (lat, lon) in %

    """
    if not exists(cache_dir):
        print('Creating directory "{}"'.format(cache_dir))
        makedirs(cache_dir)

    url_base = 'https://storage.googleapis.com/global-surface-water/downloads/occurrence/{}'

    sw = np.zeros(lat.shape, dtype='float32')

    # determine SRTM tiles
    ilatlon = 1000*(90+np.floor(np.array(lat)).astype('int')) + 180 + np.floor(np.array(lon)).astype('int')


    missing = ListOnDisk(join(cache_dir, 'empty_tiles_jrc.txt'))

    tids = set(ilatlon.ravel())
    for i, tid in enumerate(tids):
        ilat = (((tid//1000 - 90)//10)+1)*10
        ilon = ((tid%1000 - 180)//10)*10

        tile_name = '{}{}_{}{}'.format(abs(ilon), {True: 'E', False: 'W'}[ilon>=0], abs(ilat), {True: 'N', False: 'S'}[ilat>=0])

        file_name = 'occurrence_{}.tif'.format(tile_name)

        # check is file is known to be missing
        if tile_name in missing:
            continue

        # check in cache
        sw_cache = join(cache_dir, file_name)
        if exists(sw_cache):
            # tile is present in cache
            print('[{}/{}] {}: {}'.format(i+1, len(tids), tile_name, 'from cache'))
        else:
            url = url_base.format(file_name)
            raw_data = download_SW(url)
            if raw_data is None:
                missing.append(tile_name)
                print('[{}/{}] {}: {}'.format(i+1, len(tids), tile_name, 'empty'))
                continue
            else:
                print('[{}/{}] {}: {}'.format(i+1, len(tids), tile_name, 'non-empty'))

                # write (safely) to cache
                with open(sw_cache+'.tmp', 'wb') as fp:
                    fp.write(raw_data)
                rename(sw_cache+'.tmp', sw_cache)

        data = read_tif(sw_cache)

        n = data.shape[0]
        assert data.shape == (n, n)

        # fills altitude for the current tile
        ok = (ilatlon == tid)

        ilat_ = ((ilat - np.array(lat)[ok] )*n/10).astype('int')
        ilon_ = ((np.array(lon)[ok] - ilon)*n/10).astype('int')

        sw[ok] = data[(ilat_, ilon_)]
        sw[(sw==255)] = 100

    # write missing files list
    missing.write()

    return sw


class DEM_SW(object):
    def __init__(self, cache_dir='cache_sw', on_error=0., threshold=50.):
        self.cache_dir = cache_dir
        self.on_error = on_error
        self.threshold = threshold

    def get(self, lat, lon):
        """
        Returns water occurrence in %
        """
        data = get_sw(lat, lon, cache_dir=self.cache_dir)
#        if self.on_error == 'raise':
#            assert not (data == -32768).any(), 'Invalid data in SRTM' 
#        else:
#            data[data == -32768] = self.on_error A
        data[(data<self.threshold)] = 0.
        return data

if __name__=='__main__':
    dataobj = DEM_SW(cache_dir='/rfs/user/bruno/jrcgsw', threshold=40)

    lat = np.array([[43.554875,43.554875,43.554875],[43.554625,43.554625,43.554625],[43.554375,43.554375,43.554375]])
    lon = np.array([[4.624375,4.624625,4.624875],[4.624375,4.624625,4.624875],[4.624375,4.624625,4.624875]])
    sw_test = np.array([[67,0,0],[91,49,0],[91,49,0]])
    sw = dataobj.get(lat, lon)
    print('sw : ', sw)

    if (np.equal(sw_test, sw).all()):
        print('lecture ok!!')
