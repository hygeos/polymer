#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gzip
import sys
from os.path import exists, join
from os import rename
import numpy as np
try:
    from urllib.request import urlopen
except ImportError:  # python2
    from urllib import urlopen


class GLOBE(object):
    def __init__(self, directory='globe_dem'):
        """
        GLOBE DEM
        https://www.ngdc.noaa.gov/mgg/topo/gltiles.html
        """
        self.directory = directory
        if not exists(directory):
            raise IOError('Directory {} does not exist.'.format(directory))
        self.file_props = [   # tilename, deltalat, offlat, offlon
            ('a10g.gz', 40, 0,   0  ),
            ('b10g.gz', 40, 0,   90 ),
            ('c10g.gz', 40, 0,   180),
            ('d10g.gz', 40, 0,   270),
            ('e10g.gz', 50, 40,  0  ),
            ('f10g.gz', 50, 40,  90 ),
            ('g10g.gz', 50, 40,  180),
            ('h10g.gz', 50, 40,  270),
            ('i10g.gz', 50, 90,  0  ),
            ('j10g.gz', 50, 90,  90 ),
            ('k10g.gz', 50, 90,  180),
            ('l10g.gz', 50, 90,  270),
            ('m10g.gz', 40, 140, 0  ),
            ('n10g.gz', 40, 140, 90 ),
            ('o10g.gz', 40, 140, 180),
            ('p10g.gz', 40, 140, 270),]


    def fetch_all(self):
        for (tilename, deltalat, _, _) in self.file_props:
            self.read_tile(tilename, deltalat)



    def read_tile(self, tilename, deltalat):
        """
        Reads a single tile

        Example: read_tile('m10g.gz')
        """
        filename = join(self.directory, tilename)

        if not exists(filename):
            print('Downloading {}...'.format(filename))
            url = 'https://www.ngdc.noaa.gov/mgg/topo/DATATILES/elev/'+tilename
            with urlopen(url) as response:
                gzipped = response.read()

            # write (safely) to directory
            with open(filename+'.tmp', 'wb') as fp:
                fp.write(gzipped)
            rename(filename+'.tmp', filename)

        # read the tile
        with gzip.open(filename, 'rb') as fp:
            raw_data = fp.read()

        nlat=21600
        nlon=43200
        NLON = nlon//4
        NLAT = nlat*deltalat//180
        data = np.fromstring(raw_data, dtype='int16').reshape(NLAT,NLON)

        return data


    def get(self, lat, lon, altitude=None):
        """
        Returns the GLOBE altitude for lat, lon (arrays) in m
        when providing altitude, use this array as output
        """
        assert lat.shape == lon.shape

        if altitude is None:
            alt = np.zeros(lat.shape, dtype='float32')+np.NaN
        else:
            assert altitude.shape == lat.shape
            alt = altitude

        isnan = np.isnan(alt)

        for (tilename, deltalat, offlat, offlon) in self.file_props:
            lat_max = 90 - offlat
            lat_min = 90 - offlat - deltalat
            lon_min = -180 + offlon
            lon_max = lon_min + 90

            target = ((lat_min <= lat) & (lat <= lat_max) &
                      (lon_min <= lon) & (lon <= lon_max))

            if not (target & isnan).any():
                continue

            data = self.read_tile(tilename, deltalat)[::-1,:]
            data[data<0] = 0.
            (h, w) = data.shape

            # fill only the target pixels
            ok = target & isnan

            # coordinates of the target pixels in data
            ilat_ = ((lat[ok] - lat_min)/deltalat*(h-1)).astype('int')
            ilon_ = ((lon[ok] - lon_min)/(90.)*(w-1)).astype('int')

            alt[ok] = data[(ilat_, ilon_)]

        assert not np.isnan(alt).any(), 'Error in GLOBE DEM'

        return alt


if __name__ == "__main__":
    directory = sys.argv[1]
    print('Altitude of the Mount Everest:',
          GLOBE(directory=directory).get(np.array([27.986065]), np.array([86.922623])))

