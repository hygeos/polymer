#!/usr/bin/env python
# encoding: utf-8

import numpy as np

class Block(object):

    def __init__(self, size, offset=None, bands=None):

        self.size = size
        self.offset = offset
        self.bands = bands
        self._data  = {}

        self.sza = None  # sun zenith angle (degrees)
        self.vza = None  # view zenith angle (degrees)
        self.saa = None  # sun azimuth angle (degrees)
        self.vaa = None  # view azimuth angle (degrees)
        self.F0 = None   # solar irradiance

        self.Ltoa = None  # TOA radiance (nbands, nx, ny)
        self.Rtoa = None  # TOA reflectance
        self.Rtoa_gc = None  # TOA reflectance, corrected for gas absorption
        self.Rprime = None  # TOA reflectance, corrected for gas absorption, Rayleigh scattering and wind speed

        self.wind_speed = None # wind speed module (m/s)

    def __str__(self):
        return 'block: size {}, offset {}'.format(self.size, self.offset)

    @property
    def raa(self):
        ''' relative azimuth angle, in degrees '''
        if not 'raa' in self._data:
            raa = self.saa - self.vaa
            raa[raa<0.] += 360;
            raa[raa>360.] -= 360;
            raa[raa>180.] = 360. - raa[raa>180.]
            self._data['raa'] = raa
        return self._data['raa']

    @property
    def mus(self):
        if not 'mus' in self._data:
            self._data['mus'] = np.cos(self.sza*np.pi/180.)
        return self._data['mus']

    @property
    def muv(self):
        if not 'muv' in self._data:
            self._data['muv'] = np.cos(self.vza*np.pi/180.)
        return self._data['muv']

    @property
    def nbands(self):
        return len(self.bands)

