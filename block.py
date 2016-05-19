#!/usr/bin/env python
# encoding: utf-8

import numpy as np

class Block(object):

    def __init__(self, size, offset=None, bands=None):

        self.size = size
        self.offset = offset
        self.bands = bands

        # self.sza = None       # sun zenith angle [degrees] (nx, ny)
        # self.vza = None       # view zenith angle [degrees] (nx, ny)
        # self.saa = None       # sun azimuth angle [degrees] (nx, ny)
        # self.vaa = None       # view azimuth angle [degrees] (nx, ny)
        # self.F0 = None        # per-pixel solar irradiance (nbands, nx, ny)
        # self.wavelen = None   # per-pixel wavelength [nm] (nbands, nx, ny)


        # self.Ltoa = None  # TOA radiance (nbands, nx, ny)
        # self.Rtoa = None  # TOA reflectance
        # self.Rtoa_gc = None  # TOA reflectance, corrected for gas absorption
        # self.Rprime = None  # TOA reflectance, corrected for gas absorption, Rayleigh scattering and wind speed

        # self.wind_speed = None # wind speed module (m/s)

        # properties
        # self._raa = None
        # self._mus = None
        # self._muv = None
        # self._air_mass = None

    def datasets(self):
        '''
        returns a list of the datasets of the block
        '''
        return self.__dict__.keys()

    def __getitem__(self, name):
        return self.__dict__[name]

    def __str__(self):
        return 'block: size {}, offset {}'.format(self.size, self.offset)

    @property
    def raa(self):
        ''' relative azimuth angle, in degrees '''
        if '_raa' not in self.datasets():
            raa = self.saa - self.vaa
            raa[raa<0.] += 360;
            raa[raa>360.] -= 360;
            raa[raa>180.] = 360. - raa[raa>180.]
            self._raa = raa
        return self._raa

    @property
    def mus(self):
        if '_mus' not in self.datasets():
            self._mus = np.cos(self.sza*np.pi/180.)
        return self._mus

    @property
    def air_mass(self):
        if '_air_mass' not in self.datasets():
            self._air_mass = 1/self.muv + 1/self.mus
        return self._air_mass

    @property
    def muv(self):
        if '_muv' not in self.datasets():
            self._muv = np.cos(self.vza*np.pi/180.)
        return self._muv

    @property
    def nbands(self):
        return len(self.bands)

