#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function, division, absolute_import
from numpy import cos, sqrt, pi, arccos
from collections import OrderedDict

class Block(object):

    def __init__(self, size, offset=None, bands=None):

        self.size = size
        self.offset = offset
        self.bands = bands
        self.attributes = OrderedDict()

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
            raa[raa<0.] += 360
            raa[raa>360.] -= 360
            raa[raa>180.] = 360. - raa[raa>180.]
            self._raa = raa
        return self._raa

    @property
    def mus(self):
        if '_mus' not in self.datasets():
            self._mus = cos(self.sza*pi/180.)
        return self._mus

    @property
    def air_mass(self):
        if '_air_mass' not in self.datasets():
            self._air_mass = 1/self.muv + 1/self.mus
        return self._air_mass

    @property
    def muv(self):
        if '_muv' not in self.datasets():
            self._muv = cos(self.vza*pi/180.)
        return self._muv

    @property
    def scattering_angle(self):
        if '_scat_angle' not in self.datasets():
            mu_s = self.mus
            mu_v = self.muv
            phi = self.raa
            sa = -mu_s*mu_v - sqrt( (1.-mu_s*mu_s)*(1.-mu_v*mu_v) ) * cos(phi*pi/180.)
            self._scat_angle = arccos(sa)*180./pi
        return self._scat_angle

    @property
    def nbands(self):
        return len(self.bands)

