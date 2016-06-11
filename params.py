#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

class Params(object):
    '''
    Sensor non-specific parameters
    '''
    def __init__(self):
        '''
        define common parameters
        '''

        # no2 absorption data
        self.no2_climatology = '/home/francois/MERIS/POLYMER/auxdata/common/no2_climatology.hdf'
        self.no2_frac200m  = '/home/francois/MERIS/POLYMER/auxdata/common/trop_f_no2_200m.hdf'

        # cloud masking
        self.thres_Rcloud = 0.2
        self.thres_Rcloud_std = 0.04

        # optimization parameters
        self.force_initialization = False
        self.max_iter = 100
        self.size_end_iter = 0.005
        self.initial_point_1 = [-1, 0]
        self.initial_point_2 = [1, 1]
        self.initial_step = [0.2, 0.2]
        self.bounds = [[-2, 2], [-3, 3]]

        self.thres_chi2 = 0.005

        # Constraint on bbs: amplitude, sigma(chl=0.01), sigma(chl=0.1)
        # (disactivate with amplitude == 0)
        self.constraint_bbs = [1e-3, 0.2258, 0.9233]

        self.partial = 0    # whether to perform partial processing
                            #       0: standard processing
                            #       1: stop at minimize
                            #       2: stop at rayleigh correction

    def bands_read(self):
        assert (np.diff(self.bands_corr) > 0).all()
        assert (np.diff(self.bands_oc) > 0).all()
        assert (np.diff(self.bands_rw) > 0).all()
        bands_read = set(self.bands_corr)
        bands_read = bands_read.union(self.bands_oc)
        bands_read = bands_read.union(self.bands_rw)
        return sorted(bands_read)

    def print_info(self):
        print self.__class__
        for k, v in self.__dict__.iteritems():
            print '*', k,':', v

    def update(self, **kwargs):

        # don't allow for 'new' attributes
        for k in kwargs:
            if k not in self.__dict__:
                raise Exception('{}: attribute "{}" is unknown'.format(self.__class__, k))

        self.__dict__.update(kwargs)


