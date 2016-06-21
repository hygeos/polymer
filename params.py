#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import os
from os.path import join

class Params(object):
    '''
    A class to store the processing parameters
    '''
    def __init__(self, sensor, **kwargs):

        if 'dir_base' in kwargs:
            self.dir_base = kwargs['dir_base']
        else:
            self.dir_base = os.getcwd()

        self.dir_common = join(self.dir_base, 'auxdata/common/')

        # define common parameters
        self.common()

        # define sensor-specific parameters
        self.sensor_specific(sensor)

        # setup custom parameters
        self.update(**kwargs)

    def common(self):
        '''
        define common parameters
        '''

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
                            #   0: standard processing
                            #   1: stop before minimize
                            #   2: stop before rayleigh correction
                            #   3: stop before cloud mask
                            #   4: stop before gaseous correction
                            #   5: stop before conversion to reflectance

        self.multiprocessing = False

        # no2 absorption data
        self.no2_climatology = join(self.dir_base, 'auxdata/common/no2_climatology.hdf')
        self.no2_frac200m  = join(self.dir_base, 'auxdata/common/trop_f_no2_200m.hdf')


    def sensor_specific(self, sensor):
        '''
        define sensor-specific default parameters
        '''
        if sensor == 'MERIS':
            self.defaults_meris()
        elif sensor == 'MSI':
            self.defaults_msi()
        elif sensor == 'OLCI':
            self.defaults_olci()
        elif sensor == 'VIIRS':
            self.defaults_viirs()
        elif sensor == 'MODIS':
            self.defaults_olci()
        elif sensor == 'SeaWiFS':
            self.defaults_seawifs()
        else:
            raise Exception('Params.sensor_specific: invalid sensor "{}"'.format(sensor))

    def defaults_meris(self):
        '''
        define default parameters for ENVISAT/MERIS
        '''

        self.bands_corr = [412,443,490,510,560,620,665,        754,    779,865]
        self.bands_oc =   [412,443,490,510,560,620,665,        754,    779,865]
        self.bands_rw =   [412,443,490,510,560,620,665,        754,    779,865]

        self.lut_file = join(self.dir_base, 'LUTS/MERIS/LUTB.hdf')
        self.bands_lut = [412,443,490,510,560,620,665,681,709,754,760,779,865,885,900]

        self.band_cloudmask = 865

        self.K_OZ = {
                    412: 0.000301800 , 443: 0.00327200 ,
                    490: 0.0211900   , 510: 0.0419600  ,
                    560: 0.104100    , 620: 0.109100   ,
                    665: 0.0511500   , 681: 0.0359600  ,
                    709: 0.0196800   , 754: 0.00955800 ,
                    760: 0.00730400  , 779: 0.00769300 ,
                    865: 0.00219300  , 885: 0.00121100 ,
                    900: 0.00151600  ,
                }

        self.K_NO2 = {
                412: 6.074E-19 , 443: 4.907E-19,
                490: 2.916E-19 , 510: 2.218E-19,
                560: 7.338E-20 , 620: 2.823E-20,
                665: 6.626E-21 , 681: 6.285E-21,
                709: 4.950E-21 , 754: 1.384E-21,
                760: 4.717E-22 , 779: 3.692E-22,
                865: 2.885E-23 , 885: 4.551E-23,
                900: 5.522E-23 ,
                }

        self.central_wavelength = {
                412: 412.691 , 443: 442.559,
                490: 489.882 , 510: 509.819,
                560: 559.694 , 620: 619.601,
                665: 664.573 , 681: 680.821,
                709: 708.329 , 754: 753.371,
                760: 761.508 , 779: 778.409,
                865: 864.876 , 885: 884.944,
                900: 900.000 ,
                }

        self.NO2_CLIMATOLOGY = join(self.dir_base, 'auxdata/common/no2_climatology.hdf')
        self.NO2_FRAC200M = join(self.dir_base, 'auxdata/common/trop_f_no2_200m.hdf')


    def defaults_olci(self):
        '''
        define default parameters for Sentinel-3/OLCI
        '''

        self.bands_corr = [    412,443,490,510,560,620,665,754,779,865]
        self.bands_oc   = [    412,443,490,510,560,620,665,754,779,865]
        self.bands_rw   = [400,412,443,490,510,560,620,665,754,779,865]

        self.bands_lut = [400,412,443,490,510,560,620,665,674,681,
                          709,754,760,764,767,779,865,885,900,940,
                          1020,1375,1610,2250]

        self.lut_file = join(self.dir_base, 'LUTS/OLCI/LUT.hdf')

        self.band_cloudmask = 865

        # central wavelength of the detector where the Rayleigh optical thickness is calculated
        # (detector 374 of camera 3)
        self.central_wavelength = {
                400 : 400.664  , 412 : 412.076 ,
                443 : 443.183  , 490 : 490.713 ,
                510 : 510.639  , 560 : 560.579 ,
                620 : 620.632  , 665 : 665.3719,
                674 : 674.105  , 681 : 681.66  ,
                709 : 709.1799 , 754 : 754.2236,
                760 : 761.8164 , 764 : 764.9075,
                767 : 767.9734 , 779 : 779.2685,
                865 : 865.4625 , 885 : 884.3256,
                900 : 899.3162 , 940 : 939.02  ,
                1020: 1015.9766, 1375: 1375.   ,
                1610: 1610.    , 2250: 2250.   ,
                }

        # from SeaDAS v7.3.2
        self.K_OZ = {
                400 : 2.985E-06, 412 : 2.341E-04,
                443 : 2.897E-03, 490 : 2.066E-02,
                510 : 4.129E-02, 560 : 1.058E-01,
                620 : 1.085E-01, 665 : 5.005E-02,
                674 : 4.095E-02, 681 : 3.507E-02,
                709 : 1.887E-02, 754 : 8.743E-03,
                760 : 6.713E-03, 764 : 6.916E-03,
                768 : 6.754E-03, 779 : 7.700E-03,
                865 : 2.156E-03, 885 : 1.226E-03,
                900 : 1.513E-03, 940 : 7.120E-04,
                1020: 8.448E-05,
                }

        # from SeaDAS v7.3.2
        self.K_NO2 = {
                400 : 6.175E-19, 412 : 6.083E-19,
                443 : 4.907E-19, 490 : 2.933E-19,
                510 : 2.187E-19, 560 : 7.363E-20,
                620 : 2.818E-20, 665 : 6.645E-21,
                674 : 1.014E-20, 681 : 6.313E-21,
                709 : 4.938E-21, 754 : 1.379E-21,
                761 : 4.472E-22, 764 : 6.270E-22,
                768 : 5.325E-22, 779 : 3.691E-22,
                865 : 2.868E-23, 885 : 4.617E-23,
                900 : 5.512E-23, 940 : 3.167E-24,
                1020: 0.000E+00,
                }

    def defaults_msi(self):
        self.lut_file = join(self.dir_base, 'LUTS/MSI/LUT.hdf')

        # FIXME
        self.bands_corr = [443,490,560,665,705,740,783,    865,         1610,    ]
        self.bands_oc   = [443,490,560,665,705,740,783,    865,         1610,    ]
        self.bands_rw   = [443,490,560,665,705,740,783,    865,         1610,    ]

        self.bands_lut =  [443,490,560,665,705,740,783,842,865,945,1375,1610,2190]
        self.central_wavelength = dict(map(
            lambda x: (x, float(x)),
            [443,490,560,665,705,740,783,842,865,945,1375,1610,2190]))

        self.band_cloudmask = 865

        self.K_OZ = {   # FIXME
                443 : 0.,
                490 : 0.,
                560 : 0.,
                665 : 0.,
                705 : 0.,
                740 : 0.,
                783 : 0.,
                842 : 0.,
                865 : 0.,
                945 : 0.,
                1375: 0.,
                1610: 0.,
                2190: 0.,
                }

        self.K_NO2 = {   # FIXME
                443 : 0.,
                490 : 0.,
                560 : 0.,
                665 : 0.,
                705 : 0.,
                740 : 0.,
                783 : 0.,
                842 : 0.,
                865 : 0.,
                945 : 0.,
                1375: 0.,
                1610: 0.,
                2190: 0.,
                }

    def defaults_viirs(self):

        self.bands_corr = [410,443,486,551,671,745,862]
        self.bands_oc   = [410,443,486,551,671,745,862]
        self.bands_rw   = [410,443,486,551,671,745,862]
        self.bands_lut  = [410,443,486,551,671,745,862,1238,1601,2257]

        self.band_cloudmask = 862

        self.central_wavelength = dict(map(
            lambda x: (x, float(x)),
            [410,443,486,551,671,745,862,1238,1601,2257]))

        self.K_OZ = {  # from SeaDAS
                410: 5.827E-04,
                443: 3.079E-03,
                486: 1.967E-02,
                551: 8.934E-02,
                671: 4.427E-02,
                745: 1.122E-02,
                862: 2.274E-03,
                1238:0.
                }
        self.K_NO2 = {  # from SeaDAS
                410: 5.914E-19,
                443: 5.013E-19,
                486: 3.004E-19,
                551: 1.050E-19,
                671: 1.080E-20,
                745: 2.795E-21,
                862: 3.109E-22,
                1238:0.000E+00,
                }

        self.lut_file = join(self.dir_base, 'LUTS/VIIRS/LUT.hdf')

    def defaults_seawifs(self):
        raise NotImplementedError

    def defaults_modis(self):
        raise NotImplementedError

    def bands_read(self):
        assert (np.diff(self.bands_corr) > 0).all()
        assert (np.diff(self.bands_oc) > 0).all()
        assert (np.diff(self.bands_rw) > 0).all()
        bands_read = set(self.bands_corr)
        bands_read = bands_read.union(self.bands_oc)
        bands_read = bands_read.union(self.bands_rw)
        return sorted(bands_read)

    def print_info(self):
        print(self.__class__)
        for k, v in self.__dict__.iteritems():
            print('*', k,':', v)

    def update(self, **kwargs):

        # don't allow for 'new' attributes
        for k in kwargs:
            if k not in self.__dict__:
                raise Exception('{}: attribute "{}" is unknown'.format(self.__class__, k))

        self.__dict__.update(kwargs)


