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
        self.sensor = sensor

        self.dir_common = join(self.dir_base, 'auxdata/common/')

        # define common parameters
        self.common()

        # define sensor-specific parameters
        self.sensor_specific(sensor)

        # setup custom parameters
        self.update(**kwargs)

        # finalization
        self.finalize()

    def common(self):
        '''
        define common parameters
        '''

        # cloud masking (negative values disactivate)
        self.thres_Rcloud = 0.2
        self.thres_Rcloud_std = 0.04

        # optimization parameters
        self.force_initialization = False
        self.reinit_rw_neg = False
        self.max_iter = 100
        self.size_end_iter = 0.005
        self.initial_point_1 = [-1, 0]
        self.initial_point_2 = [1, 1]
        self.initial_step = [0.2, 0.2]
        self.bounds = [[-2, 2], [-3, 3]]
        self.metrics = 'W_dR2_norm'
        self.glint_precorrection = True

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

        self.weights_corr = None
        self.weights_oc = None
        self.atm_model = 'T0,-1,Rmol'
        self.normalize = True

        self.water_model = 'PR05'

        self.alt_gamma_bb = False  # PR05 model only

        # no2 absorption data
        self.no2_climatology = join(self.dir_base, 'auxdata/common/no2_climatology.hdf')
        self.no2_frac200m  = join(self.dir_base, 'auxdata/common/trop_f_no2_200m.hdf')

        self.multiprocessing = False
        self.verbose = True

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
            self.defaults_modis()
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

        self.calib = {
                    412: 1.0, 443: 1.0,
                    490: 1.0, 510: 1.0,
                    560: 1.0, 620: 1.0,
                    665: 1.0, 681: 1.0,
                    709: 1.0, 754: 1.0,
                    760: 1.0, 779: 1.0,
                    865: 1.0, 885: 1.0,
                    900: 1.0,
                }

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
        self.bands_rw   = [400,412,443,490,510,560,620,665,754,779,865,1020]

        self.bands_lut = [400,412,443,490,510,560,620,665,674,681,
                          709,754,760,764,767,779,865,885,900,940,
                          1020,1375,1610,2250]

        self.lut_file = join(self.dir_base, 'LUTS/OLCI/LUT.hdf')

        self.band_cloudmask = 865

        self.calib = {
                400 : 1.0, 412 : 1.0,
                443 : 1.0, 490 : 1.0,
                510 : 1.0, 560 : 1.0,
                620 : 1.0, 665 : 1.0,
                674 : 1.0, 681 : 1.0,
                709 : 1.0, 754 : 1.0,
                760 : 1.0, 764 : 1.0,
                767 : 1.0, 779 : 1.0,
                865 : 1.0, 885 : 1.0,
                900 : 1.0, 940 : 1.0,
                1020: 1.0, 1375: 1.0,
                1610: 1.0, 2250: 1.0,
                }


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

        self.calib = {
                443 : 1.0,
                490 : 1.0,
                560 : 1.0,
                665 : 1.0,
                705 : 1.0,
                740 : 1.0,
                783 : 1.0,
                842 : 1.0,
                865 : 1.0,
                945 : 1.0,
                1375: 1.0,
                1610: 1.0,
                2190: 1.0,
                }

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

        self.lut_file = join(self.dir_base, 'LUTS/VIIRS/LUT.hdf')

        self.bands_corr = [410,443,486,551,671,745,862]
        self.bands_oc   = [410,443,486,551,671,745,862]
        self.bands_rw   = [410,443,486,551,671,745,862]
        self.bands_lut  = [410,443,486,551,671,745,862,1238,1601,2257]

        self.band_cloudmask = 862

        self.central_wavelength = dict(map(
            lambda x: (x, float(x)),
            [410,443,486,551,671,745,862,1238,1601,2257]))

        self.calib = {  # vicarious calibration R2014.0
                410: 0.9631,
                443: 1.0043,
                486: 1.0085,
                551: 0.9765,
                671: 1.0204,
                745: 1.0434,
                862: 1.0   ,
                1238:1.0   ,
                }

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


    def defaults_seawifs(self):
        self.bands_corr = [412,443,490,510,555,670,    865]
        self.bands_oc   = [412,443,490,510,555,670,    865]
        self.bands_rw   = [412,443,490,510,555,670,765,865]

        self.bands_lut  = [412,443,490,510,555,670,765,865]

        self.band_cloudmask = 865

        self.central_wavelength = dict(map(
            lambda x: (x, float(x)),
            [412,443,490,510,555,670,765,865]))

        self.calib = {
                412: 1.0,
                443: 1.0,
                490: 1.0,
                510: 1.0,
                555: 1.0,
                670: 1.0,
                765: 1.0,
                865: 1.0,
                }
        self.K_OZ = {  # from SeaDAS
                412: 4.114E-04,
                443: 3.162E-03,
                490: 2.346E-02,
                510: 4.094E-02,
                555: 9.568E-02,
                670: 4.649E-02,
                765: 8.141E-03,
                865: 3.331E-03,
                }
        self.K_NO2 = {  # from SeaDAS
                412: 6.004E-19,
                443: 4.963E-19,
                490: 2.746E-19,
                510: 2.081E-19,
                555: 9.411E-20,
                670: 9.234E-21,
                765: 1.078E-21,
                865: 1.942E-21,
                }

        self.lut_file = join(self.dir_base, 'LUTS/SEAWIFS/LUT.hdf')

    def defaults_modis(self):
        self.bands_corr = [412,443,    488,531,547,        667,678,748,    869,    ]
        self.bands_oc   = [412,443,    488,531,547,        667,678,748,    869,    ]
        self.bands_rw   = [412,443,    488,531,547,        667,678,748,    869,    ]
        self.bands_lut  = [412,443,469,488,531,547,555,645,667,678,748,858,869,1240]

        self.band_cloudmask = 869

        self.central_wavelength = dict(map(
            lambda x: (x, float(x)),
            [412,443,469,488,531,547,555,645,667,678,748,858,869,1240]))

        self.calib = {  # OC_CCI VICARIOUS CALIBRATION (VIC2, ERA-INTERIM)
                412 : 0.995,
                443 : 1.001,
                469 : 1.0  ,
                488 : 1.000,
                531 : 1.001,
                547 : 1.000,
                555 : 1.0  ,
                645 : 1.0  ,
                667 : 0.992,
                678 : 1.0  ,
                748 : 1.0  ,
                858 : 1.0  ,
                869 : 1.0  ,
                1240: 1.0  ,
                }
        self.K_OZ = {  # from SeaDAS
                412 :1.987E-03,
                443 :3.189E-03,
                469 :8.745E-03,
                488 :2.032E-02,
                531 :6.838E-02,
                547 :8.622E-02,
                555 :9.553E-02,
                645 :7.382E-02,
                667 :4.890E-02,
                678 :3.787E-02,
                748 :1.235E-02,
                858 :2.347E-03,
                869 :1.936E-03,
                1240:0.000E+00,
                }
        self.K_NO2 = {  # from SeaDAS
                412 :5.814E-19,
                443 :4.985E-19,
                469 :3.938E-19,
                488 :2.878E-19,
                531 :1.525E-19,
                547 :1.194E-19,
                555 :9.445E-20,
                645 :1.382E-20,
                667 :7.065E-21,
                678 :8.304E-21,
                748 :2.157E-21,
                858 :6.212E-23,
                869 :7.872E-23,
                1240:0.000E+00,
                }

        self.lut_file = join(self.dir_base, 'LUTS/MODIS/LUTB.hdf')


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


    def finalize(self):

        if hasattr(self.weights_corr, '__call__'):
            self.weights_corr = map(self.weights_corr, self.bands_corr)

        if hasattr(self.weights_oc, '__call__'):
            self.weights_oc = map(self.weights_oc, self.bands_oc)

        # number of terms in the model
        self.Ncoef = self.atm_model.count(',')+1


