#!/usr/bin/env python
# encoding: utf-8


from __future__ import print_function, division, absolute_import

import numpy as np
from polymer.luts import read_mlut_hdf, Idx
from polymer.utils import stdNxN, raiseflag
from polymer.common import L2FLAGS
from pyhdf.SD import SD
from multiprocessing import Pool
from datetime import datetime
from polymer.params import Params
from polymer.level1 import Level1
from polymer.level2 import Level2
from polymer.bodhaine import rod
from polymer.polymer_main import PolymerMinimizer
from polymer.water import ParkRuddick, MorelMaritorena
from warnings import warn

import sys
if sys.version_info[:2] >= (3, 0):
    xrange = range
    imap = map
else:  # python 2
    from itertools import imap



class InitCorr(object):
    '''
    Implementation of the initial corrections:
        * convert to reflectance
        * gaseous correction
        * cloud mask
        * Rayleigh correction
    '''
    def __init__(self, params):
        self.params = params

        # read the look-up table
        self.mlut = read_mlut_hdf(params.lut_file)


    def init_minimizer(self):
        '''
        Initialization of the minimizer class
        '''
        if self.params.water_model == 'PR05':
            watermodel = ParkRuddick(
                            self.params.dir_common,
                            bbopt=self.params.bbopt,
                            min_abs=self.params.min_abs,
                            absorption=self.params.absorption)
        elif self.params.water_model.startswith('MM01'):
            directional = {'MM01': False,
                           'MM01_FOQ': True}[self.params.water_model]
            watermodel = MorelMaritorena(self.params.dir_common, directional=directional)
        else:
            raise Exception('Invalid water model "{}"'.format(self.params.water_model))

        return PolymerMinimizer(watermodel, self.params)


    def preprocessing(self, block):

        #
        # filter pixels such that ths > 88° as EXCEPTION
        #
        raiseflag(block.bitmask,
                  L2FLAGS['EXCEPTION'],
                  block.sza > 88)
        raiseflag(block.bitmask,
                  L2FLAGS['EXCEPTION'],
                  np.isnan(block.ozone))

        #
        # apply external mask
        #
        if self.params.external_mask is not None:
            ox, oy = block.offset
            sx, sy = block.size
            raiseflag(block.bitmask,
                      L2FLAGS['EXTERNAL_MASK'],
                      self.params.external_mask[ox:ox+sx,
                                                oy:oy+sy] != 0
                      )



    def convert_reflectance(self, block):

        if self.params.partial >= 5:
            return

        if hasattr(block, 'Rtoa'):
            return

        block.Rtoa = np.zeros(block.Ltoa.shape)+np.NaN

        ok = (block.bitmask & self.params.BITMASK_INVALID) == 0

        for i in xrange(block.nbands):

            block.Rtoa[ok,i] = block.Ltoa[ok,i]*np.pi/(block.mus[ok]*block.F0[ok,i])

    def apply_calib(self, block):
        '''
        Apply calibration coefficients on Rtoa
        '''
        if self.params.calib is None:
            return

        ok = (block.bitmask & self.params.BITMASK_INVALID) == 0
        for i, b in enumerate(block.bands):
            block.Rtoa[ok,i] *= self.params.calib[b]


    def read_no2_data(self, month):
        '''
        read no2 data from month (1..12) or for all months if month < 0

        shape of arrays is (month, lat, lon)
        '''
        hdf1 = SD(self.params.no2_climatology)
        hdf2 = SD(self.params.no2_frac200m)

        if month < 0:
            months = range(1, 13)
            nmo = 12
        else:
            months = [month]
            nmo = 1

        self.no2_total_data = np.zeros((nmo,720,1440), dtype='float32')
        self.no2_tropo_data = np.zeros((nmo,720,1440), dtype='float32')
        self.no2_frac200m_data = np.zeros((90,180), dtype='float32')


        for i, m in enumerate(months):
            # read total and tropospheric no2 data
            self.no2_total_data[i,:,:] = hdf1.select('tot_no2_{:02d}'.format(m)).get()

            self.no2_tropo_data[i,:,:] = hdf1.select('trop_no2_{:02d}'.format(m)).get()

        # read fraction of tropospheric NO2 above 200mn
        self.no2_frac200m_data[:,:] = hdf2.select('f_no2_200m').get()

        hdf1.end()
        hdf2.end()


    def get_no2(self, block):
        '''
        returns no2_frac, no2_tropo, no2_strat at the pixels coordinates
        '''
        ok = (block.bitmask & self.params.BITMASK_INVALID) == 0

        # get month
        if isinstance(block.month, np.ndarray):
            mon = -1
            imon = block.month-1
        else:
            mon = block.month
            imon = 0

        try:
            self.no2_tropo_data
        except:
            self.read_no2_data(mon)

        # coordinates of current block in 1440x720 grid
        ilat = (4*(90 - block.latitude)).astype('int')
        ilon = (4*block.longitude).astype('int')
        ilon[ilon<0] += 4*360
        ilat[~ok] = 0
        ilon[~ok] = 0

        no2_tropo = self.no2_tropo_data[imon,ilat,ilon]*1e15
        no2_strat = (self.no2_total_data[imon,ilat,ilon]
                     - self.no2_tropo_data[imon,ilat,ilon])*1e15

        # coordinates of current block in 90x180 grid
        ilat = (0.5*(90 - block.latitude)).astype('int')
        ilon = (0.5*(block.longitude)).astype('int')
        ilon[ilon<0] += 180
        ilat[~ok] = 0
        ilon[~ok] = 0
        no2_frac = self.no2_frac200m_data[ilat,ilon]

        return no2_frac, no2_tropo, no2_strat


    def gas_correction(self, block):
        '''
        Correction for gaseous absorption (ozone and NO2)
        '''
        params = self.params
        if self.params.partial >= 4:
            return

        block.Rtoa_gc = np.zeros(block.Rtoa.shape, dtype='float32') + np.NaN
        nightpixel = block.sza >= 90

        ok = (block.bitmask & self.params.BITMASK_INVALID) == 0
        ok &= ~nightpixel
        raiseflag(block.bitmask, L2FLAGS['EXCEPTION'], nightpixel)

        #
        # ozone correction
        #
        # make sure that ozone is in DU
        ozone_warn = (block.ozone[ok] < 50) | (block.ozone[ok] > 1000)
        if ozone_warn.any():
            warn('ozone is assumed in DU ({})'.format(block.ozone[ok][ozone_warn]))

        # bands loop
        for i, b in enumerate(block.bands):

            tauO3 = params.K_OZ[b] * block.ozone[ok] * 1e-3  # convert from DU to cm*atm

            # ozone transmittance
            trans_O3 = np.exp(-tauO3 * block.air_mass[ok])

            block.Rtoa_gc[ok,i] = block.Rtoa[ok,i]/trans_O3

        #
        # NO2 correction
        #
        no2_frac, no2_tropo, no2_strat = self.get_no2(block)

        no2_tr200 = no2_frac * no2_tropo

        no2_tr200[no2_tr200<0] = 0

        for i, b in enumerate(block.bands):

            k_no2 = params.K_NO2[b]

            a_285 = k_no2 * (1.0 - 0.003*(285.0-294.0))
            a_225 = k_no2 * (1.0 - 0.003*(225.0-294.0))

            tau_to200 = a_285*no2_tr200 + a_225*no2_strat

            t_no2  = np.exp(-(tau_to200[ok]/block.mus[ok]))
            t_no2 *= np.exp(-(tau_to200[ok]/block.muv[ok]))

            block.Rtoa_gc[ok,i] /= t_no2

    def cloudmask(self, block):
        '''
        Polymer basic cloud mask
        '''
        if self.params.partial >= 3:
            return

        params = self.params
        ok = (block.bitmask & self.params.BITMASK_INVALID) == 0

        # flag out night pixels
        musmin = self.mlut.axis('dim_mu')[-1]
        nightpixel = block.mus <= musmin
        ok &= ~nightpixel
        raiseflag(block.bitmask, L2FLAGS['EXCEPTION'], nightpixel)

        inir_block = block.bands.index(params.band_cloudmask)
        inir_lut = params.bands_lut.index(params.band_cloudmask)

        block.Rnir = np.zeros(block.size, dtype='float32')
        block.Rnir[ok] = block.Rtoa_gc[:,:,inir_block][ok] - self.mlut['Rmol'][
                Idx(block.muv[ok]),
                Idx(block.raa[ok]),
                Idx(block.mus[ok]),
                inir_lut]

        if params.thres_Rcloud >= 0:
            cloudmask = block.Rnir > params.thres_Rcloud
        else:
            cloudmask = np.zeros_like(block.Rnir, dtype='uint8')
        if params.thres_Rcloud_std >= 0:
            cloudmask |= stdNxN(block.Rnir, 3, ok, fillv=0.) > params.thres_Rcloud_std

        raiseflag(block.bitmask, L2FLAGS['CLOUD_BASE'], cloudmask)


    def rayleigh_correction(self, block):
        '''
        Rayleigh correction
        + transmission interpolation
        '''
        params = self.params
        mlut = self.mlut
        if params.partial >= 2:
            return

        # flag high air mass
        # (air_mass > 5)
        raiseflag(block.bitmask, L2FLAGS['HIGH_AIR_MASS'],
                  block.air_mass > 5.)

        block.Rprime = np.zeros(block.Rtoa.shape, dtype='float32')+np.NaN
        block.Rprime_noglint = np.zeros(block.Rtoa.shape, dtype='float32')+np.NaN
        block.Rmol = np.zeros(block.Rtoa.shape, dtype='float32')+np.NaN
        block.Rmolgli = np.zeros(block.Rtoa.shape, dtype='float32')+np.NaN
        block.Tmol = np.zeros(block.Rtoa.shape, dtype='float32')+np.NaN

        ok = (block.bitmask & self.params.BITMASK_INVALID) == 0

        for i in xrange(block.nbands):
            wind = block.wind_speed[ok]
            wmax = np.amax(mlut.axis('dim_wind'))
            wind[wind > wmax] = wmax  # clip to max wind

            # calculate Rayleigh optical thickness
            # for current band
            wav = block.wavelen[ok, i]
            if not hasattr(block, 'tau_ray'):
                # default: calculate Rayleigh optical thickness on the fly
                tau_ray = rod(wav/1000., 400., 45.,
                              block.altitude[ok],
                              block.surf_press[ok])
            else:
                # if level1 provides its Rayleigh optical thickness, use it
                tau_ray = block.tau_ray[ok,i]

            Rmolgli = mlut['Rmolgli'][
                    Idx(block.muv[ok]),
                    Idx(block.raa[ok]),
                    Idx(block.mus[ok]),
                    Idx(tau_ray),
                    Idx(wind)]
            Rmol = mlut['Rmol'][
                    Idx(block.muv[ok]),
                    Idx(block.raa[ok]),
                    Idx(block.mus[ok]),
                    Idx(tau_ray)]

            block.Rmolgli[ok,i] = Rmolgli
            block.Rmol[ok,i] = Rmol

            if self.params.glint_precorrection:
                block.Rprime[ok,i] = block.Rtoa_gc[ok,i] - Rmolgli
            else:
                block.Rprime[ok,i] = block.Rtoa_gc[ok,i] - Rmol

            block.Rprime_noglint[ok,i] = block.Rtoa_gc[ok,i] - Rmol

            # TODO: share axes indices
            # and across wavelengths
            block.Tmol[ok,i]  = mlut['Tmolgli'][
                    Idx(block.mus[ok]),
                    Idx(tau_ray),
                    Idx(wind)]
            block.Tmol[ok,i] *= mlut['Tmolgli'][
                    Idx(block.muv[ok]),
                    Idx(tau_ray),
                    Idx(wind)]


    def set_attributes(self, block):
        flag_meanings = ', '.join(['{}:{}'.format(x[0], x[1])
                                   for x in sorted(L2FLAGS.items(),
                                                   key=lambda x: x[1])])
        block.attributes['bitmask'] = {
                'description': flag_meanings,
                'bitmask_reject': 'bitmask & {} != 0'.format(self.params.BITMASK_REJECT),
                }
        block.attributes['Rw'] = {
                'description': 'water reflectance (dimensionless ; fully normalized)'
                }
        block.attributes['logchl'] = {
                'description': 'log10 of the chl-a concentration in mg/m3',
                }


def process_block(args):
    '''
    Process one block of data
    '''

    (block, c, opt) = args

    if opt is None:
        opt = c.init_minimizer()

    c.preprocessing(block)

    c.convert_reflectance(block)

    c.apply_calib(block)

    c.gas_correction(block)

    c.cloudmask(block)

    c.rayleigh_correction(block)

    opt.minimize(block)

    c.set_attributes(block)

    return block


def blockiterator(level1, params, multi=False):
    '''
    Block iterator
    if multi (boolean), iterate in multiprocessing mode:
        The minimizer is created in the processing function instead of here,
        because as a cython class it is not picklable.
    Otherwise, the minimizer is created once.
    '''

    c = InitCorr(params)

    if multi:
        opt = None
    else:
        opt = c.init_minimizer()

    for block in level1.blocks(params.bands_read()):

        if params.verbose:
            print('Processing', block)

        yield (block, c, opt)




def run_atm_corr(level1, level2, **kwargs):
    '''
    Polymer atmospheric correction: main function
    https://www.osapublishing.org/oe/abstract.cfm?uri=oe-19-10-9783

    ARGUMENTS:

    level1: level1 instance
        Example:
        Level1('MER_RR__1PRACR20050501_092849_000026372036_00480_16566_0000.N1',
               sline=1500, eline=2000)   # all-purpose sensor-detecting level1
        Level1_NASA('A2004181120500.L1C', sensor='MODIS',
                    sline=1500, eline=2000, srow=100, erow=500)
        Level1_ASCII('extraction.csv', square=5, sensor='MERIS')

    level2: level2 instance
        argument fmt determines the level2 class to use
        ('hdf4, 'netcdf4')
        See appropriate Level2_* class for argument list (the additional
        arguments kwargs are passed directly to this class)

        Examples:
        # using the all-purpose level2
        Level2(fmt='hdf4', ext='.polymer.hdf', outdir='/data/')
        Level2(filename='/data/out.hdf', fmt='hdf4', compress=True)
        Level2('memory')   # store output in memory
        # using specific level2 classes
        Level2_NETCDF('out.nc', overwrite=True)

    Additional keyword arguments:
    see attributes defined in Params class
    Examples:
    - multiprocessing: number of threads to use for processing (int)
        N = 0: single thread (multiprocessing disactivated)
        N != 0: use multiple threads, with
        N < 0: use as many threads as there are CPUs on local machine
    - dir_base: location of base directory to locate auxiliary data
    - calib: a dictionary for applying calibration coefficients
    - normalize: select water reflectance normalization
           * no geometry nor wavelength normalization (0)
           * apply normalization of the water reflectance at nadir-nadir (1)
           * apply wavelength normalization for MERIS and OLCI (2)
           * apply both (3)

    RETURNS: the level2 instance
    '''

    t0 = datetime.now()
    if ('verbose' not in kwargs) or kwargs['verbose']:
        print('Starting processing at {}'.format(t0))

    # initialize level1 and level2 instances
    with level2 as l2, level1 as l1:

        params = Params(l1.sensor, **kwargs)
        params.preprocess(l1)

        l2.init(l1)

        # initialize the block iterator
        if params.multiprocessing != 0:
            if params.multiprocessing < 0:
                nproc = None  # use as many processes as there are CPUs
            else:
                nproc = params.multiprocessing
            pool = Pool(nproc)
            block_iter = pool.imap_unordered(process_block,
                    blockiterator(l1, params, True))
        else:
            block_iter = imap(process_block,
                    blockiterator(l1, params, False))

        # loop over the blocks
        for block in block_iter:
            l2.write(block)

        # finalize level2 file and include global attributes
        params.processing_duration = datetime.now()-t0
        params.update(**l1.attributes('%Y-%m-%d %H:%M:%S'))
        params.update(**l2.attributes())
        l2.finish(params)

        if params.multiprocessing != 0:
            pool.terminate()

        if params.verbose:
            print('Done in {}'.format(datetime.now()-t0))

        return l2

