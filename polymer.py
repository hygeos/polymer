#!/usr/bin/env python
# encoding: utf-8


from __future__ import print_function, division

import numpy as np
from luts import read_mlut_hdf, Idx
from utils import stdNxN
from common import BITMASK_INVALID, L2FLAGS
from pyhdf.SD import SD
from multiprocessing import Pool
from datetime import datetime
from utils import coeff_sun_earth_distance
from params import Params
from level1 import Level1
from level2 import Level2

from polymer_main import PolymerMinimizer
from water import ParkRuddick, MorelMaritorena

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
                            alt_gamma_bb=self.params.alt_gamma_bb,
                            min_abs=self.params.min_abs)
        elif self.params.water_model == 'MM01':
            watermodel = MorelMaritorena()
        else:
            raise Exception('Invalid water model "{}"'.format(self.params.water_model))

        return PolymerMinimizer(watermodel, self.params)


    def convert_reflectance(self, block):

        if self.params.partial >= 5:
            return

        if hasattr(block, 'Rtoa'):
            return

        block.Rtoa = np.zeros(block.Ltoa.shape)+np.NaN

        coef = coeff_sun_earth_distance(block.jday)

        ok = (block.bitmask & BITMASK_INVALID) == 0

        if isinstance(coef, np.ndarray):
            coef = coef[ok]

        for i in xrange(block.nbands):

            block.Rtoa[ok,i] = block.Ltoa[ok,i]*np.pi/(block.mus[ok]*block.F0[ok,i]*coef)

    def apply_calib(self, block):
        '''
        Apply calibration coefficients on Rtoa
        '''
        if self.params.calib is None:
            return

        ok = (block.bitmask & BITMASK_INVALID) == 0
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

        no2_tropo = self.no2_tropo_data[imon,ilat,ilon]*1e15
        no2_strat = (self.no2_total_data[imon,ilat,ilon]
                     - self.no2_tropo_data[imon,ilat,ilon])*1e15

        # coordinates of current block in 90x180 grid
        ilat = (0.5*(90 - block.latitude)).astype('int')
        ilon = (0.5*(block.longitude)).astype('int')
        ilon[ilon<0] += 180
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

        ok = (block.bitmask & BITMASK_INVALID) == 0

        #
        # ozone correction
        #
        # make sure that ozone is in DU
        if (block.ozone[ok] < 50).any() or (block.ozone[ok] > 1000).any():
            raise Exception('Error, ozone is assumed in DU')

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
        ok = (block.bitmask & BITMASK_INVALID) == 0

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

        block.bitmask += L2FLAGS['CLOUD_BASE'] * cloudmask.astype('uint8')


    def rayleigh_correction(self, block):
        '''
        Rayleigh correction
        + transmission interpolation
        '''
        params = self.params
        mlut = self.mlut
        if params.partial >= 2:
            return

        block.Rprime = np.zeros(block.Rtoa.shape, dtype='float32')+np.NaN
        block.Rprime_noglint = np.zeros(block.Rtoa.shape, dtype='float32')+np.NaN
        block.Rmol = np.zeros(block.Rtoa.shape, dtype='float32')+np.NaN
        block.Rmolgli = np.zeros(block.Rtoa.shape, dtype='float32')+np.NaN
        block.Tmol = np.zeros(block.Rtoa.shape, dtype='float32')+np.NaN

        ok = (block.bitmask & BITMASK_INVALID) == 0

        for i in xrange(block.nbands):
            ilut = params.bands_lut.index(block.bands[i])

            wind = block.wind_speed[ok]
            wmax = np.amax(mlut.axis('dim_wind'))
            wind[wind > wmax] = wmax  # clip to max wind

            Rmolgli = mlut['Rmolgli'][
                    Idx(block.muv[ok]),
                    Idx(block.raa[ok]),
                    Idx(block.mus[ok]),
                    ilut, Idx(wind)]
            Rmol = mlut['Rmol'][
                    Idx(block.muv[ok]),
                    Idx(block.raa[ok]),
                    Idx(block.mus[ok]),
                    ilut]

            wl = block.wavelen[ok,i]
            wl0 = self.params.central_wavelength[block.bands[i]]

            # wavelength adjustment
            Rmolgli *= (wl/wl0)**(-4.)
            Rmol *= (wl/wl0)**(-4.)

            # adjustment for atmospheric pressure
            Rmolgli *= block.surf_press[ok]/1013.
            Rmol *= block.surf_press[ok]/1013.

            block.Rmolgli[ok,i] = Rmolgli
            block.Rmol[ok,i] = Rmol

            if self.params.glint_precorrection:
                block.Rprime[ok,i] = block.Rtoa_gc[ok,i] - Rmolgli
            else:
                block.Rprime[ok,i] = block.Rtoa_gc[ok,i] - Rmol

            block.Rprime_noglint[ok,i] = block.Rtoa_gc[ok,i] - Rmol

            # TODO: share axes indices
            # and across wavelengths
            block.Tmol[ok,i]  = mlut['Tmolgli'][Idx(block.mus[ok]),
                    ilut, Idx(wind)]
            block.Tmol[ok,i] *= mlut['Tmolgli'][Idx(block.muv[ok]),
                    ilut, Idx(wind)]

            # correction for atmospheric pressure
            taumol = 0.00877*((block.wavelen[ok,i]/1000.)**-4.05)
            block.Tmol[ok,i] *= np.exp(-taumol/2. * (block.surf_press[ok]/1013. - 1.) * block.air_mass[ok])


def process_block(args):
    '''
    Process one block of data
    '''

    (block, c, opt) = args

    if opt is None:
        opt = c.init_minimizer()

    c.convert_reflectance(block)

    c.apply_calib(block)

    c.gas_correction(block)

    c.cloudmask(block)

    c.rayleigh_correction(block)

    opt.minimize(block)

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
            print('Processing block', block)

        yield (block, c, opt)




def polymer(level1, level2, **kwargs):
    '''
    Polymer atmospheric correction
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
        ('hdf4, 'netcdf')
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
        N = 1: single thread
        N > 1: multiple threads
        N < 1: use as many threads as there are CPUs on local machine
    - dir_base: location of base directory to locate auxiliary data
    - calib: a dictionary for applying calibration coefficients
    - normalize: if True (default), apply normalization of the water reflectance at nadir-nadir

    RETURNS: the level2 instance
    '''

    t0 = datetime.now()
    print('Starting processing at {}'.format(t0))

    # initialize level1 and level2 instances
    with level2 as l2, level1 as l1:

        params = Params(l1.sensor, **kwargs)

        l2.init(l1)

        # initialize the block iterator
        if params.multiprocessing != 1:
            if params.multiprocessing <= 0:
                nproc = None  # use as many processes as there are CPUs
            else:
                nproc = params.multiprocessing
            block_iter = Pool(nproc).imap_unordered(process_block,
                    blockiterator(l1, params, True))
        else:
            block_iter = imap(process_block,
                    blockiterator(l1, params, False))

        # loop over the blocks
        for block in block_iter:
            l2.write(block)

        params.processing_duration = datetime.now()-t0
        l2.finish(params)

        print('Done in {}'.format(datetime.now()-t0))

        return l2


if __name__ == "__main__":

    from sys import argv

    if len(argv) != 3:
        print('Usage: polymer.py <level1> <level2>')
        print('       minimal code to run polymer on the command line')
        print('       NOTE: to pass additional parameters, it is advised')
        print('       to run it directly as a python function')
        exit(1)

    file_in = argv[1]
    file_out = argv[2]

    polymer(Level1(file_in), Level2(filename=file_out))

