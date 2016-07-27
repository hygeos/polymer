import numpy as np
cimport numpy as np

from os.path import join

from clut cimport CLUT
from libc.math cimport exp, M_PI, isnan, log
from warnings import warn



cdef class WaterModel:
    '''
    Base class for water reflectance models
    '''
    cdef init(self, float[:] wav, float sza, float vza, float raa):
        raise Exception('WaterModel.init(...) shall be implemented')

    cdef float[:] calc_rho(self, float[:] x):
        raise Exception('WaterModel.calc_rho(...) shall be implemented')


cdef class ParkRuddick(WaterModel):

    cdef float[:] wav  # wavelength (nm)
    cdef float[:] Rw   # water reflectance (0+)
    cdef float[:] bw   # pure water scattering coefficient
    cdef float[:] aw   # pure water absorption coefficient
    cdef float[:] a_bric    # absorption parameters A and B (Bricaud)
    cdef float[:] b_bric    # absorption parameters A and B (Bricaud)
    cdef float[:] a_star    # mineral absorption coefficient

    cdef CLUT BW
    cdef CLUT AW_POPEFRY
    cdef CLUT AW_PALMERW
    cdef CLUT GI_PR  # gi coefficients
    cdef CLUT GII_PR  # pre-interpolated gi coefficients (for one pixel)
    cdef CLUT AB_BRIC
    cdef CLUT RAMAN
    cdef CLUT ASTAR
    cdef float bw500
    cdef float a700
    cdef float mus
    cdef int Nwav
    cdef int debug
    cdef int alt_gamma_bb
    cdef int min_abs
    cdef float[:] atot, bbtot, btot, aphy, aCDM, aNAP
    cdef float gamma, SPM
    cdef object out_type

    cdef int[:] index  # multi-purpose vector

    def __init__(self, directory, alt_gamma_bb=False, min_abs=False, debug=False):

        self.Rw = None
        self.index = np.zeros(2, dtype='int32')
        self.debug = debug
        self.alt_gamma_bb = alt_gamma_bb
        self.min_abs = min_abs  # activate mineral absorption

        self.read_iop(directory)
        self.read_gi(directory)

    def read_iop(self, directory):

        #
        # read water scattering coefficient
        #
        data_bw = np.genfromtxt(join(directory, 'morel_buiteveld_bsw.txt'), skip_header=1)
        self.BW = CLUT(data_bw[:,1], axes=[data_bw[:,0]])
        assert data_bw[-1,0] == 500.
        self.bw500 = data_bw[-1,1]

        #
        # read pure water absorption coefficient
        #
        # Pope&Fry
        data = np.genfromtxt(join(directory, 'pope97.dat'), skip_header=6)
        data[:,1] *= 100 #  convert from cm-1 to m-1
        self.AW_POPEFRY = CLUT(data[:,1], axes=[data[:,0]])
        # Palmer&Williams
        data = np.genfromtxt(join(directory, 'palmer74.dat'), skip_header=5)
        data[:,1] *= 100 #  convert from cm-1 to m-1
        self.AW_PALMERW = CLUT(data[:,1], axes=[data[:,0]])

        #
        # read phytoplankton absorption
        #
        ap_bricaud = np.loadtxt(join(directory, 'aph_bricaud_1995.txt'), delimiter=',')
        lambda_bric95 = ap_bricaud[:, 0]
        self.AB_BRIC = CLUT(ap_bricaud[:,1:], axes=[lambda_bric95, None])
        assert lambda_bric95[-1] == 700
        self.a700 = ap_bricaud[-1,1]

        #
        # read mineral absorption
        #
        astar_ = np.genfromtxt(join(directory, 'astarmin_average_2015_SLSTR.txt'), comments='%')
        self.ASTAR = CLUT(astar_[:-1,1],
                          axes=[astar_[:-1,0]])

        #
        # read Raman correction
        #
        raman = np.genfromtxt(join(directory, 'raman_westberry13.txt'), comments='#')
        # (wl, chl)
        self.RAMAN = CLUT(raman[:,1:], axes=[raman[:,0],
                [0.01,0.02,0.03,0.04,0.07,0.1,0.2,0.3,0.5,0.7,1.,2.,5.]])

    def read_gi(self, directory):
        '''
        read gi coefficients
        '''

        fp = open(join(directory, 'AboveRrs_gCoef_w5.dat'))

        # read first lines
        fp.readline() # first line useless
        line = fp.readline()
        gb = list(map(float, line[:line.find('=')].split()))
        line = fp.readline()
        th0 = list(map(float, line[:line.find('=')].split()))
        line = fp.readline()
        th = list(map(float, line[:line.find('=')].split()))
        line = fp.readline()
        dphi = list(map(float, line[:line.find('=')].split()))

        # initialize output array
        ngb = len(gb)
        nth0 = len(th0)
        nth = len(th)
        ndphi = len(dphi)
        gi = np.zeros((ngb, 4, nth0, nth, ndphi), dtype='float32') - 999.

        # read blocks
        for (ith0, ith, idphi) in np.ndindex(nth0, nth, ndphi):
            fp.readline()

            # read next 8 lines
            block = ''
            for i in xrange(8): block += fp.readline()

            # format block and fill array slice
            block = np.array(list(map(float, block.split()))).reshape((ngb, 4))
            gi[:,:,ith0, ith, idphi] = block[:,:]

        fp.close()

        self.GI_PR = CLUT(gi, axes=[gb, None, th0, th, dphi])

        # initialize pre-interpolated gi coefficients
        # (empty)
        self.GII_PR = CLUT(np.zeros((ngb, 4)), axes=[gb, None])


    cdef init(self, float[:] wav, float sza, float vza, float raa):
        '''
        initialize the model parameters for current pixel
        '''
        cdef int i, j
        cdef int ret
        self.wav = wav
        self.Nwav = len(wav)
        self.mus = np.cos(sza*np.pi/180.)

        #
        # array initialization (unique)
        #
        if self.Rw is None:
            self.Rw = np.zeros(len(wav), dtype='float32') + np.NaN
            self.bw = np.zeros(len(wav), dtype='float32') + np.NaN
            self.aw = np.zeros(len(wav), dtype='float32') + np.NaN
            self.a_bric = np.zeros(len(wav), dtype='float32') + np.NaN
            self.b_bric = np.zeros(len(wav), dtype='float32') + np.NaN
            self.a_star = np.zeros(len(wav), dtype='float32') + np.NaN

            if self.debug:
                self.atot = np.zeros(len(wav), dtype='float32') + np.NaN
                self.bbtot = np.zeros(len(wav), dtype='float32') + np.NaN
                self.btot = np.zeros(len(wav), dtype='float32') + np.NaN
                self.aCDM = np.zeros(len(wav), dtype='float32') + np.NaN
                self.aNAP = np.zeros(len(wav), dtype='float32') + np.NaN
                self.aphy = np.zeros(len(wav), dtype='float32') + np.NaN
        elif len(wav) != len(self.Rw):
            raise Exception('Invalid length of wav')

        #
        # interpolate scattering coefficient
        #
        for i, w in enumerate(wav):
            ret = self.BW.lookup(0, w)
            if ret > 0:
                self.bw[i] = self.bw500 * (w/500.)**-4.
            elif ret < 0:
                raise Exception('Error in BW lookup')
            else:
                self.bw[i] = self.BW.interp()

        #
        # interpolate absorption coefficients
        #
        for i, w in enumerate(wav):

            ret = self.AW_POPEFRY.lookup(0, w)
            if ret < 0:
                raise Exception('Error in AW_POPEFRY lookup')
            elif ret > 0:
                if self.AW_PALMERW.lookup(0, w) != 0:
                    raise Exception('Error in AW_PALMERW lookup')
                self.aw[i] = self.AW_PALMERW.interp()
            else:
                self.aw[i] = self.AW_POPEFRY.interp()

            ret = self.AB_BRIC.lookup(0, w)
            if (ret < 0) and (w > 399.) and (w <= 400):
                ret = self.AB_BRIC.lookup(0, 400.)

            if ret == 0:
                # success
                self.AB_BRIC.index(1, 0)
                self.a_bric[i] = self.AB_BRIC.interp()
                self.AB_BRIC.index(1, 1)
                self.b_bric[i] = self.AB_BRIC.interp()
            elif ret > 0:  # above axis
                self.a_bric[i] = self.a700 * (w/700.)**(-80.)
                self.b_bric[i] = 0
            else:
                raise Exception('Error in AB_BRIC lookup (lambda={})'.format(w))

            ret = self.ASTAR.lookup(0, w)
            if ret != 0:
                raise Exception('Error on A_STAR lookup (wavelength={})'.format(w))
            else:
                self.a_star[i] = self.ASTAR.interp()


        #
        # empty the pre-interpolated gi coefficients
        #
        for i in range(self.GII_PR.shape[0]):
            for j in range(self.GII_PR.shape[1]):
                self.index[0] = i
                self.index[1] = j
                self.GII_PR.set(np.NaN, self.index)
        #
        # lookup the ths, thv and phi axes
        #
        ret = self.GI_PR.lookup(2, sza)
        if ret != 0: raise Exception('Error in GI_PR sza lookup')
        ret = self.GI_PR.lookup(3, vza)
        if ret != 0: raise Exception('Error in GI_PR vza lookup')
        ret = self.GI_PR.lookup(4, raa)
        if ret != 0: raise Exception('Error in GI_PR raa lookup')


    cdef float[:] calc_rho(self, float[:] x):
        '''
        calculate the reflectance above the water surface at all bands,
        for a parameter vector x
        '''

        # x is [logchl, logfb, logfa] or shorter
        cdef int N = len(x)
        cdef float fa, fb
        cdef float logchl = x[0]
        cdef float chl = 10**logchl
        cdef float bbw, gamma, bp550, bbp550, bb, bbp, bbp650
        cdef float SPM
        cdef float aw, aphy, aCDM, a, aCDM443, S, aNAP
        cdef float lam
        cdef float gammab, omegab, omegapow, gi
        cdef int i, j
        cdef int ret
        cdef int igb
        cdef float rho

        if N >= 2:
            fb = 10**x[1]
        else:
            fb = 1.

        if N >= 3:
            fa = 10**x[2]
        else:
            fa = 1.

        #
        # wavelength-independent parameters
        #

        # phytoplankton scattering
        bp550 = 0.416 * (chl**0.766) * fb
        if (logchl < 2):
            bbp550 = (0.002 + 0.01*(0.5 - 0.25*logchl)) * bp550
        else:
            bbp550 = (0.002 + 0.01*(0.5 - 0.25*2.    )) * bp550

        # spectral dependency
        if self.alt_gamma_bb:
            # alternate version (H. Loisel), improved in turbid waters
            # phytoplankton + other particles
            # dependency function of bbp550
            # - high values: personal comm. Loisel
            # - low bbp values: slope from Antoine LO (2011)
            #   offset adjusted for tangency to Loisel
            if bbp550 < 0.00227:
                gamma = -0.845 * log(bbp550) - 3.13
            else:
                gamma = 0.156 * bbp550**(-0.42)

            if gamma > 4:
                gamma = 4.
        else:
            # phytoplankton + other particles
            # spectral dependency (from Antoine LO (2011))
            gamma = -0.733 * log(chl) + 1.499   # /!\ log is ln
            if gamma < 0: gamma = 0

        bbp650 = bbp550 * (650./550.)**(-gamma)
        SPM = 100.*bbp650  # Neukermans, log10(bbp) = 1.03*log10(SPM)

        # CDM absorption central value
        # from Bricaud et al GBC, 2012 (data from nov 2007)
        aCDM443 = fa * 0.069 * (chl**1.070)   # Bricaud et al, GBC, 2012

        S = 0.00262*(aCDM443**(-0.448))
        if (S > 0.025): S=0.025
        if (S < 0.011): S=0.011

        # wavelength loop
        for i in range(self.Nwav):
            lam = self.wav[i]

            #
            # 1) scattering parameters
            #

            # pure water scattering
            bbw = 0.5*self.bw[i]

            bbp = bbp550 * (lam/550)**(-gamma)

            bb = bbw + bbp


            #
            # 2) absorption parameters
            #

            # pure water absorption
            aw = self.aw[i]

            # phytoplankton absorption
            aphy = self.a_bric[i] * chl**(1-self.b_bric[i])

            aCDM = aCDM443 * exp(-S*(lam - 443))

            # mineral absorption
            if self.min_abs:
                aNAP = self.a_star[i]*SPM
            else:
                aNAP = 0.

            # total absorption
            a = aw + aphy + aCDM + aNAP

            omegab = bb/(a + bb)
            gammab = bbp/bb

            omegapow = 1.
            rho = 0.
            ret = self.GII_PR.lookup(0, gammab)
            # if ret != 0:  # FIXME: test return values
                # raise Exception('GII_PR lookup error: gammab={}'.format(gammab))

            # pre-interpolation
            for igb in range(self.GII_PR._inf[0], self.GII_PR._inf[0]+2):
                if igb >= self.GII_PR.shape[0]:
                    continue
                self.index[0] = igb
                self.index[1] = 0
                if isnan(self.GII_PR.get(self.index)):
                    self.GI_PR.index(0, igb)
                    # NOTE: axes ths, thv and phi have already been lookedup on init()
                    for j in range(4):
                        self.GI_PR.index(1, j)
                        self.index[1] = j
                        self.GII_PR.set(self.GI_PR.interp(), self.index)


            for j in range(4):
                self.GII_PR.index(1, j)

                omegapow *= omegab

                gi = self.GII_PR.interp()

                rho += gi * omegapow

            rho *= M_PI # conversion remote sensing reflectance -> reflectance

            # raman correction
            # TODO: pre-interpolate RAMAN in lam?
            ret = self.RAMAN.lookup(0, lam)
            if (ret < 0) and (lam > 399.) and (lam <= 400):
                ret = self.RAMAN.lookup(0, 400.)
            if ret < 0:
                raise Exception('Error in lookup for RAMAN (lambda={})'.format(lam))
            ret = self.RAMAN.lookup(1, chl)  # clip both ends
            rho *= 1. + (self.RAMAN.interp()*self.mus/0.866)

            self.Rw[i] = rho

            if self.debug:
                self.atot[i] = a
                self.bbtot[i] = bb
                self.btot[i] = self.bw[i] + bp550*(lam/550.)**(-gamma)
                self.aphy[i] = aphy
                self.aCDM[i] = aCDM
                self.aNAP[i] = aNAP
                self.gamma = gamma
                self.SPM = SPM


        return self.Rw

    def calc(self, w, logchl, logfb=0., logfa=0., sza=0., vza=0., raa=0.):
        '''
        water reflectance calculation (python interface)
        w: wavelength (nm) [float, list or array]
        returns above-water reflectance as float, list or array
        '''
        if isinstance(w, np.ndarray):
            self.out_type = np.array
            wav = w.astype('float32')
        elif isinstance(w, list):
            self.out_type = list
            wav = np.array(w, dtype='float32')
        else: # float
            self.out_type = lambda x: x[0]
            wav = np.array([w], dtype='float32')

        self.init(wav.astype('float32'), float(sza), float(vza), float(raa))
        params = np.zeros(3, dtype='float32')
        params[0] = logchl
        params[1] = logfb
        params[2] = logfa
        R = self.calc_rho(params)
        return self.out_type(R)

    def iops(self):
        '''
        returns the IOPs for the previously calculated spectrum
        '''
        assert self.debug

        return {
                'bw':   self.out_type(self.bw),
                'aw':   self.out_type(self.aw),
                'atot': self.out_type(self.atot),
                'bbtot':self.out_type(self.bbtot),
                'btot': self.out_type(self.btot),
                'aphy': self.out_type(self.aphy),
                'aCDM': self.out_type(self.aCDM),
                'aNAP': self.out_type(self.aNAP),
                'gamma': self.gamma,
                'SPM': self.SPM,
                }


cdef class MorelMaritorena(WaterModel):

    cdef float[:] wav  # wavelength (nm)
    cdef int Nwav
    cdef CLUT Kw_tab, bw_tab, Chi_tab, e_tab, simspec
    cdef float[:] Kw_i
    cdef float[:] Chi_i
    cdef float[:] e_i
    cdef float[:] bw_i
    cdef float[:] Rw
    cdef float[:] simspec_i
    cdef float lam_join
    cdef initialized
    cdef object out_type

    cdef int debug
    cdef float[:] bw, atot, bbtot

    def __init__(self, debug=False):

        warn('f/Q is not implemented')
        self.debug = debug

        self.Kw_tab = CLUT(np.array([
                    0.02710, 0.02380, 0.02160, 0.01880, 0.01770, 0.01595, 0.01510, 0.01376, 0.01271, 0.01208,
                    0.01042, 0.00890, 0.00812, 0.00765, 0.00758, 0.00768, 0.00770, 0.00792, 0.00885, 0.00990,
                    0.01148, 0.01182, 0.01188, 0.01211, 0.01251, 0.01320, 0.01444, 0.01526, 0.01660, 0.01885,
                    0.02188, 0.02701, 0.03385, 0.04090, 0.04214, 0.04287, 0.04454, 0.04630, 0.04846, 0.05212,
                    0.05746, 0.06053, 0.06280, 0.06507, 0.07034, 0.07801, 0.09038, 0.11076, 0.13584, 0.16792,
                    0.22310, 0.25838, 0.26506, 0.26843, 0.27612, 0.28400, 0.29218, 0.30176, 0.31134, 0.32553,
                    0.34052, 0.37150, 0.41048, 0.42947, 0.43946, 0.44844, 0.46543, 0.48642, 0.51640, 0.55939,
                    0.62438], dtype='float32'),
                    axes=[np.arange(350, 701, 5.)]
                )
        self.Chi_tab = CLUT(np.array([
                    0.15300, 0.14900, 0.14400, 0.14000, 0.13600, 0.13100, 0.12700, 0.12300, 0.11900, 0.11800,
                    0.11748, 0.12066, 0.12259, 0.12326, 0.12269, 0.12086, 0.11779, 0.11372, 0.10963, 0.10560,
                    0.10165, 0.09776, 0.09393, 0.09018, 0.08649, 0.08287, 0.07932, 0.07584, 0.07242, 0.06907,
                    0.06579, 0.06257, 0.05943, 0.05635, 0.05341, 0.05072, 0.04829, 0.04611, 0.04419, 0.04253,
                    0.04111, 0.03996, 0.03900, 0.03750, 0.03600, 0.03400, 0.03300, 0.03280, 0.03250, 0.03300,
                    0.03400, 0.03500, 0.03600, 0.03750, 0.03850, 0.04000, 0.04200, 0.04300, 0.04400, 0.04450,
                    0.04500, 0.04600, 0.04750, 0.04900, 0.05150, 0.05200, 0.05050, 0.04400, 0.03900, 0.03400,
                    0.03000 ], dtype='float32'),
                    axes=[np.arange(350, 701, 5.)]
                )
        self.e_tab = CLUT(np.array([
                    0.77800, 0.76700, 0.75600, 0.73700, 0.72000, 0.70000, 0.68500, 0.67300, 0.67000, 0.66000,
                    0.64358, 0.64776, 0.65175, 0.65555, 0.65917, 0.66259, 0.66583, 0.66889, 0.67175, 0.67443,
                    0.67692, 0.67923, 0.68134, 0.68327, 0.68501, 0.68657, 0.68794, 0.68903, 0.68955, 0.68947,
                    0.68880, 0.68753, 0.68567, 0.68320, 0.68015, 0.67649, 0.67224, 0.66739, 0.66195, 0.65591,
                    0.64927, 0.64204, 0.64000, 0.63000, 0.62300, 0.61500, 0.61000, 0.61400, 0.61800, 0.62200,
                    0.62600, 0.63000, 0.63400, 0.63800, 0.64200, 0.64700, 0.65300, 0.65800, 0.66300, 0.66700,
                    0.67200, 0.67700, 0.68200, 0.68700, 0.69500, 0.69700, 0.69300, 0.66500, 0.64000, 0.62000,
                    0.60000], dtype='float32'),
                    axes=[np.arange(350, 701, 5.)]
                )
        self.bw_tab = CLUT(np.array([
                    0.0121, 0.0113, 0.0107, 0.0099, 0.0095, 0.0089, 0.0085, 0.0081, 0.0077, 0.0072,
                    0.0069, 0.0065, 0.0062, 0.0059, 0.0056, 0.0054, 0.0051, 0.0049, 0.0047, 0.0044,
                    0.0043, 0.0040, 0.0039, 0.0037, 0.0035, 0.0034, 0.0033, 0.0031, 0.0030, 0.0029,
                    0.0027] + map(lambda x: 0.0027*((x/500.)**-4.), np.arange(505, 701, 5.)),
                    dtype='float32'),
                    axes=[np.arange(350, 701, 5.)]
                )
        self.simspec = CLUT(np.array([
                4.953, 4.858, 4.734, 4.586, 4.432, 4.293, 4.177, 4.082, 4.017, 3.976, 3.949,
                3.939, 3.937, 3.974, 4.016, 4.046, 4.061, 4.015, 3.948, 3.862, 3.757, 3.621,
                3.466, 3.297, 3.118, 2.931, 2.754, 2.560, 2.350, 2.144, 1.937, 1.736, 1.551,
                1.393, 1.273, 1.185, 1.123, 1.080, 1.053, 1.032, 1.013, 1.001, 0.994, 1.012,
                1.029, 1.033, 1.016, 0.985, 0.971, 0.968, 0.972, 0.985, 1.000, 1.015, 1.029,
                1.046, 1.067, 1.087, 1.108, 1.127, 1.145, 1.159, 1.169, 1.173, 1.175, 1.171,
                1.159, 1.138, 1.098, 1.043, 0.980, 0.912, 0.846, 0.788, 0.742, 0.707, 0.678,
                0.658, 0.640, 0.627, 0.616, 0.603, 0.592, 0.579, 0.564, 0.553, 0.544, 0.534,
                0.523, 0.512, 0.501, 0.488, 0.476, 0.465, 0.454, 0.440, 0.431, 0.425, 0.419,
                0.413, 0.409], dtype='float32'),
                axes=[np.arange(650, 901, 2.5)])

        self.initialized = 0
        self.lam_join = 690.


    cdef init(self, float[:] wav, float sza, float vza, float raa):
        self.Nwav = len(wav)
        cdef int i
        cdef float lam

        if not self.initialized:
            self.wav   = np.zeros(self.Nwav+1, dtype='float32')
            self.Kw_i  = np.zeros(self.Nwav+1, dtype='float32')
            self.Chi_i = np.zeros(self.Nwav+1, dtype='float32')
            self.e_i   = np.zeros(self.Nwav+1, dtype='float32')
            self.bw_i  = np.zeros(self.Nwav+1, dtype='float32')
            self.simspec_i = np.zeros(self.Nwav+1, dtype='float32')
            self.Rw = np.zeros(self.Nwav, dtype='float32')
            self.initialized = 1

            if self.debug:
                self.bw  = np.zeros(self.Nwav, dtype='float32')
                self.atot  = np.zeros(self.Nwav, dtype='float32')
                self.bbtot  = np.zeros(self.Nwav, dtype='float32')
        elif len(wav) != len(self.Rw):
            raise Exception('Invalid length of wav')

        for i in range(self.Nwav):
            self.wav[i] = wav[i]
        self.wav[self.Nwav] = self.lam_join

        for i in range(self.Nwav+1):
            # FIXME: this loop can be made faster
            lam = self.wav[i]

            self.Kw_tab.lookup(0, lam)
            self.Kw_i[i] = self.Kw_tab.interp()

            self.Chi_tab.lookup(0, lam)
            self.Chi_i[i] = self.Chi_tab.interp()

            self.e_tab.lookup(0, lam)
            self.e_i[i] = self.e_tab.interp()

            self.bw_tab.lookup(0, lam)
            self.bw_i[i] = self.bw_tab.interp()

            if self.simspec.lookup(0, lam) == 0:
                self.simspec_i[i] = self.simspec.interp()


    cdef float[:] calc_rho(self, float[:] x):
        cdef float rw_join

        # wavelength loop: visible
        for i in range(self.Nwav):
            self.Rw[i] = self.calc_rho_vis(i, x)

        # towards NIR
        rw_join = self.calc_rho_vis(self.Nwav, x)

        # wavelength loop: NIR
        for i in range(self.Nwav):
            if self.Rw[i] < 0:
                self.Rw[i] = self.calc_rho_nir(i, rw_join)

        return self.Rw

    cdef float calc_rho_vis(self, int i, float[:] x):
        '''
        reflectance calculation for visible bands
        return -1 if invalid wavelength
        '''

        cdef float Kw
        cdef float Chi, e, lam
        cdef float Kbio
        cdef float logchl = x[0]
        cdef float bbs0 = x[1]
        cdef float bbs, bp550, bbp, bb, bw
        cdef float ays0, a_ys
        cdef float Sys, Snap
        cdef float v
        cdef int j
        cdef float bbs_spec
        ays0 = 0.

        Kw  = self.Kw_i[i]
        Chi = self.Chi_i[i]
        e   = self.e_i[i]
        bw  = self.bw_i[i]
        lam = self.wav[i]

        if lam > 700:
            return -1

        Kbio = Chi * (10.**(e * logchl))

        if (logchl < 0.301029995664):
            v = 0.5 * (logchl - 0.3) # // 0.301029995664 == log10(2)
        else:
            v = 0;

        bbs_spec = -1.
        bbs = bbs0*((lam/550.)**bbs_spec)

        bp550 = 0.416 * (10.**(0.766 * logchl))
        bbp = (0.002 + 0.01*(0.5 - 0.25*logchl)*((lam/550.)**v)) * bp550
        bb = 0.5 * bw + bbp + bbs

        # absorption by yellow substances
        Sys = 0.014
        a_ys = ays0*exp(-Sys*(lam - 410.))

        Kd = Kw + Kbio + bbs + a_ys

        # absorption by detritus (non-algal particles)
        Snap = 0.011

        # Correction of detrital absorption for clear waters
        # (see Morel et al, 2007, "Optical properties of the 'clearest' natural waters")
        a_nap = - exp(-7.55*(10**(logchl*0.08)))*exp(-Snap*(lam - 420.))


        # iterations
        u = 0.75
        for j in range(3):
            a = u * Kd
            R = 0.33 * bb / (a + a_ys + a_nap)

            u = 0.90 * 1/(1 + 2.25*R) * (1 - R)

        # TODO:
        # interpolate f/Q coefficients

        if self.debug:
            self.bw[i] = bw
            self.bbtot[i] = bb
            self.atot[i] = a + a_ys + a_nap

        return R * 0.544

    cdef float calc_rho_nir(self, int i, float rw_join):
        return self.simspec_i[i]*rw_join/self.simspec_i[self.Nwav]

    def calc(self, w, logchl, bbs=0., sza=0., vza=0., raa=0.):
        '''
        water reflectance calculation (python interface)
        w: wavelength (nm) [float, list or array]
        returns above-water reflectance as float, list or array
        '''
        if isinstance(w, np.ndarray):
            self.out_type = np.array
            wav = w.astype('float32')
        elif isinstance(w, list):
            self.out_type = list
            wav = np.array(w, dtype='float32')
        else: # float
            self.out_type = lambda x: x[0]
            wav = np.array([w], dtype='float32')

        self.init(wav, float(sza), float(vza), float(raa))
        params = np.zeros(2, dtype='float32')
        params[0] = logchl
        params[1] = bbs
        R = self.calc_rho(params)
        return self.out_type(R)

    def iops(self):
        '''
        returns the IOPs for the previously calculated spectrum
        '''
        assert self.debug

        return {
                'bw':    self.out_type(self.bw),
                'atot':  self.out_type(self.atot),
                'bbtot': self.out_type(self.bbtot),
                }


def test():
    pr = ParkRuddick('auxdata/common/')
    pr.init(np.linspace(401, 800, 100, dtype='float32'), 0, 0, 0)
    a = pr.calc_rho(np.array([0., 0.], dtype='float32'))
    # print np.array(a)
