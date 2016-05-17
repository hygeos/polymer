import numpy as np
cimport numpy as np

from os.path import join

from clut cimport CLUT
from libc.math cimport exp, M_PI, isnan



cdef class WaterModel:
    '''
    Base class for water reflectance models
    '''
    cdef init(self, float [:] wav, float sza, float vza, float raa):
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

    cdef CLUT BW
    cdef CLUT AW_POPEFRY
    cdef CLUT AW_PALMERW
    cdef CLUT GI_PR  # gi coefficients
    cdef CLUT GII_PR  # pre-interpolated gi coefficients (for one pixel)
    cdef CLUT AB_BRIC
    cdef CLUT RAMAN
    cdef float bw500
    cdef float a700
    cdef float mus
    cdef int Nwav

    cdef int[:] index  # multi-purpose vector

    def __init__(self, directory):

        self.Rw = None
        self.index = np.zeros(2, dtype='int32')

        self.read_iop(directory)
        self.read_gi(directory)

    def read_iop(self, directory):

        #
        # read water scattering coefficient
        #
        data_bw = np.genfromtxt(join(directory, 'morel_buiteveld_bsw.txt'), skip_header=1)
        self.BW = CLUT(data_bw[:,1], axes=[data_bw[:,0]], debug=True)   # FIXME (DEBUG)
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
        gb = map(float, line[:line.find('=')].split())
        line = fp.readline()
        th0 = map(float, line[:line.find('=')].split())
        line = fp.readline()
        th = map(float, line[:line.find('=')].split())
        line = fp.readline()
        dphi = map(float, line[:line.find('=')].split())

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
            block = np.array(map(float, block.split())).reshape((ngb, 4))
            gi[:,:,ith0, ith, idphi] = block[:,:]

        fp.close()

        self.GI_PR = CLUT(gi, axes=[gb, None, th0, th, dphi])

        # initialize pre-interpolated gi coefficients
        # (empty)
        self.GII_PR = CLUT(np.zeros((ngb, 4)), axes=[gb, None])


    cdef init(self, float [:] wav, float sza, float vza, float raa):
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
            self.Rw = np.zeros(len(wav), dtype='float32')
            self.bw = np.zeros(len(wav), dtype='float32')
            self.aw = np.zeros(len(wav), dtype='float32')
            self.a_bric = np.zeros(len(wav), dtype='float32')
            self.b_bric = np.zeros(len(wav), dtype='float32')

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
            if ret == 0:
                self.AB_BRIC.index(1, 0)
                self.a_bric[i] = self.AB_BRIC.interp()
                self.AB_BRIC.index(1, 1)
                self.b_bric[i] = self.AB_BRIC.interp()
            elif ret > 0:  # above axis
                self.a_bric[i] = self.a700 * (w/700.)**(-80.)
                self.b_bric[i] = 0
            else:
                raise Exception('Error in AB_BRIC lookup')

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
        cdef float bbw, gamma, bp550, bbp550, bb, bbp
        cdef float aw, aphy, aCDM, a, aCDM443, S
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
        gamma = -0.733 * logchl + 1.499
        if gamma < 0: gamma = 0

        bp550 = 0.416 * (chl**0.766) * fb
        if (logchl < 2):
            bbp550 = (0.002 + 0.01*(0.5 - 0.25*logchl)) * bp550
        else:
            bbp550 = (0.002 + 0.01*(0.5 - 0.25*2.    )) * bp550

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

            a = aw + aphy + aCDM

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
            if ret < 0:
                raise Exception('Error in lookup for RAMAN (lambda)')
            ret = self.RAMAN.lookup(1, chl)  # clip both ends
            rho *= 1. + (self.RAMAN.interp()*self.mus/0.866)

            self.Rw[i] = rho


        return self.Rw

    def calc(self, wav, logchl, logfb=0., sza=0., vza=0., raa=0.):
        ''' water reflectance calculation (python interface) '''
        self.init(wav.astype('float32'), float(sza), float(vza), float(raa))
        params = np.zeros(2, dtype='float32')
        params[0] = logchl
        params[1] = logfb
        R = self.calc_rho(params)
        return np.array(R)

cdef class MorelMaritorena(WaterModel):
    def __init__(self):
        raise NotImplementedError
    cdef float[:] calc_rho(self, float[:] x):
        raise NotImplementedError

def test():
    pr = ParkRuddick('/home/francois/MERIS/POLYMER/auxdata/common/')
    pr.init(np.linspace(401, 800, 100, dtype='float32'), 0, 0, 0)
    a = pr.calc_rho(np.array([0., 0.], dtype='float32'))
    # print np.array(a)
