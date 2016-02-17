import numpy as np
cimport numpy as np

from os.path import join


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
    cdef float[:] bw   # scattering coefficient
    cdef float[:] A    # absorption parameters A and B (Bricaud)
    cdef float[:] B    # absorption parameters A and B (Bricaud)
    cdef CLUT BW
    cdef float bw500

    def __init__(self, directory):

        self.Rw = None

        # read water scattering coefficient
        data_bw = np.genfromtxt(join(directory, 'morel_buiteveld_bsw.txt'), skip_header=1)
        self.BW = CLUT(data_bw[:,1].astype('float32'), axes=[data_bw[:,0]], debug=True)   # FIXME (DEBUG)
        assert data_bw[-1,0] == 500.
        self.bw500 = data_bw[-1,1]


    cdef init(self, float [:] wav, float sza, float vza, float raa):
        '''
        set initialize the model parameters
        '''
        cdef int i
        cdef int ret
        self.wav = wav

        # array initialization (unique)
        if self.Rw is None:
            self.Rw = np.zeros(len(wav), dtype='float32')
            self.bw = np.zeros(len(wav), dtype='float32')

        # interpolate scattering coefficient
        for i, w in enumerate(wav):
            ret = self.BW.lookup(0, w)
            if ret > 0:
                self.bw[i] = self.bw500 * (w/500.)**-4.
            elif ret < 0:
                raise Exception('Error')
            else:
                self.bw[i] = self.BW.interp()

    cdef float[:] calc_rho(self, float[:] x):
        '''
        calculate the reflectance above the water surface at all bands
        '''

        # x is [logchl, logfb, logfa] or shorter
        cdef int N = len(x)
        cdef float fa, fb
        cdef float logchl = x[0]
        cdef float chl = 10**logchl

        if N >= 2:
            fb = 10**x[1]
        else:
            fb = 1.

        if N >= 3:
            fa = 10**x[2]
        else:
            fa = 1.

        # wavelength loop
        for i in range(N):
            pass # FIXME



            #
            # IOP
            #
            # 1) scattering parameters
            # pure sea water (Morel/Buiteveld, lambda^-4,
            # see Morel2007)

            # 2) absorption parameters


            # FIXME
        return self.Rw
