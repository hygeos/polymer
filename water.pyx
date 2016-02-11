import numpy as np
cimport numpy as np


cdef class WaterModel:
    '''
    Base class for water reflectance models
    '''
    cdef float[:] calc_rho(self, float[:] x):
        raise Exception('WaterModel.calc_rho() shall be implemented')


cdef class ParkRuddick(WaterModel):

    cdef float[:] wav  # wavelength (nm)
    cdef float[:] Rw   # water reflectance (0+)
    cdef float[:] bw   # scattering coefficient
    cdef float[:] A    # absorption parameters A and B (Bricaud)
    cdef float[:] B    # absorption parameters A and B (Bricaud)

    def __init__(self, filename):
        print 'TODO: read', filename
        self.Rw = None


    cdef init(self, float [:] wav,
            float sza, float vza, float raa,
            float wind):
        '''
        set initialize the model parameters
        '''
        self.wav = wav


        # array initialization (unique)
        if self.Rw is None:
            self.Rw = np.array(len(wav), dtype='float32')
            self.A = np.array(len(wav), dtype='float32')
            self.B = np.array(len(wav), dtype='float32')

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
