
import numpy as np
cimport numpy as np

from neldermead cimport NelderMeadMinimizer
from water cimport WaterModel


cdef class F(NelderMeadMinimizer):
    '''
    Defines the cost function minimized by Polymer
    Inherits from NelderMeadMinimizer which provides method minimize
    '''

    cdef float[:] Rprime
    cdef float[:] wav
    cdef WaterModel w

    def __init__(self, watermodel, *args, **kwargs):

        super(self.__class__, self).__init__(*args, **kwargs)

        self.w = watermodel

    cdef init(self, float[:] Rprime, float[:] wav, float sza, float vza, float raa):
        '''
        set the input parameters for the current pixel
        '''
        self.Rprime = Rprime
        self.wav = wav

        self.w.init(wav, sza, vza, raa)


    cdef float eval(self, float[:] x):
        '''
        Evaluate cost function for vector parameters x
        '''
        cdef float[:] Rw

        # calculate the. water reflectance for the current parameters
        Rw = self.w.calc_rho(x)

        return 0.


cdef class PolymerMinimizer:

    cdef F f
    cdef int Nparams

    def __init__(self, watermodel):

        self.Nparams = 2
        self.f = F(watermodel, self.Nparams)

    cdef loop(self, float [:,:,:] Rprime,
              float [:,:,:] wav,
              float [:,:] sza,
              float [:,:] vza,
              float [:,:] raa
              ):
        '''
        cython method which does the main pixel loop
        '''

        cdef int Nb = Rprime.shape[0]
        cdef int Nx = Rprime.shape[1]
        cdef int Ny = Rprime.shape[2]

        print 'processing a block of {}x{}x{}'.format(Nx, Ny, Nb)

        cdef float [:] x0 = np.ndarray(self.Nparams, dtype='float32')

        #
        # pixel loop
        #
        for i in range(Nx):
            for j in range(Ny):
                self.f.init(Rprime[:,i,j], wav[:,i,j], sza[i,j], vza[i,j], raa[i,j])
                self.f.minimize(x0)

    # cdef test_interp(self):
    #     # TODO: remove
    #     cdef int[:] i0 = np.array([1, 1], dtype='int32')

    #     interp = CLUT(np.eye(3, dtype='float32'))
    #     # print '->', interp.get(i0)
    #     cdef float[:] x0 = np.array([0.1, 0.9], dtype='float32')
    #     # print '->', interp.interp(x0)
    #     x0[0] = -1
    #     # print '->', interp.interp(x0, i0)
    #     interp = CLUT(np.eye(5, dtype='float32'),
    #             debug=True,
    #             axes=[[10, 11, 12, 12.5, 12.7][::1], np.arange(5)*10])
    #     interp2 = CLUT(np.eye(5, dtype='float32'),
    #             debug=True,
    #             axes=[[10, 11, 12, 12.5, 12.7][::-1], np.arange(5)*10])
    #     for v in np.linspace(9.9,13,50):
    #         # res = interp.lookup(0, v)
    #         # print '->', v, res, interp._inf[0], interp._x[0], interp._interp[0]
    #         res = interp2.lookup(0, v)
    #         print '->', v, res, interp2._inf[0], interp2._x[0], interp2._interp[0]


    def minimize(self, block):
        '''
        Call minimization code for a block
        (def method visible from python code)
        '''
        # self.test_interp()   # FIXME

        self.loop(block.Rprime, block.wavelen, block.sza, block.vza, block.raa)


