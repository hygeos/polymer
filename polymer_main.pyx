
import numpy as np
cimport numpy as np

include "minimization.pyx"
include "water.pyx"


cdef class F(NelderMeadMinimizer):

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)

    cdef float eval(self, float [:] x):
        '''
        Evaluate cost function for vector parameters x
        '''
        return 0.

cdef class PolymerMinimizer:

    cdef F f
    cdef int Nparams

    def __init__(self):

        self.Nparams = 2
        self.f = F(self.Nparams)

    cdef loop(self, double [:,:,:] Rprime):

        Nx = Rprime.shape[0]
        Ny = Rprime.shape[1]
        Nb = Rprime.shape[2]
        print 'processing a block of {}x{}x{}'.format(Nx, Ny, Nb)

        cdef float [:] x0 = np.ndarray(self.Nparams, dtype='float32')

        #
        # pixel loop
        #
        for i in range(Nx):
            for j in range(Ny):
                self.f.eval(x0)
                # self.f.minimize(x0)
                pass

    def minimize(self, block):
        '''
        Call minimization code for a block
        (def method visible from python code)
        '''
        self.loop(block.Rprime)

