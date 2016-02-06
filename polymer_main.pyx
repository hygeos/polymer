
import numpy as np
cimport numpy as np

include "minimization.pyx"


cdef class F(NelderMeadMinimizer):

    def __init__(self):
        super().__init__(self)

    cdef float eval(self, float [:] x):
        return 0.

cdef optimize(double [:,:,:] Rprime):
    Nx = Rprime.shape[0]
    Ny = Rprime.shape[1]
    Nb = Rprime.shape[2]
    print 'processing a block of {}x{}x{}'.format(Nx, Ny, Nb)

    f = F(2)
    cdef float [:] x0 = np.ndarray(2, dtype='float32')

    #
    # pixel loop
    #
    for i in range(Nx):
        for j in range(Ny):
            pass


def polymer_optimize(block):
    print block.Rprime.shape
    optimize(block.Rprime)

