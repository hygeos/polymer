
import numpy as np
cimport numpy as np
from numpy.linalg import inv

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

    # [Ratm] = [A] . [C]
    # where A is the matrix of the polynomial exponents for each wavelength (lam rows, 3 columns)
    # [C] = [pA] . [Ratm]    where [pA] is the pseudoinverse of matrix [A]
    cdef float [:,:,:,:] pA

    def __init__(self, watermodel, *args, **kwargs):

        super(self.__class__, self).__init__(*args, **kwargs)

        self.w = watermodel

    cdef init(self, float[:] Rprime,
            float [:,:,:,:] pA,
            float[:] wav, float sza, float vza, float raa):
        '''
        set the input parameters for the current pixel
        '''
        self.Rprime = Rprime
        self.wav = wav
        self.pA = pA

        self.w.init(wav, sza, vza, raa)


    cdef float eval(self, float[:] x) except? -999:
        '''
        Evaluate cost function for vector parameters x
        '''
        cdef float[:] Rw

        #
        # calculate the. water reflectance for the current parameters
        #
        Rw = self.w.calc_rho(x)

        #
        # atmospheric fit
        #


        #
        # calculate residual
        #

        return 0.


def calc_coeffs(block, params):
    '''
    Calculate the atmospheric fit coefficients for the whole block

    Note: pseudo inverse de A
    A* = ((A'.A)^(-1)).A'     où B' est la transposée et B^-1 est l'inverse de B

    Ratm = A.C
    Ratm: (shp0, shp1, nlam)
    A   : (shp0, shp1, nlam, ncoef)
    C   : (shp0, shp1, ncoef)

    B = (A'.A) = tensordot(A, A, axes=[[0], [0]])
        (shp0, shp1, ncoef, ncoef)

    '''
    # bands for atmospheric fit
    Nlam = len(params.bands_corr)
    Ncoef = 3   # number of polynomial coefficients
    shp = block.size

    # correction bands wavelengths
    i_corr = np.searchsorted(params.bands_read(), params.bands_corr)
    # transpose: move the wavelength dimension to the end
    lam = np.transpose(block.wavelen[i_corr,:,:], axes=[1, 2, 0])

    # initialize the matrix for inversion
    A = np.zeros((shp[0], shp[1], Nlam, Ncoef), dtype='float32')
    for i, c in enumerate([0, -1, -4]):
        # FIXME: fix polynomial expression
        A[:,:,:,i] = (lam/1000.)**c

    # A'.A (with broadcasting)
    B = np.einsum('...ji,...jk->...ik', A, A)

    # verification
    assert np.allclose(B[0,0,:,:], A[0,0,:,:].transpose().dot(A[0,0,:,:]))

    # (A^-1).A' (with broadcasting)
    pA = np.einsum('...ij,...kj->...ik', inv(B), A)

    # verification
    assert np.allclose(pA[0,0], inv(B[0,0,:,:]).dot(A[0,0,:,:].transpose()))

    return pA


cdef class PolymerMinimizer:

    cdef F f
    cdef int Nparams

    def __init__(self, watermodel):

        self.Nparams = 2
        self.f = F(watermodel, self.Nparams)

    cdef loop(self, float [:,:,:] Rprime,
              float [:,:,:,:] pA,
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

        cdef float [:] x0 = np.zeros(self.Nparams, dtype='float32')

        #
        # pixel loop
        #
        for i in range(Nx):
            for j in range(Ny):
                self.f.init(Rprime[:,i,j], pA,
                        wav[:,i,j], sza[i,j], vza[i,j], raa[i,j])
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


    def minimize(self, block, params):
        '''
        Call minimization code for a block
        (def method visible from python code)
        '''
        # self.test_interp()   # FIXME

        # calculate the atmospheric inversion coefficients
        pA = calc_coeffs(block, params)

        self.loop(block.Rprime, pA,
                block.wavelen, block.sza, block.vza, block.raa)


