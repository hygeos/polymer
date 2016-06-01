import numpy as np
cimport numpy as np
from numpy.linalg import inv
from common import BITMASK_INVALID
from libc.math cimport nan, exp, log

from neldermead cimport NelderMeadMinimizer
from water cimport WaterModel
from glint import glitter


# TODO: deal with selection of bands in the inversion
# TODO: formalize the expression of atmospheric component
# as a sum of N arbitrary terms

cdef class F(NelderMeadMinimizer):
    '''
    Defines the cost function minimized by Polymer
    Inherits from NelderMeadMinimizer which provides method minimize
    '''

    cdef float[:] Rprime
    cdef float[:] Tmol
    cdef float[:] wav
    cdef WaterModel w

    # [Ratm] = [A] . [C]
    # where A is the matrix of the polynomial exponents for each wavelength (nlam x ncoef)
    # [C] = [pA] . [Ratm]    where [pA] is the pseudoinverse of matrix [A]  (ncoef x nlam)
    cdef float[:,:] A
    cdef float[:,:] pA
    cdef int Ncoef
    cdef float thres_chi2
    cdef float constraint_amplitude, sigma1, sigma2

    cdef float[:] C  # ci coefficients (ncoef)
    cdef float[:] Rwmod
    cdef float[:] Ratm

    def __init__(self, Ncoef, watermodel, params, *args, **kwargs):

        super(self.__class__, self).__init__(*args, **kwargs)

        self.w = watermodel
        self.C = np.zeros(Ncoef, dtype='float32')
        self.Ratm = np.zeros(len(params.bands_read()), dtype='float32')
        self.Ncoef = Ncoef

        self.thres_chi2 = params.thres_chi2
        self.constraint_amplitude, self.sigma1, self.sigma2 = params.constraint_bbs

    cdef init_pixel(self, float[:] Rprime, float[:,:] A, float[:,:] pA,
            float[:] Tmol,
            float[:] wav, float sza, float vza, float raa):
        '''
        set the input parameters for the current pixel
        '''
        self.Rprime = Rprime
        self.wav = wav
        self.pA = pA
        self.A = A
        self.Tmol = Tmol

        self.w.init(wav, sza, vza, raa)


    cdef float eval(self, float[:] x) except? -999:
        '''
        Evaluate cost function for vector parameters x
        '''
        # TODO
        # take into account bands_corr, bands_oc, etc.

        cdef float C
        cdef float sumsq, dR, norm
        cdef int il, ic
        cdef float sigma

        #
        # calculate the. water reflectance for the current parameters
        #
        self.Rwmod = self.w.calc_rho(x)
        cdef float[:] Rwmod = self.Rwmod

        #
        # atmospheric fit
        #
        for ic in range(self.Ncoef):
            C = 0.
            for il in range(len(Rwmod)):
                C += self.pA[ic,il] * (self.Rprime[il] - self.Tmol[il]*Rwmod[il])
            self.C[ic] = C

        #
        # calculate the residual
        #
        sumsq = 0.
        for il in range(len(Rwmod)):

            dR = self.Rprime[il]

            # subtract atmospheric signal
            self.Ratm[il] = 0
            for ic in range(self.Ncoef):
                self.Ratm[il] += self.C[ic] * self.A[il,ic]
            dR -= self.Ratm[il]

            # divide by transmission
            dR /= self.Tmol[il]

            dR -= Rwmod[il]

            norm = Rwmod[il]
            if norm < self.thres_chi2:
                norm = self.thres_chi2

            sumsq += dR*dR/norm


        if self.constraint_amplitude != 0:
            # sigma equals sigma1 when chl = 0.01
            # sigma equals sigma2 when chl = 0.1
            sigma = self.sigma1*self.sigma1/self.sigma2*exp(log(self.sigma1/self.sigma2)*x[0])

            sumsq += self.constraint_amplitude * (1. - exp(-x[1]*x[1]/(2*sigma*sigma)))

        return sumsq

def atm_func(block, params):
    '''
    Returns the matrix of atmospheric components
    A [im0, im1, nlam, ncoef]

    Note: pseudo-inverse of A
    A* = ((A'.A)^(-1)).A'     where B' is the transpose of B and B^-1 is the inverse of B

    Ratm = A.C
    Ratm: (shp0, shp1, nlam)
    A   : (shp0, shp1, nlam, ncoef)
    C   : (shp0, shp1, ncoef)

    B = (A'.A) = tensordot(A, A, axes=[[0], [0]])
        (shp0, shp1, ncoef, ncoef)
    '''
    # bands for atmospheric fit
    Nlam = len(params.bands_corr)
    shp = block.size

    # correction bands wavelengths
    i_corr = np.searchsorted(params.bands_read(), params.bands_corr)
    # transpose: move the wavelength dimension to the end
    lam = block.wavelen[:,:,i_corr]

    # initialize the matrix for inversion
    Ncoef = 3   # number of polynomial coefficients
    A = np.zeros((shp[0], shp[1], Nlam, Ncoef), dtype='float32')



    taum = 0.00877*((block.wavelen/1000.)**(-4.05))
    Rgli0 = 0.02
    T0 = np.exp(-(1-0.5*np.exp(-block.Rgli[...,None]/Rgli0)*taum)*(1/block.mus[...,None] + 1/block.muv[...,None]))

    A[:,:,:,0] = T0*(lam/1000.)**0.
    A[:,:,:,1] = (lam/1000.)**-1.
    A[:,:,:,2] = (lam/1000.)**-4.

    return A

def pseudoinverse(A):
    '''
    Calculate the pseudoinverse of array A over the last 2 axes
    (broadcasting the first axes)
    A* = ((A'.A)^(-1)).A'
    where X' is the transpose of X and X^-1 is the inverse of X
    '''

    # A'.A (with broadcasting)
    B = np.einsum('...ji,...jk->...ik', A, A)

    # check
    if B.ndim == 4:
        assert np.allclose(B[0,0,:,:], A[0,0,:,:].transpose().dot(A[0,0,:,:]), equal_nan=True)
        assert np.allclose(B[-1,0,:,:], A[-1,0,:,:].transpose().dot(A[-1,0,:,:]), equal_nan=True)

    # (A^-1).A' (with broadcasting)
    pA = np.einsum('...ij,...kj->...ik', inv(B), A)

    # check
    if B.ndim == 4:
        assert np.allclose(pA[0,0], inv(B[0,0,:,:]).dot(A[0,0,:,:].transpose()), equal_nan=True)
        assert np.allclose(pA[-1,0], inv(B[-1,0,:,:]).dot(A[-1,0,:,:].transpose()), equal_nan=True)

    return pA


cdef int in_bounds(float[:] x, float[:,:] bounds):
    '''
    returns whether vector x (N dimensions) is in bounds (Nx2 dimensions)
    '''
    cdef int r = 1
    cdef int i
    for i in range(x.size):
        if (x[i] < bounds[i,0]) or (x[i] > bounds[i,1]):
            r = 0
    return r


cdef class PolymerMinimizer:

    cdef F f
    cdef int Nparams
    cdef int BITMASK_INVALID
    cdef float NaN
    cdef float[:,:] bounds
    cdef float[:] initial_point
    cdef float[:] initial_step
    cdef float size_end_iter
    cdef int max_iter

    def __init__(self, watermodel, params):

        self.Nparams = 2
        Ncoef = 3   # number of atmospheric coefficients
        self.f = F(Ncoef, watermodel, params, self.Nparams)
        self.BITMASK_INVALID = BITMASK_INVALID
        self.NaN = np.NaN

        self.bounds = np.array(params.bounds, dtype='float32')
        self.initial_point = np.array(params.initial_point, dtype='float32')
        self.initial_step = np.array(params.initial_step, dtype='float32')
        self.size_end_iter = params.size_end_iter
        self.max_iter = params.max_iter

    cdef loop(self, block,
              float[:,:,:,:] A,
              float[:,:,:,:] pA
              ):
        '''
        cython method which does the main pixel loop
        '''

        cdef float[:,:,:] Rprime = block.Rprime
        cdef float[:,:,:] Tmol = block.Tmol
        cdef float[:,:,:] wav = block.wavelen
        cdef float[:,:] sza = block.sza
        cdef float[:,:] vza = block.vza
        cdef float[:,:] raa = block.raa

        cdef unsigned short[:,:] bitmask = block.bitmask
        # cdef int Nb = Rprime.shape[0]
        cdef int Nx = Rprime.shape[0]
        cdef int Ny = Rprime.shape[1]
        cdef float[:] x

        cdef float[:] x0 = np.zeros(self.Nparams, dtype='float32')
        x0[:] = self.initial_point[:]

        # create the output datasets
        block.logchl = np.zeros(block.size, dtype='float32')
        cdef float[:,:] logchl = block.logchl
        block.niter = np.zeros(block.size, dtype='uint32')
        cdef unsigned int[:,:] niter = block.niter
        block.Rw = np.zeros(block.size+(block.nbands,), dtype='float32')
        cdef float[:,:,:] Rw = block.Rw
        cdef int i, j, ib

        #
        # pixel loop
        #
        for i in range(Nx):
            for j in range(Ny):

                if (bitmask[i,j] & self.BITMASK_INVALID) != 0:
                    logchl[i,j] = self.NaN
                    continue

                self.f.init_pixel(
                        Rprime[i,j,:],
                        A[i,j,:,:], pA[i,j,:,:],
                        Tmol[i,j,:],
                        wav[i,j,:],
                        sza[i,j], vza[i,j], raa[i,j])

                self.f.init(x0, self.initial_step)

                while self.f.niter < self.max_iter:

                    self.f.iterate()

                    if self.f.size() < self.size_end_iter:
                        break
                    if not in_bounds(self.f.xmin, self.bounds):
                        break

                # case2 optimization if first fails
                # TODO

                logchl[i,j] = self.f.xmin[0]
                niter[i,j] = self.f.niter

                # initialization of next pixel
                if in_bounds(self.f.xmin, self.bounds):
                    x0[:] = self.f.xmin[:]
                else:
                    x0[:] = self.initial_point[:]


                # option to evaluate f a last time (?)
                # TODO

                # calculate water reflectance
                for ib in range(len(self.f.Rwmod)):
                    Rw[i,j,ib] = Rprime[i,j,ib] - self.f.Ratm[ib]
                    Rw[i,j,ib] /= Tmol[i,j,ib]

                # water reflectance normalization
                # TODO


            # reinitialize
            x0[:] = self.f.xmin[:]


    def minimize(self, block, params):
        '''
        Call minimization code for a block
        (def method visible from python code)
        '''
        # FIXME
        # avoid passing params several times

        if params.partial >= 1:
            return

        # calculate glint reflectance from wind speed
        ok = (block.bitmask & BITMASK_INVALID) == 0
        block.Rgli = np.zeros_like(block.wind_speed) + np.NaN
        block.Rgli[ok] = glitter(block.wind_speed[ok],
                                 block.mus[ok], block.muv[ok],
                                 block.scattering_angle[ok], phi=None, phi_vent=None)

        # calculate the atmospheric inversion coefficients
        A = atm_func(block, params)
        pA = pseudoinverse(A)

        # TODO
        # 'A' should be provided at the OC bands.
        # For now, we assert bands_rw are the same as bands_corr
        assert set(params.bands_corr) == set(params.bands_oc)


        self.loop(block, A, pA)


