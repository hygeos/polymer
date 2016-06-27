import numpy as np
cimport numpy as np
from numpy.linalg import inv
from common import BITMASK_INVALID, L2FLAGS
from libc.math cimport nan, exp, log
from cpython.exc cimport PyErr_CheckSignals

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

    # bands
    cdef int N_bands_corr
    cdef int[:] i_corr_read  # index or the 'corr' bands within the 'read' bands
    cdef int N_bands_oc
    cdef int[:] i_oc_read  # index or the 'oc' bands within the 'read' bands
    cdef int N_bands_read

    def __init__(self, Ncoef, watermodel, params, *args, **kwargs):

        super(self.__class__, self).__init__(*args, **kwargs)

        self.w = watermodel
        self.C = np.zeros(Ncoef, dtype='float32')
        self.Ratm = np.zeros(len(params.bands_read()), dtype='float32') + np.NaN
        self.Ncoef = Ncoef

        self.thres_chi2 = params.thres_chi2
        self.constraint_amplitude, self.sigma2, self.sigma1 = params.constraint_bbs

        self.N_bands_corr = len(params.bands_corr)
        self.i_corr_read = np.searchsorted(
                params.bands_read(),
                params.bands_corr).astype('int32')
        self.N_bands_oc = len(params.bands_oc)
        self.i_oc_read = np.searchsorted(
                params.bands_read(),
                params.bands_oc).astype('int32')
        self.N_bands_read = len(params.bands_read())

    cdef init_pixel(self, float[:] Rprime, float[:,:] A, float[:,:] pA,
            float[:] Tmol,
            float[:] wav, float sza, float vza, float raa):
        '''
        set the input parameters for the current pixel
        '''
        self.Rprime = Rprime
        self.wav = wav  # bands_read
        self.pA = pA
        self.A = A
        self.Tmol = Tmol

        self.w.init(wav, sza, vza, raa)


    cdef float eval(self, float[:] x) except? -999:
        '''
        Evaluate cost function for vector parameters x
        '''

        cdef float C
        cdef float sumsq, dR, norm
        cdef int icorr, icorr_read
        cdef int ioc, ioc_read, iread
        cdef float sigma

        #
        # calculate the. water reflectance for the current parameters
        # (at bands_read)
        #
        self.Rwmod = self.w.calc_rho(x)
        cdef float[:] Rwmod = self.Rwmod

        #
        # Atmospheric fit
        #
        for ic in range(self.Ncoef):
            C = 0.
            for icorr in range(self.N_bands_corr):
                icorr_read = self.i_corr_read[icorr]
                C += self.pA[ic,icorr] * (self.Rprime[icorr_read]
                                          - self.Tmol[icorr_read]*Rwmod[icorr_read])
            self.C[ic] = C

        #
        # Calculate Ratm
        #
        for iread in range(self.N_bands_read):
            self.Ratm[iread] = 0.
            for ic in range(self.Ncoef):
                self.Ratm[iread] += self.C[ic] * self.A[iread,ic]


        #
        # calculate the residual
        #
        sumsq = 0.
        for ioc in range(self.N_bands_oc):
            ioc_read = self.i_oc_read[ioc]

            dR = self.Rprime[ioc_read]

            # subtract atmospheric signal
            dR -= self.Ratm[ioc_read]

            # divide by transmission
            dR /= self.Tmol[ioc_read]

            dR -= Rwmod[ioc_read]

            norm = Rwmod[ioc_read]
            if norm < self.thres_chi2:
                norm = self.thres_chi2

            sumsq += dR*dR/norm


        if self.constraint_amplitude != 0:
            # sigma equals sigma1 when chl = 0.01
            # sigma equals sigma2 when chl = 0.1
            sigma = self.sigma1*self.sigma1/self.sigma2*exp(log(self.sigma1/self.sigma2)*x[0])

            sumsq += self.constraint_amplitude * (1. - exp(-x[1]*x[1]/(2*sigma*sigma)))

        return sumsq

def atm_func(block, params, bands):
    '''
    Returns the matrix of coefficients for the atmospheric function
    A [im0, im1, bands, ncoef]

    Ratm = A.C
    Ratm: (shp0, shp1, nlam)
    A   : (shp0, shp1, nlam, ncoef)
    C   : (shp0, shp1, ncoef)
    '''
    # bands for atmospheric fit
    Nlam = len(bands)
    shp = block.size

    # correction bands wavelengths
    idx = np.searchsorted(params.bands_read(), bands)
    # transpose: move the wavelength dimension to the end
    lam = block.wavelen[:,:,idx]

    # initialize the matrix for inversion
    Ncoef = 3   # number of polynomial coefficients
    A = np.zeros((shp[0], shp[1], Nlam, Ncoef), dtype='float32')

    taum = 0.00877*((np.array(block.bands)[idx]/1000.)**(-4.05))
    Rgli0 = 0.02
    T0 = np.exp(-taum*((1-0.5*np.exp(-block.Rgli/Rgli0))*block.air_mass)[:,:,None])

    A[:,:,:,0] = T0*(lam/1000.)**0.
    A[:,:,:,1] = (lam/1000.)**-1.
    if params.atm_model == 'T0,-1,-4':
        A[:,:,:,2] = (lam/1000.)**-4.
    elif params.atm_model == 'T0,-1,Rmol':
        A[:,:,:,2] = block.Rmol[:,:,idx]
    else:
        raise Exception('Invalid atmospheric model "{}"'.format(params.atm_model))

    return A

def pseudoinverse(A):
    '''
    Calculate the pseudoinverse of array A over the last 2 axes
    (broadcasting the first axes)
    A* = ((A'.A)^(-1)).A'
    where X' is the transpose of X and X^-1 is the inverse of X

    shapes: A:  [...,i,j]
            A*: [...,j,i]
    '''

    # A'.A (with broadcasting)
    B = np.einsum('...ji,...jk->...ik', A, A)

    # check
    # if B.ndim == 4:
        # assert np.allclose(B[0,0,:,:], A[0,0,:,:].transpose().dot(A[0,0,:,:]), equal_nan=True)
        # assert np.allclose(B[-1,0,:,:], A[-1,0,:,:].transpose().dot(A[-1,0,:,:]), equal_nan=True)

    # (A^-1).A' (with broadcasting)
    pA = np.einsum('...ij,...kj->...ik', inv(B), A)

    # check
    # if B.ndim == 4:
        # assert np.allclose(pA[0,0], inv(B[0,0,:,:]).dot(A[0,0,:,:].transpose()), equal_nan=True)
        # assert np.allclose(pA[-1,0], inv(B[-1,0,:,:]).dot(A[-1,0,:,:].transpose()), equal_nan=True)

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


cdef raiseflag(unsigned short[:,:] bitmask, int i, int j, int flag):
    if not testflag(bitmask, i, j, flag):
        bitmask[i,j] += flag

cdef int testflag(unsigned short[:,:] bitmask, int i, int j, int flag):
    return bitmask[i,j] & flag != 0

cdef class PolymerMinimizer:

    cdef F f
    cdef int Nparams
    cdef int BITMASK_INVALID
    cdef float NaN
    cdef float[:,:] bounds
    cdef float[:] initial_point_1
    cdef float[:] initial_point_2
    cdef float[:] initial_step
    cdef float size_end_iter
    cdef int max_iter
    cdef int L2_FLAG_CASE2
    cdef object params
    cdef int normalize
    cdef int force_initialization

    def __init__(self, watermodel, params):

        self.Nparams = 2
        Ncoef = 3   # number of atmospheric coefficients
        self.f = F(Ncoef, watermodel, params, self.Nparams)
        self.BITMASK_INVALID = BITMASK_INVALID
        self.NaN = np.NaN

        self.bounds = np.array(params.bounds, dtype='float32')
        self.initial_point_1 = np.array(params.initial_point_1, dtype='float32')
        self.initial_point_2 = np.array(params.initial_point_2, dtype='float32')
        self.initial_step = np.array(params.initial_step, dtype='float32')
        self.size_end_iter = params.size_end_iter
        self.max_iter = params.max_iter
        self.L2_FLAG_CASE2 = L2FLAGS['CASE2']
        self.params = params
        self.normalize = params.normalize
        self.force_initialization = params.force_initialization

    cdef loop(self, block,
              float[:,:,:,:] A,
              float[:,:,:,:] pA
              ):
        '''
        cython method which does the main pixel loop
        (over a block)
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
        x0[:] = self.initial_point_1[:]

        # create the output datasets
        block.logchl = np.zeros(block.size, dtype='float32')
        cdef float[:,:] logchl = block.logchl
        block.niter = np.zeros(block.size, dtype='uint32')
        cdef unsigned int[:,:] niter = block.niter
        block.Rw = np.zeros(block.size+(block.nbands,), dtype='float32')
        cdef float[:,:,:] Rw = block.Rw
        block.Ratm = np.zeros(block.size+(block.nbands,), dtype='float32')
        cdef float[:,:,:] Ratm = block.Ratm

        cdef int i, j, ib

        #
        # pixel loop
        #
        for j in range(Ny):
            for i in range(Nx):

                if (bitmask[i,j] & self.BITMASK_INVALID) != 0:
                    logchl[i,j] = self.NaN
                    Rw[i,j,:] = self.NaN
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
                        raiseflag(bitmask, i, j, self.L2_FLAG_CASE2)
                        break

                # case2 optimization if first optimization fails
                if testflag(bitmask, i, j,  self.L2_FLAG_CASE2):

                    self.f.init(self.initial_point_2, self.initial_step)

                    while self.f.niter < self.max_iter:

                        self.f.iterate()

                        if self.f.size() < self.size_end_iter:
                            break
                        if not in_bounds(self.f.xmin, self.bounds):
                            break


                logchl[i,j] = self.f.xmin[0]
                niter[i,j] = self.f.niter

                # initialization of next pixel
                if self.force_initialization or testflag(bitmask, i, j,  self.L2_FLAG_CASE2):
                    x0[:] = self.initial_point_1[:]
                else:
                    x0[:] = self.f.xmin[:]


                # calculate water reflectance
                # and store atmospheric reflectance
                for ib in range(len(self.f.Rwmod)):
                    Rw[i,j,ib] = Rprime[i,j,ib] - self.f.Ratm[ib]
                    Rw[i,j,ib] /= Tmol[i,j,ib]

                    Ratm[i,j,ib] = self.f.Ratm[ib]

                # water reflectance normalization
                if self.normalize:
                    # Rw -> Rw*Rwmod[nadir]/Rwmod

                    for ib in range(len(self.f.Rwmod)):
                        Rw[i,j,ib] /= self.f.Rwmod[ib]

                    # calculate model reflectance at nadir
                    self.f.init_pixel(
                            Rprime[i,j,:],
                            A[i,j,:,:], pA[i,j,:,:],
                            Tmol[i,j,:],
                            wav[i,j,:],
                            0., 0., 0.)
                    self.f.w.calc_rho(self.f.xmin)

                    for ib in range(len(self.f.Rwmod)):
                        Rw[i,j,ib] *= self.f.Rwmod[ib]


            # reinitialize
            x0[:] = self.initial_point_1[:]

            # check for pending signals
            # (allowing to interrupt execution)
            PyErr_CheckSignals()


    def minimize(self, block):
        '''
        Call minimization code for a block
        (def method visible from python code)
        '''

        if self.params.partial >= 1:
            return

        # calculate glint reflectance from wind speed
        ok = (block.bitmask & BITMASK_INVALID) == 0
        block.Rgli = np.zeros_like(block.wind_speed) + np.NaN
        block.Rgli[ok] = glitter(block.wind_speed[ok],
                                 block.mus[ok], block.muv[ok],
                                 block.scattering_angle[ok], phi=None, phi_vent=None)

        # calculate the atmospheric inversion coefficients
        # at bands_corr
        A = atm_func(block, self.params, self.params.bands_corr)
        pA = pseudoinverse(A)

        # the model coefficients, at bands_read
        A = atm_func(block, self.params, self.params.bands_read())

        self.loop(block, A, pA)


