import numpy as np
cimport numpy as np
from numpy.linalg import inv
from polymer.common import L2FLAGS
from libc.math cimport nan, exp, log, abs, sqrt, isnan
from cpython.exc cimport PyErr_CheckSignals
import pandas as pd
from pathlib import Path
import xarray as xr
from core import tools

from polymer.neldermead cimport dot
from polymer.glint import glitter
from polymer.atm import atm_func, weighted_pseudoinverse, pseudoinverse
from polymer.polymer_minimizer cimport PolymerMinimizer

'''
main polymer iterative optimization module
'''


cdef int in_bounds(float[:] x, float[:,:] bounds):
    '''
    returns whether vector x (N dimensions) is in bounds (Nx2 dimensions)
    '''
    cdef int r = 1
    cdef int i
    for i in range(x.shape[0]):
        if (x[i] < bounds[i,0]) or (x[i] > bounds[i,1]):
            r = 0
    return r


cdef int raiseflag(unsigned short[:,:] bitmask, int i, int j, int flag):
    if not testflag(bitmask, i, j, flag):
        bitmask[i,j] += flag

cdef int testflag(unsigned short[:,:] bitmask, int i, int j, int flag):
    return bitmask[i,j] & flag != 0


cdef class PolymerSolver:

    cdef PolymerMinimizer f
    cdef int Nparams
    cdef int BITMASK_INVALID
    cdef float NaN
    cdef float[:,:] bounds
    cdef float[:] initial_point_1
    cdef float[:] initial_point_2
    cdef float[:,:] initial_points   # check consistency WRT above
    cdef float[:] initial_step
    cdef float size_end_iter
    cdef int max_iter
    cdef int L2_FLAG_CASE2
    cdef int L2_FLAG_INCONSISTENCY
    cdef int L2_FLAG_THICK_AEROSOL
    cdef int L2_FLAG_OUT_OF_BOUNDS
    cdef int L2_FLAG_EXCEPTION
    cdef int L2_FLAG_ANOMALY_RWMOD_BLUE
    cdef object params
    cdef int normalize
    cdef int force_initialization
    cdef int reinit_rw_neg
    cdef int[:] dbg_pt
    cdef int Rprime_consistency
    cdef int N_bands_oc
    cdef int[:] i_oc_read  # index or the 'oc' bands within the 'read' bands
    cdef int N_bands_read
    cdef int uncertainties
    cdef int Ncoef
    cdef int firstguess_method

    def __init__(self, watermodel, params):

        self.Nparams = len(params.initial_step)
        self.Ncoef = params.Ncoef   # number of atmospheric coefficients
        self.f = PolymerMinimizer(self.Ncoef, watermodel, params, self.Nparams)
        self.BITMASK_INVALID = params.BITMASK_INVALID
        self.NaN = np.nan

        self.bounds = np.array(params.bounds, dtype='float32')
        self.initial_point_1 = np.array(params.initial_point_1, dtype='float32')
        self.initial_point_2 = np.array(params.initial_point_2, dtype='float32')
        self.initial_points = np.array(params.initial_points, dtype='float32')
        self.initial_step = np.array(params.initial_step, dtype='float32')
        self.size_end_iter = params.size_end_iter
        self.max_iter = params.max_iter
        self.L2_FLAG_CASE2 = L2FLAGS['CASE2']
        self.L2_FLAG_INCONSISTENCY = L2FLAGS['INCONSISTENCY']
        self.L2_FLAG_THICK_AEROSOL = L2FLAGS['THICK_AEROSOL']
        self.L2_FLAG_OUT_OF_BOUNDS = L2FLAGS['OUT_OF_BOUNDS']
        self.L2_FLAG_EXCEPTION = L2FLAGS['EXCEPTION']
        self.L2_FLAG_ANOMALY_RWMOD_BLUE = L2FLAGS['ANOMALY_RWMOD_BLUE']
        self.params = params
        self.uncertainties = params.uncertainties
        self.normalize = params.normalize
        self.force_initialization = params.force_initialization
        self.reinit_rw_neg = params.reinit_rw_neg
        self.dbg_pt = np.array(params.dbg_pt, dtype='int32')
        self.Rprime_consistency = params.Rprime_consistency
        self.firstguess_method = params.firstguess_method

        self.N_bands_oc = len(params.bands_oc)
        self.i_oc_read = np.searchsorted(
                params.bands_read(),
                params.bands_oc).astype('int32')
        self.N_bands_read = len(params.bands_read())


    cdef loop(self,
              float[:,:,:] Rprime,
              float[:,:,:] Rprime_noglint,
              float[:,:,:] Rmol,
              float[:,:] Rnir,
              float[:,:] Rgli,
              float[:,:,:] Tmol,
              float[:,:,:] wav,
              float[:] cwav,
              float[:,:] sza,
              float[:,:] vza,
              float[:,:] raa,
              float[:,:] air_mass,
              float[:,:] wind_speed,
              unsigned short[:,:] bitmask,
              float[:,:,:] Rtoa_var,
              ):
        '''
        cython method which does the main pixel loop
        (over a block)
        '''

        cdef int Nx = Rprime.shape[0]
        cdef int Ny = Rprime.shape[1]
        cdef int block_nbands = Rprime.shape[2]
        cdef int rw_neg
        block_size = (Nx, Ny)

        cdef float[:] x0 = np.zeros(self.Nparams, dtype='float32')
        x0[:] = self.initial_point_1[:]

        # create the output datasets
        block_logchl = np.zeros(block_size, dtype='float32')
        cdef float[:,:] logchl = block_logchl
        block_fa = np.zeros(block_size, dtype='float32')
        cdef float[:,:] fa = block_fa
        block_logfb = np.zeros(block_size, dtype='float32')
        cdef float[:,:] logfb = block_logfb
        block_SPM = np.zeros(block_size, dtype='float32')
        cdef float[:,:] SPM = block_SPM
        block_niter = np.zeros(block_size, dtype='uint32')
        cdef unsigned int[:,:] niter = block_niter
        block_Rw = np.zeros(block_size+(block_nbands,), dtype='float32')
        cdef float[:,:,:] Rw = block_Rw
        block_Ratm = np.zeros(block_size+(block_nbands,), dtype='float32')
        cdef float[:,:,:] Ratm = block_Ratm
        block_Rwmod = np.zeros(block_size+(block_nbands,), dtype='float32') + np.nan
        cdef float[:,:,:] Rwmod = block_Rwmod
        block_eps = np.zeros(block_size, dtype='float32')
        cdef float[:,:] eps = block_eps
        block_Ci = np.zeros(block_size+(self.Ncoef,), dtype='float32')
        cdef float[:,:,:] Ci = block_Ci

        cdef float[:,:] logchl_unc
        cdef float[:,:] logfb_unc
        cdef float[:,:,:] rho_w_unc
        cdef float[:,:] rho_w_mod_cov
        cdef float[:,:] d_rw_x_cov
        cdef float[:,:] d_rw_x
        if self.uncertainties:
            block_logchl_unc = np.zeros(block_size, dtype='float32') + np.nan
            logchl_unc = block_logchl_unc
            block_logfb_unc = np.zeros(block_size, dtype='float32') + np.nan
            logfb_unc = block_logfb_unc
            block_rho_w_unc = np.zeros(block_size+(block_nbands,), dtype='float32') + np.nan
            rho_w_unc = block_rho_w_unc
            d_rw_x = np.zeros((block_nbands, self.Nparams), dtype='float32') + np.nan
            rho_w_mod_cov = np.zeros((block_nbands, block_nbands), dtype='float32') + np.nan
            d_rw_x_cov = np.zeros((block_nbands, self.Nparams), dtype='float32') + np.nan

        cdef int i, j, ib, ioc, iparam
        cdef int flag_reinit = 0
        cdef float Rw_max
        cdef float[:] wav0
        cdef float sza0, vza0, raa0
        cdef float sigmasq
        cdef float delta = 0.05
        cdef float[:,:] Rwmod_fg

        
        if self.initial_points.size:
            Rwmod_fg = np.zeros((self.initial_points.shape[0], block_nbands),
                                dtype='float32') + np.nan
            self.init_first_guess(Rwmod_fg, cwav)

        # Atmospheric model calculation
        # at bands_corr
        A = atm_func(wav,
                     Rmol,
                     Tmol,
                     Rgli,
                     air_mass,
                     self.params,
                     self.params.bands_corr)
    
        if self.params.weights_corr is None:
            pA = pseudoinverse(A)
        else:
            pA = weighted_pseudoinverse(
                    A, np.diag(self.params.weights_corr).astype('float32'))

        # the model coefficients, at bands_read
        A = atm_func(wav,
                     Rmol,
                     Tmol,
                     Rgli,
                     air_mass,
                     self.params,
                     self.params.bands_read())

        #
        # pixel loop
        #
        for j in range(Ny):
            for i in range(Nx):

                if (bitmask[i,j] & self.BITMASK_INVALID) != 0:
                    logchl[i,j] = self.NaN
                    fa[i,j] = self.NaN
                    SPM[i,j] = self.NaN
                    logfb[i,j] = self.NaN
                    Rw[i,j,:] = self.NaN
                    Ci[i,j,:] = self.NaN
                    continue

                if self.f.init_pixel(
                        Rprime[i,j,:],
                        Rprime_noglint[i,j,:],
                        A[i,j,:,:], pA[i,j,:,:],
                        Tmol[i,j,:],
                        wav[i,j,:],
                        sza[i,j], vza[i,j], raa[i,j],
                        wind_speed[i,j]):
                    raiseflag(bitmask, i, j, self.L2_FLAG_EXCEPTION)
                    continue

                # first guess
                if self.initial_points.size:
                    self.first_guess(Rwmod_fg, x0, i, j)
                
                self.f.init(x0, self.initial_step)

                # visualization of the cost function
                if self.dbg_pt[0] >= 0:
                    if ((self.dbg_pt[0] == i) and (self.dbg_pt[1] == j)):
                        self.visu_costfunction()
                    else:
                        continue


                # optimization loop
                while self.f.niter < self.max_iter:

                    self.f.iterate()

                    if self.f.size() < self.size_end_iter:
                        break
                    if not in_bounds(self.f.xmin, self.bounds):
                        raiseflag(bitmask, i, j, self.L2_FLAG_CASE2)
                        break

                # case2 optimization if first optimization fails
                if testflag(bitmask, i, j, self.L2_FLAG_CASE2) and (not self.initial_points.size):

                    self.f.init(self.initial_point_2, self.initial_step)

                    while self.f.niter < self.max_iter:

                        self.f.iterate()

                        if self.f.size() < self.size_end_iter:
                            break
                        if not in_bounds(self.f.xmin, self.bounds):
                            raiseflag(bitmask, i, j, self.L2_FLAG_OUT_OF_BOUNDS)
                            break
            


                # update water model with final parameters
                self.f.w.calc_rho(self.f.xmin)

                logchl[i,j] = self.f.xmin[0]
                eps[i,j] = self.f.fsim[0]
                if self.Nparams >= 2:
                    logfb[i,j] = self.f.xmin[1]
                if self.Nparams >= 3:
                    fa[i,j] = self.f.xmin[2]
                niter[i,j] = self.f.niter
                SPM[i,j] = self.f.w.SPM

                # calculate water reflectance
                # and store atmospheric reflectance
                rw_neg = 0
                for ib in range(self.N_bands_read):
                    Rw[i,j,ib] = Rprime[i,j,ib] - self.f.Ratm[ib]
                    Rw[i,j,ib] /= Tmol[i,j,ib]
                    if Rw[i,j,ib] < 0:
                        rw_neg = 1

                    Rwmod[i,j,ib] = self.f.Rwmod[ib]

                    Ratm[i,j,ib] = self.f.Ratm[ib]
                
                if self.uncertainties:
                    # 1) Uncertainty on the marine parameters
                    # normalize by sigma² = y_min/(N-n), with N = number of observations,
                    # and n = number of parameters fitted
                    # see [Nelder Mead, 1965]
                    sigmasq = self.f.fsim[0]/(self.N_bands_oc-self.Nparams-self.params.Ncoef)
                    self.f.calc_cov(2*sigmasq)

                    logchl_unc[i,j] = self.f.cov[0, 0]
                    logfb_unc[i,j] = self.f.cov[1, 1]

                    # 2) calculate the sensitivity of Rw to the marine parameters
                    for iparam in range(self.Nparams):
                        x0[iparam] = self.f.xmin[iparam]
                    for iparam in range(self.Nparams):
                        x0[iparam] += delta
                        self.f.eval(x0)
                        for ib in range(self.N_bands_read):
                            # the variation of Rw is equal to the opposite of the variation of Ratm
                            d_rw_x[ib, iparam] = (Ratm[i,j,ib] - self.f.Ratm[ib])/delta
                        x0[iparam] = self.f.xmin[iparam]

                    # 3) calculate rho_w_mod_cov from the Jacobian matrix of the model
                    # (eq 55 - 58 of E3UB)
                    # rho_w_mod_cov = d_rw_x . f.cov . d_rw_x'
                    #    [NbxNb]      [NbxNp] [NpxNp] [NpxNb]
                    dot(d_rw_x_cov, d_rw_x, self.f.cov, 0)
                    dot(rho_w_mod_cov, d_rw_x_cov, d_rw_x, 1)

                    for ib in range(self.N_bands_read):
                        rho_w_unc[i,j,ib] = sqrt(rho_w_mod_cov[ib, ib] + Rtoa_var[i,j,ib])/Tmol[i,j,ib]

                    self.f.w.calc_rho(self.f.xmin)

                
                # Store Ci coefficients
                for ib in range(self.Ncoef):
                    Ci[i,j,ib] = self.f.C[ib]
                
                # consistency test at bands_oc
                for ioc in range(self.N_bands_oc):
                    ib = self.i_oc_read[ioc]
                    if (self.Rprime_consistency and (
                              (self.f.Ratm[ib] > Rprime_noglint[i,j,ib])
                           or (self.f.Rwmod[ib]*Tmol[i,j,ib] > Rprime_noglint[i,j,ib]))):
                        raiseflag(bitmask, i, j, self.L2_FLAG_INCONSISTENCY)
                        flag_reinit = 1

                # water reflectance normalization
                if self.normalize:
                    # Rw -> Rw*Rwmod[nadir,lambda0]/Rwmod

                    for ib in range(self.N_bands_read):
                        Rw[i,j,ib] /= self.f.Rwmod[ib]

                    if self.normalize & 1:
                        # activate geometry normalization
                        sza0 = 0.
                        vza0 = 0.
                        raa0 = 0.
                    else:
                        sza0 = sza[i,j]
                        vza0 = vza[i,j]
                        raa0 = raa[i,j]

                    if self.normalize & 2:
                        # activate wavelength normalization
                        wav0 = cwav
                    else:
                        wav0 = wav[i,j,:]

                    # calculate model reflectance at nadir
                    self.f.init_pixel(
                            Rprime[i,j,:],
                            Rprime_noglint[i,j,:],
                            A[i,j,:,:], pA[i,j,:,:],
                            Tmol[i,j,:],
                            wav0,
                            sza0, vza0, raa0,
                            wind_speed[i,j])
                    self.f.w.calc_rho(self.f.xmin)

                    for ib in range(self.N_bands_read):
                        Rw[i,j,ib] *= self.f.Rwmod[ib]

                # thick aerosol flag
                # Rnir/max(Rw) > 10 - 1.5*logchl
                # avoid erroneous retrieval in case of very thick aerosol plumes
                Rw_max = 0.
                for ib in range(self.N_bands_read):
                    if Rw[i,j,ib] > Rw_max:
                        Rw_max = Rw[i,j,ib]
                if (Rnir[i,j]/Rw_max > 10 - 1.5*logchl[i,j]):
                    raiseflag(bitmask, i, j, self.L2_FLAG_THICK_AEROSOL)
                
                # ANOMALY_RWMOD_BLUE flag
                # Removes outliers appearing on MODIS results at high SZA
                # on recent years (eg 2019).
                if Rw[i,j,0] - Rwmod[i,j,0] > 0.005:
                    raiseflag(bitmask, i, j, self.L2_FLAG_ANOMALY_RWMOD_BLUE)

                # initialization of next pixel
                if (self.force_initialization
                        or testflag(bitmask, i, j,  self.L2_FLAG_CASE2)
                        or (rw_neg and self.reinit_rw_neg)
                        or (flag_reinit)
                        ):
                    x0[:] = self.initial_point_1[:]
                    flag_reinit = 0
                else:
                    x0[:] = self.f.xmin[:]


            # reinitialize
            x0[:] = self.initial_point_1[:]

            # check for pending signals
            # (allowing to interrupt execution)
            PyErr_CheckSignals()

        if self.uncertainties:
            ret = (
                block_logchl,
                block_fa,
                block_logfb,
                block_SPM,
                block_niter,
                block_Rw,
                block_Ratm,
                block_Rwmod,
                block_eps,
                # block_Ci,
                bitmask,
                block_logchl_unc,
                block_logfb_unc,
                block_rho_w_unc,
            )
        else:
            ret = (
                block_logchl,
                block_fa,
                block_logfb,
                block_SPM,
                block_niter,
                block_Rw,
                block_Ratm,
                block_Rwmod,
                block_eps,
                bitmask,
                # block_Ci,
            )

        return ret


    cdef int init_first_guess(self,
                              float[:,:] Rwmod_fg,
                              float[:] cwav):
        """
        Initialize reflectances for first guess

        Rwmod_fg: reflectance spectra [Npts, nbands]
        """
        cdef int i, j
        self.f.w.init_pixel(cwav, 0, 0, 0, 5)
        for i in range(Rwmod_fg.shape[0]):
            self.f.Rwmod = self.f.w.calc_rho(self.initial_points[i,:])
            for j in range(Rwmod_fg.shape[1]):
                Rwmod_fg[i,j] = self.f.Rwmod[j]

        return 0


    cdef first_guess(self,
                     float[:,:] Rwmod_fg, # Spectra calculated for first guess points
                     float[:] x0,
                     int i, int j):
        cdef float v_fguess, vmin_fguess
        cdef int i_fguess=0, ii
        vmin_fguess = -1
        v_fguess = -1
        for ii in range(self.initial_points.shape[0]):
            if self.firstguess_method == 0:
                # old method
                v_fguess = self.f.eval(self.initial_points[ii,:])
            else:
                # new method
                # avoid calling the water model each time, by
                # using the pre-calculated Rwmod_fg
                for k in range(Rwmod_fg.shape[1]):
                    self.f.Rwmod[k] = Rwmod_fg[ii,k]
                v_fguess = self.f.eval_atm(self.initial_points[ii,:])

            if (vmin_fguess < 0) or (v_fguess < vmin_fguess):
                vmin_fguess = v_fguess
                i_fguess = ii
            
        # Include last point in first guess
        # => With current values of self.f.Rwmod and self.f.xmin
        if ((not self.force_initialization)
            and not isnan(self.f.xmin[0])
            and in_bounds(self.f.xmin, self.bounds)
            and (self.f.eval_atm(self.f.xmin) < v_fguess)):

            # Reuse previous pixel (if better than all first guess pixels)
            for ii in range(x0.shape[0]):
                x0[ii] = self.f.xmin[ii]
        else:
            # Use first guess value
            for ii in range(x0.shape[0]):
                x0[ii] = self.initial_points[i_fguess,ii]

        if self.dbg_pt[0] >= 0:
            if ((self.dbg_pt[0] == i) and (self.dbg_pt[1] == j)):
                print('first guess: selected [{}] : ({}, {})'.format(i_fguess, x0[0], x0[1]))


    def minimize(self, block):
        '''
        Call minimization code for a block
        (def method visible from python code)
        '''

        # calculate glint reflectance from wind speed
        ok = (block.bitmask & self.BITMASK_INVALID) == 0
        block.Rgli = np.zeros_like(block.wind_speed, dtype='float32') + np.nan
        block.Rgli[ok] = glitter(block.wind_speed[ok],
                                 block.mus[ok], block.muv[ok],
                                 block.scattering_angle[ok], phi=None, phi_vent=None)

        if self.params.partial >= 1:
            return

        ret = self.loop(
            block.Rprime,
            block.Rprime_noglint,
            block.Rmol,
            block.Rnir,
            block.Rgli,
            block.Tmol,
            block.wavelen,
            block.cwavelen,
            block.sza,
            block.vza,
            block.raa,
            1/block.mus + 1/block.muv,  # air mass
            block.wind_speed.astype('float32'),
            block.bitmask,
            block.Rtoa_var if self.uncertainties else np.zeros_like(block.Rprime),
            )
        
        block.logchl = ret[0]
        block.fa = ret[1]
        block.logfb = ret[2]
        block.SPM = ret[3]
        block.niter = ret[4]
        block.Rw = ret[5]
        block.Ratm = ret[6]
        block.Rwmod = ret[7]
        block.eps = ret[8]
        block.bitmask[:] = ret[9]
        if self.uncertainties:
            block.logchl_unc = ret[10]
            block.logfb_unc = ret[11]
            block.rho_w_unc = ret[12]


    def visu_costfunction(self):
        '''
        Visualization of the cost function for current pixel
        '''
        from matplotlib.pyplot import pcolor, show, colorbar, plot

        NX, NY = 100, 100
        cost = np.zeros((NX, NY), dtype='float32')
        tab_p = np.array(np.meshgrid(
            np.linspace(-2, 0, NX),
            np.linspace(-0.5, 0.5, NY)), dtype='float32')
        for i in range(NX):
            for j in range(NY):
                cost[i,j] = self.f.eval(tab_p[:,i,j])

        pcolor(tab_p[0,:,:], tab_p[1,:,:],
               np.log10(cost), cmap='coolwarm')
        colorbar()

        # plot iterations
        xprev = None
        # self.f.init(self.initial_point_1, self.initial_step)
        xprev = self.initial_point_1
        while self.f.niter < self.max_iter:

            self.f.iterate()

            plot([xprev[0], self.f.xmin[0]],
                 [xprev[1], self.f.xmin[1]],
                 'k-')
            xprev = list(self.f.xmin)

            if self.f.size() < self.size_end_iter:
                break

    def apply(self, ds: xr.Dataset):

        if self.params.uncertainties:
            Rtoa_var = ds.Rtoa_var
        else:
            Rtoa_var = xr.zeros_like(ds.Rprime)
        wav = ds.wav.broadcast_like(ds.Rprime).astype('float32')

        ret = xr.apply_ufunc(
            self.loop,
            ds.Rprime,
            ds.Rprime_noglint,
            ds.rho_r,
            ds.Rnir,
            ds.Rgli,
            ds.Tmol,
            wav,
            ds.cwav,
            ds.sza,
            ds.vza,
            ds.raa,
            1/ds.mus + 1/ds.muv,
            ds.horizontal_wind.astype('float32'),
            ds.flags,
            Rtoa_var,
            dask='parallelized',
            input_core_dims=[
                ['bands'], ['bands'], ['bands'],
                [], [],
                ['bands'], ['bands'], ['bands'],
                [], [], [], [], [], [],
                ['bands'],
                ],
            output_core_dims=[
                [], [], [], [], [],
                ['bands'], ['bands'], ['bands'],
                [], [],
                ] + ([[], [], ['bands']] if self.uncertainties else []),
            output_dtypes=[
                'float32', 'float32', 'float32', 'float32', 'uint32',
                'float32', 'float32', 'float32',
                'float32', 'uint16',
                ] + (['float32', 'float32', 'float32'] if self.uncertainties else []),
        )
        ds['logchl'] = ret[0]
        ds['fa'] = ret[1]
        ds['logfb'] = ret[2]
        ds['SPM'] = ret[3]
        ds['niter'] = ret[4]
        ds['rho_w'] = ret[5]
        ds['Ratm'] = ret[6]
        ds['Rwmod'] = ret[7]
        ds['eps'] = ret[8]
        flags_out = ret[9]
        if self.uncertainties:
            ds['logchl_unc'] = ret[10]
            ds['logfb_unc'] = ret[11]
            ds['rho_w_unc'] = ret[12]

        # register the Polymer flags in the attributes
        flags_out.attrs.update(ds.flags.attrs)
        for k, v in L2FLAGS.items():
            tools.raiseflag(flags_out, k, v)
        ds['flags'] = flags_out


