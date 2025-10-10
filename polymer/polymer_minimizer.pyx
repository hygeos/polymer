import numpy as np
cimport numpy as np
from libc.math cimport exp, log, abs

from polymer.neldermead cimport NelderMeadMinimizer


"""
Polymer minimizer
"""

metrics_names = {
        'W_dR2_norm': W_dR2_norm,
        'W_absdR': W_absdR,
        'W_absdR_norm': W_absdR_norm,
        'W_absdR_Rprime': W_absdR_Rprime,
        'W_absdR2_Rprime2': W_absdR2_Rprime2,
        'W_dR2_Rprime_noglint2': W_dR2_Rprime_noglint2,
        'W_dR2_Rprime_noglint2_norm': W_dR2_Rprime_noglint2_norm,
        'polymer_3_5': polymer_3_5,
        }


cdef class PolymerMinimizer(NelderMeadMinimizer):
    '''
    Defines the cost function minimized by Polymer
    Inherits from NelderMeadMinimizer which provides method minimize
    '''

    def __init__(self, Ncoef, watermodel, params, *args, **kwargs):

        super(self.__class__, self).__init__(*args, **kwargs)

        self.w = watermodel
        self.C = np.zeros(Ncoef, dtype='float32')
        self.Ratm = np.zeros(len(params.bands_read()), dtype='float32') + np.NaN
        self.Ncoef = Ncoef

        self.thres_chi2 = params.thres_chi2
        self.constraint_amplitude, self.sigma2, self.sigma1 = params.constraint_logfb

        self.N_bands_corr = len(params.bands_corr)
        self.i_corr_read = np.searchsorted(
                params.bands_read(),
                params.bands_corr).astype('int32')
        self.N_bands_oc = len(params.bands_oc)
        self.i_oc_read = np.searchsorted(
                params.bands_read(),
                params.bands_oc).astype('int32')
        self.N_bands_read = len(params.bands_read())
        if params.weights_oc is None:
            self.weights_oc = np.ones(len(params.bands_oc), dtype='float32')
        else:
            assert len(params.weights_oc) == len(params.bands_oc)
            self.weights_oc = np.array(params.weights_oc, dtype='float32')

        try:
            self.metrics = metrics_names[params.metrics]
        except KeyError:
            raise Exception('Invalid metrics "{}"'.format(params.metrics))


    cdef int init_pixel(self, float[:] Rprime, float[:] Rprime_noglint,
                   float[:,:] A, float[:,:] pA,
                   float[:] Tmol,
                   float[:] wav, float sza, float vza, float raa, float ws) except -1:
        '''
        set the input parameters for the current pixel

        return 1 on error, 0 on success
        '''
        self.Rprime = Rprime
        self.Rprime_noglint = Rprime_noglint
        self.wav = wav  # bands_read
        self.pA = pA
        self.A = A
        self.Tmol = Tmol

        return self.w.init_pixel(wav, sza, vza, raa, ws)


    cdef float eval(self, float[:] x):
        '''
        Evaluate cost function for vector parameters x
        '''
        #
        # calculate the. water reflectance for the current parameters
        # (at bands_read)
        #
        self.Rwmod = self.w.calc_rho(x)

        return self.eval_atm(x)


    cdef float eval_atm(self, float[:] x):
        cdef float C
        cdef float sumsq, sumw, dR, norm
        cdef int icorr, icorr_read
        cdef int ioc, ioc_read, iread
        cdef float sigma

        cdef float[:] Rwmod = self.Rwmod   # TODO: don't use this intermediary variable ?

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
        sumw = 0.
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

            if (self.metrics == W_dR2_norm) or (self.metrics == polymer_3_5):
                sumsq += self.weights_oc[ioc]*dR*dR/norm
                sumw += self.weights_oc[ioc]

            elif self.metrics == W_absdR:
                sumsq += self.weights_oc[ioc]*abs(dR)
                sumw += self.weights_oc[ioc]

            elif self.metrics == W_absdR_norm:
                sumsq += self.weights_oc[ioc]*abs(dR)/norm
                sumw += self.weights_oc[ioc]

            elif self.metrics == W_absdR_Rprime:
                sumsq += self.weights_oc[ioc]*abs(dR/self.Rprime[ioc_read])
                sumw += self.weights_oc[ioc]

            elif self.metrics == W_absdR2_Rprime2:
                sumsq += self.weights_oc[ioc]*(dR/self.Rprime[ioc_read])**2
                sumw += self.weights_oc[ioc]

            elif self.metrics == W_dR2_Rprime_noglint2:
                sumsq += self.weights_oc[ioc]*(dR/self.Rprime_noglint[ioc_read])**2
                sumw += self.weights_oc[ioc]

            elif self.metrics ==  W_dR2_Rprime_noglint2_norm:
                sumsq += self.weights_oc[ioc]*(dR/self.Rprime_noglint[ioc_read])**2
                sumw += self.weights_oc[ioc]*(0.001/self.Rprime_noglint[ioc_read])**2

        if self.metrics != polymer_3_5:
            sumsq = sumsq/sumw

        if self.constraint_amplitude != 0:
            # sigma equals sigma1 when chl = 0.01
            # sigma equals sigma2 when chl = 0.1
            sigma = self.sigma1*self.sigma1/self.sigma2*exp(log(self.sigma1/self.sigma2)*x[0])

            sumsq += self.constraint_amplitude * (1. - exp(-x[1]*x[1]/(2*sigma*sigma)))

        return sumsq
