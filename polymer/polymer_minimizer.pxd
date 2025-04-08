from polymer.neldermead cimport NelderMeadMinimizer
from polymer.water cimport WaterModel

cdef enum METRICS:
    W_dR2_norm = 1
    W_absdR = 2
    W_absdR_norm = 3
    W_absdR_Rprime = 4
    W_absdR2_Rprime2 = 5
    W_dR2_Rprime_noglint2 = 6
    polymer_3_5 = 7
    W_dR2_Rprime_noglint2_norm = 8

cdef class PolymerMinimizer(NelderMeadMinimizer):
    cdef float[:] Rprime
    cdef float[:] Rprime_noglint
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
    cdef float[:] weights_oc

    cdef METRICS metrics

    cdef int init_pixel(self, float[:] Rprime, float[:] Rprime_noglint,
                   float[:,:] A, float[:,:] pA,
                   float[:] Tmol,
                   float[:] wav, float sza, float vza, float raa, float ws) except -1
    cdef float eval(self, float[:] x)
    cdef float eval_atm(self, float[:] x)