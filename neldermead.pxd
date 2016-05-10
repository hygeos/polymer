cdef class NelderMeadMinimizer:
    cdef int N   # number of dimensions
    cdef int niter  # number of iterations
    cdef float [:] fsim
    cdef float [:,:] sim
    cdef float [:,:] ssim
    cdef float[:] y, xcc, xc, xr, xe
    cdef float eval(self, float[:] x) except? -999
    cdef int[:] ind
    cdef float [:] minimize(self,
                float [:] x0,
                int maxiter=*,
                float xtol=*,
                float ftol=*,
                int disp=*)

