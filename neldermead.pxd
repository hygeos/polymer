cdef class NelderMeadMinimizer:
    cdef int N   # number of dimensions
    cdef int niter  # number of iterations
    cdef float[:] fsim
    cdef float[:] xmin
    cdef float[:,:] sim
    cdef float[:,:] ssim
    cdef float[:] xbar
    cdef float[:] y, xcc, xc, xr, xe
    cdef float eval(self, float[:] x) except? -999
    cdef int[:] ind
    cdef float[:] center
    cdef float size(self)
    cdef init(self,
            float[:] x0,
            float[:] dx,
            )
    cdef iterate(self)
    cdef float[:] minimize(self,
                float[:] x0,
                float[:] dx,
                float size_end_iter,
                int maxiter=*)
