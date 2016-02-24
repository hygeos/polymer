cdef class NelderMeadMinimizer:
    cdef int N   # number of dimensions
    cdef float [:] fsim
    cdef float [:,:] sim
    cdef float[:] y, xcc, xc, xr, xe
    cdef float eval(self, float[:] x) except? -999
    cdef minimize(self,
                float [:] x0,
                int maxiter=*,
                float xtol=*,
                float ftol=*,
                int disp=*)

