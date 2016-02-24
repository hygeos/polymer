cdef class CLUT:

    # attributes
    cdef float[:] data
    cdef int ndim
    cdef float[:,:] axes   # (axis index, values)
    cdef long int [:,:] invax  # (axis index, indices)
    cdef int[:] shape

    cdef int[:] _index
    cdef int[:] _inf  # index of lower index (interpolation)
    cdef float[:] _x  # fraction for interpolation
    cdef int[:] _dim_interp # indices of the dimensions to interpolate
    cdef int[:] _interp      # whether dimension i is interpolated
    cdef int[:] dim_has_axis
    cdef float[:] scaling  # scaling factor for index lookup (N-1)/(V(N-1) - V(0))
    cdef int[:] clip   # per-axis behaviour for values lookup
    cdef float[:,:] bounds  # lower and upper bounds of each axis
    cdef int[:] reverse   # whether each axis is reversed
    cdef int debug

    # cdef methods
    cdef float get(self, int[:] x)
    cdef set(self, float value, int[:] x)
    cdef int index(self, int i, int j)
    cdef int indexf(self, int i, float x)
    cdef int lookup(self, int i, float v) except -999
    cdef float interp(self)
