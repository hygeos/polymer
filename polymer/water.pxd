cdef class WaterModel:
    cdef float SPM
    cdef int init_pixel(self, float[:] wav, float sza, float vza, float raa, float ws) except -1
    cdef float[:] calc_rho(self, float[:] x)
