cdef class WaterModel:
    cdef float SPM
    cdef int init(self, float[:] wav, float sza, float vza, float raa)
    cdef float[:] calc_rho(self, float[:] x)
