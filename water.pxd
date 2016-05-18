cdef class WaterModel:
    cdef init(self, float[:] wav, float sza, float vza, float raa)
    cdef float[:] calc_rho(self, float[:] x)
