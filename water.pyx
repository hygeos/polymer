



cdef class WaterModel:

    cdef float calc_rho(self):
        raise Exception('WaterModel.calc_rho() shall be implemented')


cdef class ParkRuddick(WaterModel):

    def __init__(self):
        pass

    cdef init(self, float [:] wav,
            float sza, float vza, float raa,
            float wind):
        '''
        set the input parameters (wavelength, geometry, wind speed)
        '''

    cdef float calc_rho(self):
        '''
        calculate the reflectance above the water surface
        '''
        # FIXME
        return 0.
