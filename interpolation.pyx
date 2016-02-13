import numpy as np
cimport numpy as np

cdef class CLUT:

        # TODO:
        # interpolation

        # TODO:
        # index lookup

        # TODO:
        # indices without interpolation ?

        # TODO:
        # how to deal with indices outside range ?
        # for each axis and also (int, sup):
        #      - clip : clip values
        #      - return NaN
        #      - error: raise an error

    cdef float[:] data
    cdef int ndim
    cdef float[:,:] axes   # (axis index, values)
    cdef int[:] shape

    cdef int[:] _index
    cdef int[:] _inf  # index of lower index (interpolation)
    cdef float[:] _x  # fraction for interpolation
    cdef int[:] _dim_interp # indices of the dimensions to interpolate
    cdef int[:] dim_has_axis

    def __init__(self, A, axes=None):
        '''
        axes: list of axes. each axis is either:
                * a list or array of floats (index lookup is activated)
                * None: index lookup is not activated
        '''
        self.shape = np.array(list(A.shape)).astype('int32')
        self.ndim = A.ndim
        self.data = A.ravel(order='C')

        self._index = np.zeros(A.ndim, dtype='int32')
        self._inf = np.zeros(A.ndim, dtype='int32')
        self._x = np.zeros(A.ndim, dtype='float32')
        self._dim_interp = np.zeros(A.ndim, dtype='int32')

        self.dim_has_axis = np.zeros(A.ndim, dtype='int32')
        if axes is None:
            self.axes = np.zeros((A.ndim, 1), dtype='float32')+np.NaN
        else:
            max_axis_size = 1
            for a in axes:
                if (a is not None) and (len(a) > max_axis_size):
                    max_axis_size = len(a)
            self.axes = np.zeros((A.ndim, max_axis_size), dtype='float32')+np.NaN
            for i, a in enumerate(axes):
                if isinstance(a, [np.ndarray, list]):
                    self.axes[i,:len(a)] = a
                    self.dim_has_axis[i] = 1

    cpdef print_info(self):
        pass

    cdef float get(self, int[:] x):
        '''
        Get array value at integer coordinates x
        '''
        cdef int index = x[0]
        cdef int i = 0

        # row-major (C): last dimension is contiguous in memory
        for i in range(1, self.ndim):
            index *= self.shape[i]
            index += x[i]

        return self.data[index]


    cdef float interp(self, float[:] x, int[:] index=None):
        '''
        Interpolate a value in the array at floating point coordinates x
            - if an axis is provided for a given dimension i, performs index lookup
            - otherwise, interpolates using floating-point index
            - if coordinate is < 0, use int index[i]
        '''
        cdef float coef
        cdef float rvalue = 0.
        cdef int j, d, b, D
        cdef n_dim_interp = 0

        for j in range(self.ndim):

            if self.dim_has_axis[j]:
                # index lookup
                # TODO
                n_dim_interp += 1
            elif x[j] >= 0:
                # store the index of the dimension to interpolate
                self._dim_interp[n_dim_interp] = j

                self._inf[j] = <int>x[j]
                self._x[j] = x[j] - self._inf[j]

                # floating index interpolation
                n_dim_interp += 1

            else:
                # int indexing using index
                self._index[j] = index[j]

        # loop over the 2^n dimensions to interpolate
        for j in range(1<<n_dim_interp):

            # calculate the weight of the current item
            coef = 1.
            for d in range(n_dim_interp):
                # number of the current dimension
                D = self._dim_interp[d]

                # b is the value of the 'd'th bit in j (ie corresponding to
                # interpolation dimension number d)
                # it is used to determine if, in the current combination j, the
                # 'd'th dimension is a 'inf' or a 'inf+1'='sup' so as to
                # determine if coef has to be multiplied by x or 1-x
                b = (j & (1<<d))>>d

                self._index[D] = self._inf[D] + b

                if b:
                    coef *= self._x[D]
                else:
                    coef *= 1 - x[D]

            rvalue += coef * self.get(self._index)

        return rvalue


