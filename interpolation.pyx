import numpy as np
cimport numpy as np

cdef class CLUT:

    cdef float[:] data
    cdef int ndim
    cdef float[:,:] axes   # (axis index, values)
    cdef long int [:,:] invax  # (axis index, indices)
    cdef int[:] shape

    cdef int[:] _index
    cdef int[:] _inf  # index of lower index (interpolation)
    cdef float[:] _x  # fraction for interpolation
    cdef int[:] _dim_interp # indices of the dimensions to interpolate
    cdef int[:] dim_has_axis
    cdef float[:] scaling  # scaling factor for index lookup (N-1)/(V(N-1) - V(0))
    cdef int[:] clip   # per-axis behaviour for values lookup
    cdef float[:,:] bounds  # lower and upper bounds of each axis
    cdef int debug

    def __init__(self, A, axes=None, clip=None, debug=False):
        '''
        A: N-dim array
        axes: list of axes. each axis is either:
                * a list or array of floats (index lookup is activated)
                * None: index lookup is not activated
        clip: behaviour of lookup for out-of-bounds values
              list or array of N values:
                - 0: out-of-bounds returns -1 (failure)
                - 1: clip lower end
                - 2: clip higher end
                - 3: clip both ends
        '''
        self.shape = np.array(list(A.shape)).astype('int32')
        self.ndim = A.ndim
        self.data = A.ravel(order='C')
        self.debug = debug

        self._index = np.zeros(A.ndim, dtype='int32')
        self._inf = np.zeros(A.ndim, dtype='int32')
        self._x = np.zeros(A.ndim, dtype='float32')
        self._dim_interp = np.zeros(A.ndim, dtype='int32')
        self.scaling = np.zeros(A.ndim, dtype='float32')
        self.bounds = np.zeros((A.ndim, 2), dtype='float32')

        self.dim_has_axis = np.zeros(A.ndim, dtype='int32')
        if axes is None:
            axes = [None] * A.ndim
        if clip is None:
            clip = [0] * A.ndim
        self.clip = np.array(clip, dtype='int32')

        max_axis_size = 1
        max_invax_size = 1
        Nmin = 5.   # at least 5 bins per element
        for a in axes:
            if a is None:
                continue
            if (len(a) > max_axis_size):
                max_axis_size = len(a)
            deltamin = np.amin(np.diff(a))/float(abs(a[0] - a[-1]))
            if Nmin/deltamin > max_invax_size:
                max_invax_size = Nmin/deltamin

        # set up the axes
        # and the inverted axes for faster inversion
        ax = np.zeros((A.ndim, max_axis_size), dtype='float32')+np.NaN
        iax = np.zeros((A.ndim, max_invax_size), dtype='int64') - 999
        for i, a in enumerate(axes):
            if a is None:
                continue
            assert isinstance(a, (np.ndarray, list))
            ax[i,:len(a)] = a
            self.dim_has_axis[i] = 1

            reverse = a[-1] < a[0] # whether axis is reversed

            if reverse:
                a = a[::-1]

            self.bounds[i,0] = a[0]
            self.bounds[i,1] = a[-1]

            # inverted axis
            v = np.linspace(a[0], a[-1], max_invax_size, dtype='float64')
            iax[i, :] = np.searchsorted(a, v)-1
            iax[i, 0] = 0.
            # set the last item of each consecutive series to -1
            # (uncertain bracketing)
            iax[i, :-1][np.diff(iax[i,:]) != 0] = -1

            if reverse:
                iax[i,iax[i,:]>=0] = len(A) - 2 - iax[i,iax[i,:]>=0]

            # scaling factor for quick index lookup
            self.scaling[i] = (max_invax_size-1)/float(abs(a[-1]-a[0]))

        self.axes = ax
        self.invax = iax


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


    cdef set(self, float value, int[:] x):
        '''
        Set value at coordinates x
        '''
        cdef int index = x[0]
        cdef int i = 0

        # row-major (C): last dimension is contiguous in memory
        for i in range(1, self.ndim):
            index *= self.shape[i]
            index += x[i]

        self.data[index] = value


    cdef long int lookup(self, int i, float v):
        '''
        index lookup for axis i with value v:
        returns lowest index j such that v[j] < v
        if outside bounds and not clip, returns -1 when below-axis and -2 when
        above-axis (in terms of values)
        '''
        cdef long int j, jj

        # lower end clipping
        # values below axis
        if v < self.bounds[i,0]:
            if self.clip[i] & 1 != 0:
                return 0
            else:
                return -1

        # higher end clipping
        if v > self.bounds[i,1]:
            if self.clip[i] & 2 != 0:
                return self.shape[i]-2
            else:
                return -2

        j = <long int>((v - self.bounds[i,0])*self.scaling[i])
        # print 'j is', j
        jj = self.invax[i,j]
        # print 'jj is', jj
        if (jj < 0):
            # print 'DEBUG', j, jj
            if v <= self.axes[i, self.invax[i, j-1]]:
                jj = self.invax[i, j-1]
            else:
                jj = self.invax[i, j+1]

        if self.debug:
            # verifications
            if jj<0:
                raise Exception('Consistency error in lookup, jj={}'.format(jj))
            if not ((self.axes[i,jj] - v) * (self.axes[i,jj+1] - v) < 0):
                raise Exception('Could not verify {} between {} and {}'.format(
                    self.axes[i,jj], v, self.axes[i,jj+1]))

        return jj


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
                self._inf[j] = self.lookup(j, x[j])
                self._x[j] = x[j] - self._inf[j]
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


