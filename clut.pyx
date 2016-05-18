import numpy as np
cimport numpy as np
from libc.math cimport isnan

cdef class CLUT:

    def __init__(self, A, axes=None, debug=False):
        '''
        A: N-dim array
        axes: list of axes. each axis is either:
                * a list or array of floats (index lookup is activated)
                * None: index lookup is not activated
        '''
        cdef int i
        self.shape = np.array(list(A.shape)).astype('int32')
        self.ndim = A.ndim
        self.data = A.astype('float32').ravel(order='C')
        self.debug = debug

        self._index = np.zeros(A.ndim, dtype='int32')-999
        self._inf = np.zeros(A.ndim, dtype='int32')
        self._x = np.zeros(A.ndim, dtype='float32')
        self._dim_interp = np.zeros(A.ndim, dtype='int32')
        self._interp = np.zeros(A.ndim, dtype='int32')
        self.scaling = np.zeros(A.ndim, dtype='float32')
        self.reverse = np.zeros(A.ndim, dtype='int32')
        self.bounds = np.zeros((A.ndim, 2), dtype='float32')

        self.dim_has_axis = np.zeros(A.ndim, dtype='int32')
        if axes is None:
            axes = [None] * A.ndim

        max_axis_size = 1
        Ninv = 1
        Nmin = 3.   # at least Nmin bins per element
        for a in axes:
            if a is None:
                continue
            if (len(a) > max_axis_size):
                max_axis_size = len(a)
            deltamin = np.amin(np.abs(np.diff(a)))/float(abs(a[0] - a[-1]))
            if int(Nmin/deltamin) > Ninv:
                Ninv = int(Nmin/deltamin)

        # set up the axes
        # and the inverted axes for faster inversion
        ax = np.zeros((A.ndim, max_axis_size), dtype='float32')+np.NaN
        iax = np.zeros((A.ndim, Ninv), dtype='int64') - 999
        for i, a in enumerate(axes):
            if a is None:
                continue
            assert isinstance(a, (np.ndarray, list))
            if len(a) != self.shape[i]:
                raise Exception('CLUT: shape mismatch for axis {}'.format(i))
            ax[i,:len(a)] = a
            self.dim_has_axis[i] = 1

            self.reverse[i] = a[-1] < a[0] # whether axis is reversed

            # if axis sorted in decreasing order, sort it in ascending order first
            if self.reverse[i]:
                a = a[::-1]

            self.bounds[i,0] = a[0]
            self.bounds[i,1] = a[-1]

            # inverted axis
            v = np.linspace(a[0], a[-1], Ninv+1, dtype='float64')[:-1]
            iax[i, :] = np.searchsorted(a, v)-1
            iax[i, 0] = 0.
            # set the last item of each consecutive series to -1
            # (uncertain bracketing)
            iax[i,:-1][np.diff(iax[i,:]) != 0] = -1

            # if self.reverse[i]:
                # iax[i,iax[i,:]>=0] = len(a) - 2 - iax[i,iax[i,:]>=0]

            # scaling factor for quick index lookup
            self.scaling[i] = Ninv/float(a[-1]-a[0])

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
        Set value at integer coordinates x
        '''
        cdef int index = x[0]
        cdef int i = 0

        # row-major (C): last dimension is contiguous in memory
        for i in range(1, self.ndim):
            index *= self.shape[i]
            index += x[i]

        self.data[index] = value


    cdef int index(self, int i, int j):
        '''
        set current index on dimension i using integer indexing
        (no interpolation)
        '''
        self._inf[i] = j
        self._interp[i] = 0

        return 0


    cdef int indexf(self, int i, float x):
        '''
        set current index of dimension i using floating index
        (interpolation)
        '''
        self._inf[i] = <int>x
        self._x[i] = x - self._inf[i]
        self._interp[i] = 1

        return 0


    cdef int lookup(self, int i, float v) except -999:
        '''
        index lookup for axis i with value v:
        sets up index j such that v[j] < v and interpolation ratio

        returns:
            0 on success
            -1 in case of lower out-of-bounds
            1 in case of upper out-of-bounds

        in case of out-of-bounds, set up the axes by clipping
        '''
        cdef long int j, jj
        cdef float lower, upper

        if isnan(v):
            raise Exception('lookup of NaN')

        if not self.dim_has_axis[i]:
            raise Exception('Trying to use index lookup without associated axis')

        if not self.reverse[i]:
            # lower end clipping
            if v < self.bounds[i,0]:
                self._inf[i] = 0
                self._interp[i] = 0
                return -1
            # higher end clipping
            if v > self.bounds[i,1]:
                self._inf[i] = self.shape[i]-1
                self._interp[i] = 0
                return 1
        else:
            # lower end clipping
            if v > self.bounds[i,1]:
                self._inf[i] = 0
                self._interp[i] = 0
                return -1
            # higher end clipping
            if v < self.bounds[i,0]:
                self._inf[i] = self.shape[i]-1
                self._interp[i] = 0
                return 1

        self._interp[i] = 1

        # index in the lookup array
        j = <long int>((v - self.bounds[i,0])*self.scaling[i])
        if (j < 0) or (j > self.invax.shape[1]):
            raise Exception('Index error in lookup: index={}, value={}'.format(j, v))
        if j == self.invax.shape[1]:
            j -= 1

        # index in the array
        jj = self.invax[i,j]

        if (jj < 0):
            if v > self.axes[i, self.invax[i, j+1]]:
                jj = self.invax[i, j+1]
            else:
                jj = self.invax[i, j-1]

        if self.reverse[i]:
            jj = self.shape[i] - 2 - jj

        if self.debug:
            # verifications
            if jj<0:
                raise Exception('Consistency error in lookup, jj={}'.format(jj))
            if not ((self.axes[i,jj] - v) * (self.axes[i,jj+1] - v) < 0):
                raise Exception('Could not verify {} between {} and {}'.format(
                    v, self.axes[i,jj], self.axes[i,jj+1]))

        self._inf[i] = jj
        if self._inf[i] < 0:
            raise Exception('Error: negative index on axis {} (index is {}, value is {})'.format(
                i, self._inf[i], v))

        lower = self.axes[i, self._inf[i]]
        upper = self.axes[i, self._inf[i]+1]
        self._x[i] = (v - lower)/(upper - lower)

        return 0

    cdef float interp(self):
        '''
        Interpolate a value in the array at floating point coordinates x
            - if an axis is provided for a given dimension i, performs index lookup
            - otherwise, interpolates using floating-point index
            - if coordinate is < 0, use int index[i]
        '''
        cdef float coef
        cdef float rvalue = 0.
        cdef int j, d, b, D
        cdef int n_dim_interp = 0

        for j in range(self.ndim):
            if self._interp[j]:
                self._dim_interp[n_dim_interp] = j
                n_dim_interp += 1
            else:
                self._index[j] = self._inf[j]

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
                    coef *= 1 - self._x[D]

            rvalue += coef * self.get(self._index)

        return rvalue


def test_get():
    cdef CLUT A = CLUT(np.array([[1, 2, 3], [4, 5, 6]]), axes=[None, None])

    assert A.get(np.array([0, 0], dtype='int32')) == 1
    assert A.get(np.array([1, 1], dtype='int32')) == 5

def test_lookup():

    cdef CLUT A = CLUT(np.array([[1, 2, 3], [4, 5, 6]]), axes=[[1, 2], [10, 11, 12]])

    assert A.lookup(0, 1.5) == 0
    assert A.lookup(0, 0.5) < 0
    assert A.lookup(0, 2.5) > 0

    assert A.lookup(1, 11.5) == 0
    assert A.lookup(1, 0.) < 0
    assert A.lookup(1, 100.) > 0

def test_interp():
    cdef CLUT A = CLUT(np.array([[1, 2, 3], [4, 5, 6]]), axes=[[1, 2], [10, 11, 12]])

    assert A.lookup(0, 1.5) == 0
    assert A.lookup(1, 11.5) == 0

    print A.interp()
    assert A.interp() == 4

def test():
    '''
    test this module
    '''
    test_get()
    test_interp()

