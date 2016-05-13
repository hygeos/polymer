import numpy as np
cimport numpy as np
from cython cimport floating
from libc.math cimport abs


cdef class NelderMeadMinimizer:

    def __init__(self, int N):
        '''
        Initialize the minimizer with a number of dimensions N
        '''
        self.N = N
        self.fsim = np.zeros((N + 1,), dtype='float32')
        self.sim = np.zeros((N + 1, N), dtype='float32')
        self.ssim = np.zeros((N + 1, N), dtype='float32')
        self.xbar = np.zeros(N, dtype='float32')
        self.ind = np.zeros((N + 1,), dtype='int32')
        self.y = np.zeros(N, dtype='float32')
        self.xcc = np.zeros(N, dtype='float32')
        self.xc = np.zeros(N, dtype='float32')
        self.xr = np.zeros(N, dtype='float32')
        self.xe = np.zeros(N, dtype='float32')

    cdef float eval(self, float[:] x) except? -999:
        raise Exception('NelderMeadMinimizer.eval() shall be implemented')

    cdef float [:] minimize(self,
                float [:] x0,
                int maxiter=-1,
                float xtol=1e-4,
                float ftol=1e-4,
                int disp=0):
        """
        Minimization of scalar function of one or more variables using the
        Nelder-Mead algorithm.

        Options for the Nelder-Mead algorithm are:
            disp : int
                Set to True to print convergence messages.
            xtol : float
                Relative error in solution `xopt` acceptable for convergence.
            ftol : float
                Relative error in ``fun(xopt)`` acceptable for convergence.
            maxiter : int
                Maximum number of iterations to perform.

        (from scipy)
        """
        if self.N != len(x0):
            raise Exception('')
        cdef int N = self.N
        if maxiter < 0:
            maxiter = N * 200

        cdef float rho = 1
        cdef float chi = 2
        cdef float psi = 0.5
        cdef float sigma = 0.5
        cdef float nonzdelt = 0.05
        cdef float zdelt = 0.00025
        cdef int stop, k, j
        cdef float[:] y = self.y
        cdef float fxr, fxe, fxc, fxcc

        for j in range(N):
            self.sim[0,j] = x0[j]
        self.fsim[0] = self.eval(x0)
        for k in range(N):
            for j in range(N):
                y[j] = x0[j]
            if y[k] != 0:
                y[k] = (1 + nonzdelt)*y[k]
            else:
                y[k] = zdelt

            for j in range(N):
                self.sim[k + 1, j] = y[j]
            self.fsim[k + 1] = self.eval(y)

        # FIXME
        # ind = np.argsort(self.fsim)
        # print 'VERIF1', ind
        # print np.take(self.fsim, ind, 0)
        # print np.take(self.sim, ind, 0)

        combsort(self.fsim, self.N+1, self.ind)

        # use indices to sort the simulation parameters
        for k in range(self.N+1):
            for j in range(N):
                self.ssim[k,j] = self.sim[self.ind[k],j]
        for k in range(self.N+1):
            for j in range(N):
                self.sim[k,j] = self.ssim[k,j]

        # FIXME
        # print 'VERIF2', np.array(self.ind)
        # print np.array(self.fsim)
        # print np.array(self.sim)


        self.niter = 1

        while self.niter < maxiter:

            stop = 1
            for k in range(1, N):
                if abs(self.fsim[k] - self.fsim[0]) > ftol:
                    stop = 0
                for j in range(N):
                    if abs(self.sim[k,j] - self.sim[0,j]) > xtol:
                        stop = 0
            if stop:
                break

            for j in range(self.N):
                self.xbar[j] = 0.
                for k in range(self.N):
                    self.xbar[j] += self.sim[k,j]
                self.xbar[j] /= N

            # xbar = np.add.reduce(self.sim[:-1], 0) / N  # FIXME
            for k in range(N):
                self.xr[k] = (1 + rho) * self.xbar[k] - rho * self.sim[-1,k]
            fxr = self.eval(self.xr)
            doshrink = 0

            if fxr < self.fsim[0]:
                for k in range(N):
                    self.xe[k] = (1 + rho * chi) * self.xbar[k] - rho * chi * self.sim[-1,k]
                fxe = self.eval(self.xe)

                if fxe < fxr:
                    for k in range(N):
                        self.sim[N,k] = self.xe[k]
                    self.fsim[N] = fxe
                else:
                    for k in range(N):
                        self.sim[N,k] = self.xr[k]
                    self.fsim[N] = fxr
            else:  # fsim[0] <= fxr
                if fxr < self.fsim[N-1]:
                    for k in range(N):
                        self.sim[N,k] = self.xr[k]
                    self.fsim[N] = fxr
                else:  # fxr >= fsim[-2]
                    # Perform contraction
                    if fxr < self.fsim[N]:
                        for k in range(N):
                            self.xc[k] = (1 + psi * rho) * self.xbar[k] - psi * rho * self.sim[-1,k]
                        fxc = self.eval(self.xc)

                        if fxc <= fxr:
                            for k in range(N):
                                self.sim[N, k] = self.xc[k]
                            self.fsim[N] = fxc
                        else:
                            doshrink = 1
                    else:
                        # Perform an inside contraction
                        for k in range(N):
                            self.xcc[k] = (1 - psi) * self.xbar[k] + psi * self.sim[-1,k]
                        fxcc = self.eval(self.xcc)

                        if fxcc < self.fsim[N]:
                            for k in range(N):
                                self.sim[N,k] = self.xcc[k]
                            self.fsim[N] = fxcc
                        else:
                            doshrink = 1

                    if doshrink:
                        for j in range(1, N+1):
                            for k in range(N):
                                self.sim[j,k] = self.sim[0,k] + sigma * (self.sim[j,k] - self.sim[0,k])
                                y[k] = self.sim[j,k]
                            self.fsim[j] = self.eval(y)

            combsort(self.fsim, self.N+1, self.ind)
            # use indices to sort the simulation parameters
            for k in range(self.N+1):
                for j in range(N):
                    self.ssim[k,j] = self.sim[self.ind[k],j]
            for k in range(self.N+1):
                for j in range(N):
                    self.sim[k,j] = self.ssim[k,j]

            # FIXME
            # ind = np.argsort(self.fsim)
            # self.sim = np.take(self.sim, ind, 0)
            # self.fsim = np.take(self.fsim, ind, 0)

            self.niter += 1

        # x = self.sim[0,:]
        # fval = np.min(self.fsim)
        # assert (np.diff(self.fsim) >= 0).all()
        # warnflag = 0

        # if iterations >= maxiter:
            # print 'maxiter'
        # else:
            # print 'success'

        # result = OptimizeResult(fun=fval, nit=iterations, nfev=fcalls[0],
                                # status=warnflag, success=(warnflag == 0),
                                # message=msg, x=x)

        for j in range(N):
            y[j] = self.sim[0,j]
        return y


cdef combsort(float[:] inp, int N, int[:] ind):
    '''
    in-place sort of array inp of size N using comb sort.
    returns sorting indexes in array ind.
    '''
    cdef int gap = N
    cdef int swapped = 0
    cdef float shrink = 1.3
    cdef int i
    cdef int ix
    cdef float tmp
    for i in range(N):
        ind[i] = i

    while not ((gap == 1) and (not swapped)):
        gap = int(gap/shrink)
        if gap < 1:
            gap = 1
        i = 0
        swapped = 0

        while i + gap < N:

            if inp[i] > inp[i+gap]:

                # swap the values in place
                tmp = inp[i+gap]
                inp[i+gap] = inp[i]
                inp[i] = tmp

                # swap also the index
                ix = ind[i+gap]
                ind[i+gap] = ind[i]
                ind[i] = ix

                swapped = 1

            # if inp[i] == inp[i+gap]:
                # swapped = 1

            i += 1

    # verification
    # for i in range(N-1):
        # if inp[i+1] < inp[i]:
            # print 'ERROR'



def test_combsort():
    N = 10
    A = np.random.randn(N).astype('float32')
    AA = A.copy()
    I = np.zeros(N, dtype='int32')
    combsort(A, N, I)
    assert (np.diff(A) >= 0).all()
    assert (AA[I] == A).all()


cdef class Rosenbrock(NelderMeadMinimizer):
    cdef float eval(self, float[:] x) except? -999:
        # rosenbrock function
        return (1-x[0])*(1-x[0]) + 100*(x[1]-x[0]*x[0])*(x[1]-x[0]*x[0])

cdef test_minimize():
    r = Rosenbrock(2)
    for X0 in [
            np.array([0, 0], dtype='float32'),
            np.array([-1, -1], dtype='float32'),
            # np.array([0, -1], dtype='float32'),
            # np.array([10, 0], dtype='float32'),
            # np.array([0, 10], dtype='float32'),
            ]:
        X = np.array(r.minimize(X0))
        assert r.niter > 10
        assert (np.abs(X - 1) < 0.01).all(), (X0, X)



def test():
    '''
    module-wise testing
    '''
    test_combsort()
    test_minimize()

