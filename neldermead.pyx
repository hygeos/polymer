
import numpy as np
cimport numpy as np
from cython cimport floating


cdef class NelderMeadMinimizer:

    def __init__(self, int N):
        self.N = N
        self.fsim = np.zeros((N + 1,), dtype='float32')
        self.sim = np.zeros((N + 1, N), dtype='float32')
        self.y = np.zeros(N, dtype='float32')
        self.xcc = np.zeros(N, dtype='float32')
        self.xc = np.zeros(N, dtype='float32')
        self.xr = np.zeros(N, dtype='float32')
        self.xe = np.zeros(N, dtype='float32')

    cdef float eval(self, float[:] x):
        raise Exception('NelderMeadMinimizer.eval() shall be implemented')

    cdef minimize(self,
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
        assert self.N == len(x0)
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
        cdef float[:] X0 = x0  # convert to memoryview
        cdef float[:] y = self.y
        cdef float[:] xcc = self.xcc
        cdef float[:] xc = self.xc
        cdef float[:] xr = self.xr
        cdef float[:] xe = self.xe

        self.sim[0,:] = X0
        self.fsim[0] = self.eval(x0)
        for k in range(0, N):
            y[:] = X0[:]
            if y[k] != 0:
                y[k] = (1 + nonzdelt)*y[k]
            else:
                y[k] = zdelt

            self.sim[k + 1] = y
            f = self.eval(y)
            self.fsim[k + 1] = f

        cdef np.ndarray[long, ndim=1] ind = np.argsort(self.fsim)
        self.fsim = np.take(self.fsim, ind, 0)
        # sort so sim[0,:] has the lowest function value
        self.sim = np.take(self.sim, ind, 0)

        cdef int iterations = 1

        while iterations < maxiter:

            stop = 1
            for k in range(1, N):
                if np.abs(self.fsim[k] - self.fsim[0]) > ftol:
                    stop = 0
                for j in range(N):
                    if np.abs(self.sim[k,j] - self.sim[0,j]) > xtol:
                        stop = 0
            if stop:
                break

            xbar = np.add.reduce(self.sim[:-1], 0) / N
            for k in range(N):
                xr[k] = (1 + rho) * xbar[k] - rho * self.sim[-1,k]
            fxr = self.eval(xr)
            doshrink = 0

            if fxr < self.fsim[0]:
                for k in range(N):
                    xe[k] = (1 + rho * chi) * xbar[k] - rho * chi * self.sim[-1,k]
                fxe = self.eval(xe)

                if fxe < fxr:
                    self.sim[-1] = xe
                    self.fsim[-1] = fxe
                else:
                    self.sim[-1] = xr
                    self.fsim[-1] = fxr
            else:  # fsim[0] <= fxr
                if fxr < self.fsim[N-1]:
                    self.sim[-1] = xr
                    self.fsim[N] = fxr
                else:  # fxr >= fsim[-2]
                    # Perform contraction
                    if fxr < self.fsim[N]:
                        for k in range(N):
                            xc[k] = (1 + psi * rho) * xbar[k] - psi * rho * self.sim[-1,k]
                        fxc = self.eval(xc)

                        if fxc <= fxr:
                            self.sim[-1] = xc
                            self.fsim[N] = fxc
                        else:
                            doshrink = 1
                    else:
                        # Perform an inside contraction
                        for k in range(N):
                            xcc[k] = (1 - psi) * xbar[k] + psi * self.sim[-1,k]
                        fxcc = self.eval(xcc)

                        if fxcc < self.fsim[N]:
                            self.sim[-1] = xcc
                            self.fsim[N] = fxcc
                        else:
                            doshrink = 1

                    if doshrink:
                        for j in range(1, N+1):
                            for k in range(N):
                                self.sim[j,k] = self.sim[0,k] + sigma * (self.sim[j,k] - self.sim[0,k])
                            self.fsim[j] = self.eval(self.sim[j])

            ind = np.argsort(self.fsim)
            self.sim = np.take(self.sim, ind, 0)
            self.fsim = np.take(self.fsim, ind, 0)
            iterations += 1

        x = self.sim[0]
        fval = np.min(self.fsim)
        warnflag = 0

        # if iterations >= maxiter:
            # print 'maxiter'
        # else:
            # print 'success'

        # result = OptimizeResult(fun=fval, nit=iterations, nfev=fcalls[0],
                                # status=warnflag, success=(warnflag == 0),
                                # message=msg, x=x)

        return np.asarray(x), iterations, fval


# cdef class Min(NelderMeadMinimizer):
    # cdef float eval(self, float[:] x):
        # return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2

# def test():

    # x0 = np.array([0, 0], dtype='float32')

    # print Min(2).minimize(x0)
