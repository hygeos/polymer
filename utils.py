#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scipy.ndimage import convolve
from numpy import ones, sqrt, zeros_like, NaN


def stdev(S, S2, N, fillv=NaN):
    '''
    Returns standard deviation from:
        * S sum of the values
        * S2 sum of the squared values
        * N number of values
    The values where N=0 are filled with fillv
    '''

    R = zeros_like(S) + fillv
    ok = N != 0
    R[ok] = sqrt(S2[ok]/N[ok] - (S[ok]/N[ok])**2)
    return R


def stdNxN(X, N, mask=None, fillv=NaN):
    '''
    Standard deviation over NxN blocks over array X
    '''

    if mask is None:
        M = 1.
    else:
        M = mask

    # kernel
    ker = ones((N,N))

    # sum of the values
    S = convolve(X*M, ker, mode='constant', cval=0)

    # sum of the squared values
    S2 = convolve(X*X*M, ker, mode='constant', cval=0)

    # number of values
    C = convolve(ones(X.shape)*M, ker, mode='constant', cval=0)

    # result
    return stdev(S, S2, C, fillv=fillv)

