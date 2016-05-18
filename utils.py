#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scipy.ndimage import convolve
from numpy import ones, sqrt


def stdev(S, S2, N):
    '''
    Returns standard deviation from:
        * S sum of the values
        * S2 sum of the squared values
        * N number of values
    '''

    return sqrt(S2/N - (S/N)**2)


def stdNxN(X, N, mask=None):
    '''
    Standard deviation over NxN blocks over array X
    '''

    if mask == None:
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
    return stdev(S, S2, C)

