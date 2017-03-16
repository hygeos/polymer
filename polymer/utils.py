#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.ndimage import convolve
from numpy import ones, sqrt, zeros_like, NaN, isnan
from os import system
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import distance_transform_edt

def coeff_sun_earth_distance(jday):
    jday -= 1

    A=1.00014
    B=0.01671
    C=0.9856002831
    D=3.4532858
    E=360.
    F=0.00014

    coef  = 1./((A - B*np.cos(2*np.pi*(C*jday - D)/E) - F*np.cos(4*np.pi*(C*jday - D)/E))**2)

    return coef


def safemove(A, B):
    '''
    safely move a file A to B
    by moving to a temporary file B.tmp first
    '''
    if B+'.tmp' == A:
        # no need for intermediary copy
        cmd = 'mv {} {}'.format(A, B)
        if system(cmd):
            raise IOError('Error executing "{}"'.format(cmd))
    else:
        cmd = 'mv {} {}'.format(A, B+'.tmp')
        if system(cmd):
            raise IOError('Error executing "{}"'.format(cmd))

        cmd = 'mv {} {}'.format(B+'.tmp', B)
        if system(cmd):
            raise IOError('Error executing "{}"'.format(cmd))


def landmask(lat, lon, resolution='l'):
    '''
    returns a land mask for coordinates (lat, lon)
    (1 <=> LAND)

    resolution :     gshhs coastline resolution used to define land/sea
                     mask (default 'l', available 'c','l','i','h' or 'f')

    (uses basemap)
    '''
    from mpl_toolkits.basemap import maskoceans
    landmask = ~maskoceans(lon, lat, zeros_like(lat), resolution=resolution).mask
    return landmask


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
    R[ok] = S2[ok]/N[ok] - (S[ok]/N[ok])**2
    R[R<0] = 0.   # because a few values may be slightly negative
    R[ok] = sqrt(R[ok])
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


def rectBivariateSpline(A, shp):
    '''
    Bivariate spline interpolation of array A to shape shp.

    Fill NaNs with closest values, otherwise RectBivariateSpline gives no
    result.
    '''
    xin = np.arange(shp[0], dtype='float32') / (shp[0]-1) * A.shape[0]
    yin = np.arange(shp[1], dtype='float32') / (shp[1]-1) * A.shape[1]

    x = np.arange(A.shape[0], dtype='float32')
    y = np.arange(A.shape[1], dtype='float32')

    invalid = isnan(A)
    if invalid.any():
        # fill nans
        # see http://stackoverflow.com/questions/3662361/
        ind = distance_transform_edt(invalid, return_distances=False, return_indices=True)
        A = A[tuple(ind)]

    f = RectBivariateSpline(x, y, A)

    return f(xin, yin).astype('float32')

def pstr(x):
    '''
    'pretty' representation of object x as string
    '''
    if isinstance(x, dict):
        s = []
        for k, v in sorted(x.items()):
            s.append('{}: {}'.format(k, v))
        return '{' + ', '.join(s) + '}'
    else:
        return str(x)


def raiseflag(bitmask, flag_value, condition):
    '''
    raise a flag 'flag_value' in binary flags array 'bitmask' where 'condition' is met

    Arguments:
        * bitmask: array to fill
        * flag_value: flag value (should be a power of 2)
        * condition: where to raise the flag
    '''
    notraised = (bitmask & flag_value) == 0
    bitmask[condition & notraised] += flag_value

