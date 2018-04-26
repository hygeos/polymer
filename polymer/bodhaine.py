#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
Rayleigh Optical Depth calculation from Bodhaine, 99

rod(lam, co2, lat, z, P)
'''

from scipy.constants import codata
import numpy as np


def FN2(lam):
    """
    depolarisation factor of N2
        lam : um
    """
    return 1.034 + 3.17 *1e-4 *lam**(-2)


def FO2(lam):
    """
    depolarisation factor of O2
        lam : um
    """
    return 1.096 + 1.385 *1e-3 *lam**(-2) + 1.448 *1e-4 *lam**(-4)


def Fair(lam, co2):
    """
    depolarisation factor of air for CO2
        lam : um
        co2 : ppm
    """
    _FN2 = FN2(lam)
    _FO2 = FO2(lam)

    return ((78.084 * _FN2 + 20.946 * _FO2 + 0.934 +
            co2*1e-4 *1.15)/(78.084+20.946+0.934+co2*1e-4))


def n300(lam):
    """
    index of refraction of dry air (300 ppm CO2)
        lam : um
    """
    return 1e-8 * ( 8060.51 + 2480990/(132.274 - lam**(-2)) + 17455.7/(39.32957 - lam**(-2))) + 1.


def n_air(lam, co2):
    """
    index of refraction of dry air
        lam : um
        co2 : ppm
    """
    return ((n300(lam) - 1) * (1 + 0.54*(co2*1e-6 - 0.0003)) + 1.)

def ma(co2):
    """
    molecular volume
        co2 : ppm
    """
    return 15.0556 * co2*1e-6 + 28.9595

def raycrs(lam, co2):
    """
    Rayleigh cross section
        lam : um
        co2 : ppm
    """
    Avogadro = codata.value('Avogadro constant')
    Ns = Avogadro/22.4141 * 273.15/288.15 * 1e-3
    nn2 = n_air(lam, co2)**2
    return (24*np.pi**3 * (nn2-1)**2/(lam*1e-4)**4/Ns**2/(nn2+2)**2 * Fair(lam, co2))

def g0(lat):
    """
    gravity acceleration at the ground
        lat : deg
    """
    return (980.6160 * (1. - 0.0026372 * np.cos(2*lat*np.pi/180.)
            + 0.0000059 * np.cos(2*lat*np.pi/180.)**2))

def g(lat, z) :
    """
    gravity acceleration at altitude z
        lat : deg (scalar)
        z : m
    """
    return (g0(lat) - (3.085462 * 1.e-4 + 2.27 * 1.e-7 * np.cos(2*lat*np.pi/180.)) * z
            + (7.254 * 1e-11 + 1e-13 * np.cos(2*lat*np.pi/180.)) * z**2
            - (1.517 * 1e-17 + 6 * 1e-20 * np.cos(2*lat*np.pi/180.)) * z**3)

def rod(lam, co2=400., lat=45., z=0., P=1013.25):
    """
    Rayleigh optical depth calculation for Bodhaine, 99
        lam : um
        co2 : ppm
        lat : deg (scalar)
        z : m
        P : hPa

    Example: rod(0.4, 400., 45., 0., 1013.25)
    """
    Avogadro = codata.value('Avogadro constant')
    G = g(lat, z)
    return raycrs(lam, co2) * P*1e3 * Avogadro/ma(co2)/G


