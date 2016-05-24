#!/usr/bin/env python
# -*- coding: utf-8 -*-


from numpy import pi, sin, exp, sqrt, cos


def glitter(wind, mu_s, mu_v, gamma, phi=None, phi_vent=None):
    '''
    This function computes glitter's normalized radiance from
    Cox and Munk wave slope probability density function model,
    for isotropic or directionnal approximation

    Arguments:
          float wind     [IN]     : Norm of the wind surface vector [m/s]
          float mu_s     [IN]     : Cosine of the solar zenith angle
          float mu_v     [IN]     : Cosine of the viewing zenith angle
          float gamma    [IN]     : Scattering angle (degrees)
         Directional approximation:
         (not used if phi and phi_vent are None (default))
          float phi      [IN]     : Relative azimutal angle between solar azimtual
                                    angle and viewing azimutal angle (degrees)
          float phi_vent [IN]     : Relative azimutal angle between solar azimutal
                                    angle and wind azimutal angle (degrees)
    '''

    assert (phi is None) == (phi_vent is None)

    NH2O = 1.34
    SIGWA = 0.003
    SIGWB = 0.00512
    SIGCB = 0.003
    SIGCA = 0.00193
    SIGUB = 0.
    SIGUA = 0.00316
    C21B = 0.01
    C21A = -0.0086
    C03B = 0.04
    C03A = -0.033
    C40 = 0.40
    C22 = 0.12
    C04 = 0.23


    # Compute cosine of the wave slope inclination beta
    x = sin(gamma * 0.5 * (pi/180.))
    cosbeta = (mu_s + mu_v) / (2.0 * x)

    # Compute Fresnel coefficients useful to know specular radiance
    y = sqrt(1.0 - ((1.0 - x * x) / (NH2O * NH2O)))

    a = ((y - x * NH2O) / (y + x * NH2O));
    b = ((x - y * NH2O) / (x + y * NH2O));

    # Compute specular radiance for air/water interface, given by Fresnel law
    reflectspecu = 0.5 * ( a * a + b * b );

    # Compute wave slope distribution

    if (phi is None): # Isotropic approximation case

        # Compute sig2 expressed as s function of wind vector norm [m/s]
        sig2 = SIGWA + SIGWB * wind;

        # Cox and Munk formula
        pentevague = exp((cosbeta * cosbeta - 1.0) / (sig2 * cosbeta * cosbeta)) / sig2

    else: # Directionnal case

     sintetav = sqrt(1.0 - mu_v * mu_v)
     sintetas = sqrt(1.0 - mu_s * mu_s)

     zx = ((-sintetav * cos(phi * (pi/180.))) - sintetas) / (mu_v + mu_s)

     zy = (sintetav * sin(phi * (pi/180.))) / (mu_v + mu_s)

     zc = ( zy * cos(phi_vent * (pi/180.))) + (zx * sin(phi_vent * (pi/180.)))
     zu = (-zx * cos(phi_vent * (pi/180.))) + (zy * sin(phi_vent * (pi/180.)))

     sigc = sqrt(SIGCB + SIGCA * wind)
     sigu = sqrt(SIGUB + SIGUA * wind)

     c21 = C21B + C21A * wind
     c03 = C03B + C03A * wind

     x = zc/sigc
     y = zu/sigu

     # Cox and Munk formula
     pentevague = ((1.0 / (2.0 * sigc * sigu)) * exp(-(x * x + y * y)/2.0)
                * (1.0 - (c21 * (x * x-1.0) * y / 2.0) - (c03 * ( y * y * y - 3.0 * y ) / 6.0)
                + (C40*(x * x * x *x - 6.0 * x * x + 3.0) / 24.0)
                + (C22*(x * x - 1.0) * (y * y - 1.0) / 4.0)
                + (C04*(y * y * y *y - 6.0 * y * y + 3.0) / 24.0)))


    # Compute glitter's normalized radiance
    return (pentevague * reflectspecu / (4.0 * cosbeta * cosbeta * cosbeta * cosbeta * mu_v))

