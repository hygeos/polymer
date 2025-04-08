import numpy as np
from numpy.linalg import inv
import pandas as pd
from pathlib import Path



def atm_func(
        wav,
        Rmol,
        Tmol,
        Rgli,
        air_mass,
        params,
        bands):
    '''
    Returns the matrix of coefficients for the atmospheric function
    A (x, y, nlam, ncoef)

    Ratm = A.C
    Ratm: (x, y, nlam)
    A   : (x, y, nlam, ncoef)
    C   : (x, y, ncoef)

    Arguments:  # FIXME:
    ----------
    wav: per-pixel wavelength (x, y, nlam)
    Rmol: molecular reflectance, without glint (x, y, nlam)
    Rgli: sun glint reflectance (x, y)
    air_mass (x, y)
    bands: spectral bands used in the model (nlam)

    # TODO: apply band subsetting upstream ?
    '''
    # bands for atmospheric fit
    Nlam = len(bands)
    assert Nlam
    shp = Rgli.shape
    assert wav.shape[-1] == len(params.bands_read())
    Ncoef = params.Ncoef   # number of polynomial coefficients
    assert Ncoef > 0

    # convert the memoryviews to numpy arrays
    wav = np.array(wav)
    Rmol = np.array(Rmol)
    Tmol = np.array(Tmol)
    Rgli = np.array(Rgli)
    air_mass = np.array(air_mass)

    # correction bands wavelengths
    idx = np.searchsorted(params.bands_read(), bands)
    # transpose: move the wavelength dimension to the end
    lam = wav[:,:,idx]

    # initialize the matrix for inversion

    # FIXME: should use the exact wavelength
    taum = 0.00877*((np.array(params.bands_read())[idx]/1000.)**(-4.05))
    Rgli0 = 0.02
    T0 = np.exp(-taum*((1-0.5*np.exp(-Rgli/Rgli0))*air_mass)[:,:,None])

    if 'veg' in params.atm_model:
        veg = pd.read_csv(
            Path(params.dir_common)/'vegetation.grass.avena.fatua.vswir.vh352.ucsb.asd.spectrum.txt',
            skiprows=21,
            sep=None,
            names=['wav_um', 'r_percent'],
            index_col=0,
            engine='python').to_xarray()
        veg_interpolated = (veg.r_percent/100.).interp(wav_um=lam.ravel()/1000.).values.reshape(lam.shape)

    if params.atm_model == 'T0,-1,-4':
        assert Ncoef == 3
        A = np.zeros((shp[0], shp[1], Nlam, Ncoef), dtype='float32')
        A[:,:,:,0] = T0*(lam/1000.)**0.
        A[:,:,:,1] = (lam/1000.)**-1.
        A[:,:,:,2] = (lam/1000.)**-4.
    elif params.atm_model == 'T0,-1,Rmol':
        assert Ncoef == 3
        A = np.zeros((shp[0], shp[1], Nlam, Ncoef), dtype='float32')
        A[:,:,:,0] = T0*(lam/1000.)**0.
        A[:,:,:,1] = (lam/1000.)**-1.
        A[:,:,:,2] = Rmol[:,:,idx]
    elif params.atm_model == 'T0,-1':
        assert Ncoef == 2
        A = np.zeros((shp[0], shp[1], Nlam, Ncoef), dtype='float32')
        A[:,:,:,0] = T0*(lam/1000.)**0.
        A[:,:,:,1] = (lam/1000.)**-1.
    elif params.atm_model == 'T0,-2':
        assert Ncoef == 2
        A = np.zeros((shp[0], shp[1], Nlam, Ncoef), dtype='float32')
        A[:,:,:,0] = T0*(lam/1000.)**0.
        A[:,:,:,1] = (lam/1000.)**-2.
    elif params.atm_model == 'T0,-1,veg':
        assert Ncoef == 3
        A = np.zeros((shp[0], shp[1], Nlam, Ncoef), dtype='float32')
        A[:,:,:,0] = T0*(lam/1000.)**0.
        A[:,:,:,1] = (lam/1000.)**-1.
        A[:,:,:,2] = Tmol[:,:,idx] * veg_interpolated# * (lam/1000)**-4.
    elif params.atm_model == 'T0,-1,-4,veg':
        assert Ncoef == 4
        A = np.zeros((shp[0], shp[1], Nlam, Ncoef), dtype='float32')
        A[:,:,:,0] = T0*(lam/1000.)**0.
        A[:,:,:,1] = (lam/1000.)**-1.
        A[:,:,:,2] = (lam/1000.)**-4.
        A[:,:,:,3] = Tmol[:,:,idx] * veg_interpolated# * (lam/1000)**-4.
    else:
        raise Exception('Invalid atmospheric model "{}"'.format(params.atm_model))

    return A


def pseudoinverse(A):
    '''
    Calculate the pseudoinverse of array A over the last 2 axes
    (broadcasting the first axes)
    A* = ((A'.A)^(-1)).A'
    where X' is the transpose of X and X^-1 is the inverse of X

    shapes: A:  [...,i,j]
            A*: [...,j,i]
    '''

    # B = A'.A (with broadcasting)
    B = np.einsum('...ji,...jk->...ik', A, A)

    # check
    # if B.ndim == 4:
        # assert np.allclose(B[0,0,:,:], A[0,0,:,:].transpose().dot(A[0,0,:,:]), equal_nan=True)
        # assert np.allclose(B[-1,0,:,:], A[-1,0,:,:].transpose().dot(A[-1,0,:,:]), equal_nan=True)

    # (B^-1).A' (with broadcasting)
    pA = np.einsum('...ij,...kj->...ik', inv(B), A)

    # check
    # if B.ndim == 4:
        # assert np.allclose(pA[0,0], inv(B[0,0,:,:]).dot(A[0,0,:,:].transpose()), equal_nan=True)
        # assert np.allclose(pA[-1,0], inv(B[-1,0,:,:]).dot(A[-1,0,:,:].transpose()), equal_nan=True)

    return pA


def weighted_pseudoinverse(A, W):
    '''
    Calculate the weighted pseudoinverse of array A over the last 2 axes
    (broadcasting the first axes)
    W is the weight matrix (diagonal)
    A* = ((A'.W.A)^(-1)).A'.W
    '''
    assert W.dtype == 'float32'

    # A'.W.A
    B = np.einsum('...ji,...jk,...kl->...il', A, W, A)

    # (B^-1).A'.W
    pA = np.einsum('...ij,...kj,...kl->...il', inv(B), A, W)

    return pA
