from numpy import pi

def toa_uncertainties(block):
    Rtoa_var = (block.Ltoa/block.Ltyp) * (pi*block.sigma_typ/(block.F0*block.mus[...,None]))**2
    block.Rtoa_var = Rtoa_var.astype('float32')