#!/usr/bin/env python
# -*- coding: utf-8 -*-


import warnings
warnings.warn("Polymer's main function has been renamed, please use 'from polymer.main import run_atm_corr' instead of 'from polymer.polymer import polymer'")

def polymer(*args, **kwargs):
    '''
    This function is defined for backward compatibility

    Previously, the main polymer function was 'polymer' in module 'polymer.py',
    in package 'polymer'

    Now the main function is 'run_atm_corr' and and the main polymer
    module is 'main.py'
    '''

