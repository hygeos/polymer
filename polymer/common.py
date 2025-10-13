#!/usr/bin/env python
# -*- coding: utf-8 -*-


# Generic flags, not specific to Polymer
L2FLAGS = {
        'LAND'          : 1,
        'CLOUD_BASE'    : 2,
        'L1_INVALID'    : 4,
        }
    
# Polymer-specific flags
L2FLAGS_POLYMER = {
        'OUT_OF_BOUNDS' : 16,
        'EXCEPTION'     : 32,
        'THICK_AEROSOL' : 64,
        'HIGH_AIR_MASS' : 128,
        'EXTERNAL_MASK' : 512,
        'CASE2'         : 1024,
        'INCONSISTENCY' : 2048,
        'ANOMALY_RWMOD_BLUE' : 4096,
        }

