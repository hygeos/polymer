#!/usr/bin/env python
# -*- coding: utf-8 -*-


L2FLAGS = {
        'LAND'          : 1,
        'CLOUD_BASE'    : 2,
        'L1_INVALID'    : 4,
        'NEGATIVE_BB'   : 8,
        'OUT_OF_BOUNDS' : 16,
        'EXCEPTION'     : 32,
        'THICK_AEROSOL' : 64,
        'HIGH_AIR_MASS' : 128,
        'EXTERNAL_MASK' : 512,
        'CASE2'         : 1024,
        'INCONSISTENCY' : 2048,
        }

# no product (NaN) in case of...
BITMASK_INVALID = 1+2+4+32+512

# recommended pixek rejection test: (bitmask & BITMASK_REJECT) != 0
BITMASK_REJECT = 1023

