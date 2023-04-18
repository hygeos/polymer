#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pytest
from datetime import datetime, timedelta
from polymer.ancillary import Ancillary_NASA
from polymer.ancillary_era5 import Ancillary_ERA5
from matplotlib import pyplot as plt
from . import conftest

@pytest.mark.parametrize('variable,typ_value', [
    ('wind_speed', 10),
    ('surf_press', 1013),
    ('ozone', 400.),
])
@pytest.mark.parametrize('offset,allow_forecast', [  # offset=number of days
    (0, True),
    (3, False),
    (20, False),
])
@pytest.mark.parametrize('mode', ['NASA', 'ERA5'])
def test_ancillary(request, variable, typ_value, offset, allow_forecast, mode):
    if mode == 'NASA':
        anc = Ancillary_NASA(allow_forecast=allow_forecast)
    elif mode == 'ERA5':
        anc = Ancillary_ERA5()
    else :
        raise ValueError(mode)
        
    ret = anc.get(variable, datetime.now() - timedelta(days=offset))
    print(ret)
    print(ret.date)
    print(ret.filename)

    assert ret.data.data.mean() < typ_value*1.5
    assert ret.data.data.mean() > typ_value*0.5

    plt.figure()
    plt.imshow(ret.data.data)
    plt.colorbar()
    conftest.savefig(request)
