#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pytest
from datetime import datetime, timedelta
from polymer.ancillary import Ancillary_NASA
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
    (100, False),
])
def test_ancillary(request, variable, typ_value, offset, allow_forecast):
    anc = Ancillary_NASA(allow_forecast=allow_forecast)
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
