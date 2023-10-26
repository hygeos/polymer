#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pathlib import Path
import pytest
from datetime import datetime, timedelta
from polymer.ancillary import Ancillary_NASA
from polymer.ancillary_era5 import Ancillary_ERA5
from matplotlib import pyplot as plt
from . import conftest
from os import system
from tempfile import TemporaryDirectory
from polymer.ancillary import NonFatalException

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



@pytest.mark.parametrize('url',[
    'https://oceandata.sci.gsfc.nasa.gov/cgi/getfile/GMAO_FP.20231005T090000.MET.NRT.nc', # Available file
    ])
def test_download(url):
    with TemporaryDirectory() as tmpdir:
        tmpfile = Path(tmpdir)/Path(url).name
        ret = Ancillary_NASA().download(url, str(tmpfile))
        print(ret)
        assert ret == 0

@pytest.mark.parametrize('url',[
    'https://oceandata.sci.gsfc.nasa.gov/cgi/getfile/25061439.nc', # 404 Error
    # 'https://oceandata.sci.gsfc.nasa.gov/cgi/getfile/N202000300_O3_AURAOMI_24h.hdf'     , # 403 Error
    ])
def test_download_nofile(url):
    with TemporaryDirectory() as tmpdir:
        tmpfile = Path(tmpdir)/Path(url).name
        ret = Ancillary_NASA().download(url, str(tmpfile))
        assert ret == 1

@pytest.mark.parametrize('valid_auth',[True, False])
def test_download_auth(valid_auth):
    url = 'https://oceandata.sci.gsfc.nasa.gov/cgi/getfile/GMAO_FP.20231005T090000.MET.NRT.nc'
    with TemporaryDirectory() as tmpdir:
        tmpfile = Path(tmpdir)/'test_auth.tmp'
        if valid_auth:
            cmd = 'wget -nv --save-cookies ~/.urs_cookies --keep-session-cookies --auth-no-challenge {} -O {}'.format(url, tmpfile)
            assert system(cmd) == 0
        else:
            cmd = 'wget -nv --save-cookies ~/.urs_cookies --keep-session-cookies --user user --password pass --auth-no-challenge {} -O {}'.format(url, tmpfile)
            assert system(cmd) != 0