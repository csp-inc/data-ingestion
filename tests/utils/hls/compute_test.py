from datetime import datetime as dt

import xarray as xr
from dask.distributed import Future
from distributed.utils_test import *

from utils.hls import compute
from utils.hls.catalog import HLSBand

def test_get_mask():
    qa = xr.DataArray([
        0b00000001,  # cirrus should end up as False
        0b00000010,  # cloud should end up as False
        0b00000100,  # adjacent cloud should end up as False
        0b00001000,  # cloud shadow should end up as False
        0b11000000,  # high impact aerosol quality should end up as False
        0b00001111,  # combination of QA issues should end up as False
        0b01100000,  # water w/ low impact aerosol should end up as True
        0b00000000,  # no quality issues should end up as True
        0b00010000,  # snow/ice should end up as True
    ])
    mask = compute.get_mask(qa)
    xr.testing.assert_equal(mask, xr.DataArray([
        False, False, False, False, False, False, True, True, True
    ]))
    

def test_fetch_band_url():
    url = "https://hlssa.blob.core.windows.net/hls/S30/HLS.S30.T10TET.2019001.v1.4_02.tif"
    band = 'Blue'
    ds = compute.fetch_band_url(band, url, chunks={'band': 1, 'x': 256, 'y': 256})
    assert isinstance(ds, xr.Dataset)
    assert ds.dims == {'x': 3660, 'y': 3660}
    assert ds.attrs['long_name'] == band
    assert isinstance(ds.data_vars[band], xr.DataArray)
    assert set(ds.coords) == {'x', 'y'}
    # https://github.com/pydata/xarray/issues/4784
    assert isinstance(ds.attrs['scale_factor'], float)
    assert isinstance(ds.attrs['add_offset'], float)
    

def test_get_scene_dataset(client):
    scene = "S30/HLS.S30.T10TET.2019001.v1.4"
    sensor = "S"
    bands = [HLSBand.RED, HLSBand.GREEN, HLSBand.BLUE]
    band_names = ["RED", "GREEN", "BLUE"]
    chunks = {}
    future = compute.get_scene_dataset(scene, sensor, bands, band_names,  client, chunks)
    ds = future.result()

    assert isinstance(future, Future)
    assert len(ds.attrs) > 0
    assert set(ds.data_vars.keys()) == set(band_names)
    assert ds.dims == {'x': 3660, 'y': 3660}
    
def test_compute_tile_median():
    band = xr.DataArray(
        [
            [[1, 2, 3], [-1, 5, 6]],
            [[2, 3, 4], [5, -1000, 7]],
            [[3, 4, 5], [6, 7, 10001]],
            [[4, 5, 6], [7, 8, 9]],
        ],
        dims=['time', 'x', 'y'],
        coords={'x': [1, 2], 'y': [1, 2, 3], 'time': [dt(2000, 1, 1), dt(2000, 1, 2), dt(2000, 1, 3), dt(2000, 1, 4)]}
    ).to_dataset(name='band')
    qa_band = xr.DataArray(
        [
            [[0b1, 0, 0], [0, 0, 0]],
            [[0, 0b1, 0], [0, 0, 0]],
            [[0, 0, 0b1], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0]],
        ],
        dims=['time', 'x', 'y'],
        coords={'x': [1, 2], 'y': [1, 2, 3], 'time': [dt(2000, 1, 1), dt(2000, 1, 2), dt(2000, 1, 3), dt(2000, 1, 4)]}
    ).to_dataset(name='qa')
    ds = xr.merge([band, qa_band])
    
    expected_med = xr.DataArray(
        [
            [[3, 4, 4], [6, 7, 7]]
        ],
        dims=['month', 'x', 'y'],
        coords={'x': [1, 2], 'y': [1, 2, 3], 'month': [1]}
    )
    med = compute.compute_tile_median(ds, 'time.month', 'qa')
    assert set(med.data_vars.keys()) == {'band'}    
    xr.testing.assert_equal(med['band'], expected_med)
    