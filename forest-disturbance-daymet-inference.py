import os
import tempfile
import urllib

import fsspec
import pandas as pd
import rioxarray as rioxr
import xarray as xr

temp_dir = tempfile.gettempdir()
os.environ['ACCOUNT_KEY'] = ""

def download_url(url, destination_filename=None, progress_updater=None, force_download=False):
    """
    Download a URL to a temporary file
    """
    
    # This is not intended to guarantee uniqueness, we just know it happens to guarantee
    # uniqueness for this application.
    if destination_filename is None:
        url_as_filename = url.replace('://', '_').replace('/', '_')    
        destination_filename = \
            os.path.join(temp_dir,url_as_filename)
    if (not force_download) and (os.path.isfile(destination_filename)):
        print('Bypassing download of already-downloaded file {}'.format(
            os.path.basename(url)))
        return destination_filename
    print('Downloading file {} to {}'.format(os.path.basename(url),
                                             destination_filename),end='')
    urllib.request.urlretrieve(url, destination_filename, progress_updater)  
    assert(os.path.isfile(destination_filename))
    nBytes = os.path.getsize(destination_filename)
    print('...done, {} bytes.'.format(nBytes))
    return destination_filename


def prep_daymet_tile(ds, tile_name):
    path = fsspec.get_mapper(
        f"az://fia/hls/2018.0/{tile_name}.zarr",
        account_name="usfs",
        account_key=os.environ['ACCOUNT_KEY']
    )
    hls_tile = xr.open_zarr(path, chunks='auto')
    hls_crs = hls_tile.crs
    hls_tile = hls_tile['BLUE'].sel(month=1)
    hls_crs_bounds = hls_tile.rio.bounds()
    hls_tile = hls_tile.rio.reproject('+proj=lcc +lat_1=25 +lat_2=60 +lat_0=42.5 +lon_0=-100 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs')

    output_path = fsspec.get_mapper(
        f"az://fia/daymet/{tile_name}.zarr",
        account_name="usfs",
        account_key=os.environ['ACCOUNT_KEY']
    )
    ds \
        .rio.pad_box(*hls_tile.rio.bounds()) \
        .rio.clip_box(*hls_tile.rio.bounds()) \
        .rio.reproject(hls_crs) \
        .rio.pad_box(*hls_crs_bounds) \
        .rio.clip_box(*hls_crs_bounds) \
        .fillna(0) \
        .to_zarr(output_path, mode='w')
    return tile_name


def _get_compatible_fsmap(path, filename, storage_account, account_key):
    full_path = f"{path}/{filename}"
    return fsspec.get_mapper(
        full_path,
        account_name=storage_account,
        account_key=account_key
    )


daymet_mean = xr.open_zarr("../daymet_means_local.zarr/na_40yr_monthly_means_norm.zarr") \
    .rename({'prcp': 'mn_prcp', 'swe': 'mn_swe', 'tmax': 'mn_tmax', 'tmin': 'mn_tmin', 'vp': 'mn_vp'})
daymet_std = xr.open_zarr("../daymet_std_local.zarr/na_40yr_monthly_std_devs_norm.zarr") \
    .rename({'prcp': 'std_prcp', 'swe': 'std_swe', 'tmax': 'std_tmax', 'tmin': 'std_tmin', 'vp': 'std_vp'})
daymet = xr.Dataset.merge(daymet_mean, daymet_std) \
    .rio.write_crs('+proj=lcc +lat_1=25 +lat_2=60 +lat_0=42.5 +lon_0=-100 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs') \
    .drop_vars(["lat", "lon"]) \
    .compute()

catalog_path = fsspec.get_mapper(
    "az://fia/catalogs/hls_conus_2015-2019.zarr",
    account_name="usfs",
    account_key=os.environ['ACCOUNT_KEY']
)
catalog = xr.open_zarr(catalog_path)
tiles = sorted(set(catalog['tile'].values))

with open('daymet_checkpoint', 'r') as f:
    checkpointed = f.read().split('\n')
    for tile in tiles:
        if tile in checkpointed:
            print(f"skipping checkpointed {tile}")
            continue
        try:
            with open('daymet_checkpoint', 'a') as f:
                print(prep_daymet_tile(daymet, tile))
                f.write(f"{tile}\n")
        except Exception as e:
            print(e)
            print("FAILED", tile)
