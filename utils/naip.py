import os
import pickle
import tempfile
import urllib

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rtree
import shapely

import concurrent.futures
import logging
import os
import sys
import tempfile
from pathlib import Path

import fiona.transform
import fsspec
import numpy as np
import pandas as pd
import rasterio
import rioxarray
import xarray as xr
from dask.distributed import get_worker
from rasterio.windows import Window


index_files = ["tile_index.dat", "tile_index.idx", "tiles.p"]
index_blob_root = "https://naipblobs.blob.core.windows.net/naip-index/rtree/"
temp_dir = os.path.join(tempfile.gettempdir(), "naip")


class NAIPTileIndex:
    """
    Utility class for performing NAIP tile lookups by location.
    """

    tile_rtree = None
    tile_index = None
    base_path = None

    def __init__(self, base_path=None):
        for file_path in index_files:
            download_url(
                index_blob_root + file_path,
                base_path + "/" + file_path,
                progress_updater=None,
            )

        self.base_path = base_path
        self.tile_rtree = rtree.index.Index(base_path + "/tile_index")
        self.tile_index = pickle.load(open(base_path + "/tiles.p", "rb"))

    def lookup_tile(self, lat, lon):
        """"
        Given a lat/lon coordinate pair, return the list of NAIP tiles that contain
        that location.

        Returns an array containing [mrf filename, idx filename, lrc filename].
        """

        point = shapely.geometry.Point(float(lon), float(lat))
        intersected_indices = list(self.tile_rtree.intersection(point.bounds))

        intersected_files = []
        tile_intersection = False

        for idx in intersected_indices:

            intersected_file = self.tile_index[idx][0]
            intersected_geom = self.tile_index[idx][1]
            if intersected_geom.contains(point):
                tile_intersection = True
                intersected_files.append(intersected_file)

        if not tile_intersection and len(intersected_indices) > 0:
            print(
                """Error: there are overlaps with tile index,
                      but no tile completely contains selection"""
            )
            return None
        elif len(intersected_files) <= 0:
            return None
        else:
            return intersected_files


def download_url(
    url, destination_filename=None, progress_updater=None, force_download=False
):
    """
    Download a URL to a temporary file
    """

    # This is not intended to guarantee uniqueness, we just know it happens to guarantee
    # uniqueness for this application.
    if destination_filename is None:
        url_as_filename = url.replace("://", "_").replace("/", "_")
        destination_filename = os.path.join(temp_dir, url_as_filename)
    if (not force_download) and (os.path.isfile(destination_filename)):
        print(
            "Bypassing download of already-downloaded file {}".format(
                os.path.basename(url)
            )
        )
        return destination_filename
    print(
        "Downloading file {} to {}".format(os.path.basename(url), destination_filename),
        end="",
    )
    urllib.request.urlretrieve(url, destination_filename, progress_updater)
    assert os.path.isfile(destination_filename)
    nBytes = os.path.getsize(destination_filename)
    print("...done, {} bytes.".format(nBytes))
    return destination_filename


def display_naip_tile(filename):
    """
    Display a NAIP tile using rasterio.

    For .mrf-formatted tiles (which span multiple files), 'filename' should refer to the
    .mrf file.
    """

    # NAIP tiles are enormous; downsize for plotting in this notebook
    dsfactor = 10

    with rasterio.open(filename) as raster:

        # NAIP imagery has four channels: R, G, B, IR
        #
        # Stack RGB channels into an image; we won't try to render the IR channel
        #
        # rasterio uses 1-based indexing for channels.
        h = int(raster.height / dsfactor)
        w = int(raster.width / dsfactor)
        print("Resampling to {},{}".format(h, w))
        r = raster.read(1, out_shape=(1, h, w))
        g = raster.read(2, out_shape=(1, h, w))
        b = raster.read(3, out_shape=(1, h, w))

    rgb = np.dstack((r, g, b))
    plt.figure(figsize=(7.5, 7.5), dpi=100, edgecolor="k")
    plt.imshow(rgb)
    raster.close()

    
##### NAIP sampling functions
CRS = "EPSG:4326"

# Storage locations are documented at http://aka.ms/ai4edata-naip
NAIP_ROOT = "https://naipblobs.blob.core.windows.net/naip"


def get_plot_urls(input_path, logger):
    # set up
    temp_dir = os.path.join(tempfile.gettempdir(), "naip")
    os.makedirs(temp_dir, exist_ok=True)
    index = NAIPTileIndex(temp_dir)

    df = pd.read_csv(input_path)
    df = df[df['INVYR'] >= 2015]
    df['url'] = df.apply(lambda row: get_plot_url(row, index, logger), axis=1)
    df = df[~df['url'].isnull()]
    return df


def get_plot_url(plot, index, logger):
    lon, lat = float(plot["LON"]), float(plot["LAT"])
    query_year = int(float(plot["INVYR"]))

    # Find the filenames that intersect with our lat/lon
    naip_files = index.lookup_tile(lat, lon)

    if naip_files is None or len(naip_files) == 0:
        logger.debug(f'No intersection, skipping index {plot["INDEX"]}')
        return None

    # Get closest year
    naip_years = np.array([int(n.split("/")[2]) for n in naip_files])
    closest = min(naip_years, key=lambda x: abs(x - query_year))
    match_idx = np.where(naip_years == closest)[0][0]
    return naip_files[match_idx]


def get_plot_chip_and_state(plot):
    """Given a plot dictionary fetch the desired 256x256 image tile.

    Args:
        plot: dictionary with LON, LAT, INVYR

    Returns:
        Tuple[Optional[Image], Optional[str]]: image tile or None if no match could be found and the state code it was from

    """
    lon, lat = float(plot["LON"]), float(plot["LAT"])
    image_url = plot["url"]
    full_url = f"{NAIP_ROOT}/{image_url}"
    with rasterio.open(full_url) as f:

        # Convert our lat/lon point to the local NAIP coordinate system
        x_tile_crs, y_tile_crs = fiona.transform.transform(
            CRS, f.crs.to_string(), [lon], [lat]
        )
        x_tile_crs = x_tile_crs[0]
        y_tile_crs = y_tile_crs[0]

        # Convert our new x/y coordinates into pixel indices
        x_tile_offset, y_tile_offset = ~f.transform * (x_tile_crs, y_tile_crs)
        x_tile_offset = int(np.floor(x_tile_offset))
        y_tile_offset = int(np.floor(y_tile_offset))

        # The secret sauce: only read data from a 256x256 window centered on our point
        image_crop = f.read(
            window=Window(x_tile_offset - 128, y_tile_offset - 128, 256, 256)
        )

        image_crop = np.moveaxis(image_crop, 0, -1)

    # Sometimes our point will be on the edge of a NAIP tile, and our windowed reader above
    # will not actually return a 256x256 chunk of data we could handle this nicely by going
    # back up to the `naip_files` list and trying to read from one of the other tiles -
    # because the NAIP tiles have overlap with one another, there should exist an intersecting
    # tile with the full window.
    if (image_crop.shape[0] == 256) and (image_crop.shape[1] == 256):
        # NAIP path [blob root]/v002/[state]/[year]/[state]_[resolution]_[year]/[quadrangle]/filename
        state = image_url.split("/")[1]
        return image_crop, state

    else:
        try:
            get_worker().log_event(
                "message", f"Skipping {plot['INDEX']}"
            )
        except ValueError:
            print(f"Our crop was likely at the edge of a NAIP tile, skipping point {plot['INDEX']}")
        return None, None
    

def write_chip(img_arr, plot, account_name, storage_container, account_key):
    with fsspec.open(
        f"az://{storage_container}/naip/{int(plot['INDEX'])}.tif",
        account_name=account_name,
        account_key=account_key,
        mode='wb'
    ) as f:
        da = xr.DataArray(img_arr[:, :, :3], dims=('y', 'x', 'band')).transpose('band', 'y', 'x').rio.to_raster(f)
    
def chunk_df(df, chunk_size):
    return [(i, df[i:i+chunk_size]) for i in range(0,df.shape[0],chunk_size)]

def get_point_chips(job_id, job_df, account_name, storage_container, account_key):
    for _, row in job_df.iterrows():
        img, state = get_plot_chip_and_state(row)
        if img is not None:
            write_chip(img, row, account_name, storage_container, account_key)
    return job_id
