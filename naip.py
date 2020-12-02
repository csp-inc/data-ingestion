"""Ingest NAIP data:

Usage

"""
import argparse
import concurrent.futures
import logging
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from csv import DictReader
from pathlib import Path

import fiona.transform
import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window

from utils.naip import NAIPTileIndex

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# Workaround for a problem in older rasterio versions
os.environ["CURL_CA_BUNDLE"] = "/etc/ssl/certs/ca-certificates.crt"

CRS = "EPSG:4326"

# Storage locations are documented at http://aka.ms/ai4edata-naip
NAIP_ROOT = "https://naipblobs.blob.core.windows.net/naip"


def main(input_path, output_dir, workers):
    # set up
    temp_dir = os.path.join(tempfile.gettempdir(), "naip")
    os.makedirs(temp_dir, exist_ok=True)
    index = NAIPTileIndex(temp_dir)

    with open(input_path, "r") as fobj:
        plots = DictReader(fobj)
        plot_url_lst = [(plot, get_plot_url(plot, index)) for plot in plots]
    print(len(plot_url_lst))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_index = {}
        for plot, url in plot_url_lst:
            if url:
                future_to_index[executor.submit(get_and_write_tile, plot, url, output_dir)] = plot["INDEX"]
        for future in concurrent.futures.as_completed(future_to_index):
            try:
                future.result()
            except Exception as e:
                logger.exception(e)
            else:
                logger.debug("Finished %s", future_to_index[future])


def get_and_write_tile(plot, url, output_dir):
    tile, state = get_plot_tile_and_state(url, plot)
    if tile is not None:
        write_tile(tile, plot, state, output_dir)


def get_plot_url(plot, index):
    lon, lat = float(plot["LON"]), float(plot["LAT"])
    query_year = int(float(plot["INVYR"]))

    # Find the filenames that intersect with our lat/lon
    naip_files = index.lookup_tile(lat, lon)

    if naip_files is None or len(naip_files) == 0:
        logger.info(f'No intersection, skipping index {plot["INDEX"]}')
        return None

    # Get closest year
    naip_years = np.array([int(n.split("/")[2]) for n in naip_files])
    closest = min(naip_years, key=lambda x: abs(x - query_year))
    match_idx = np.where(naip_years == closest)[0][0]
    return naip_files[match_idx]


def get_plot_tile_and_state(image_url, plot):
    """Given a plot dictionary and the NAIP rtree index fetch the desired 256x256 image tile.

    Args:
        plot: dictionary with LON, LAT, INVYR
        index: rtree index of NAIP files

    Returns:
        image tile or None if no match could be found

    """
    lon, lat = float(plot["LON"]), float(plot["LAT"])
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
        logger.info(
            f"Our crop was likely at the edge of a NAIP tile, skipping point {plot['INDEX']}"
        )
        return None, None


def write_tile(tile, plot, state, output_dir):
    """Write tile to local or blob storage."""
    write_dir = Path(output_dir) / state
    write_dir.mkdir(exist_ok=True)
    path = write_dir / f'{plot["INDEX"]}.tif'
    img = Image.fromarray(tile[:, :, :3])
    img.save(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_path", help="Path to input csv either locally or on Azure blob storage"
    )
    parser.add_argument(
        "output_dir", help="Path to out dir either locally or on Azure blob storage"
    )
    parser.add_argument(
        "-w",
        "--workers",
        help="Number of processes to use",
        type=int,
        default=os.cpu_count(),
    )
    args = parser.parse_args()
    main(args.input_path, args.output_dir, args.workers)
