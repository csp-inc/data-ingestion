import time
from functools import partial

import fsspec
import pandas as pd
import xarray as xr
from dask.distributed import as_completed

from utils.hls.catalog import HLSBand
from utils.hls.catalog import scene_to_urls


def get_mask(qa_band):
    """Takes a data array HLS qa band and returns a mask of True where quality is good, False elsewhere
    Mask usage:
        ds.where(mask)
        
    Example:
        qa_mask = get_mask(dataset[HLSBand.QA])
        ds = dataset.drop_vars(HLSBand.QA)
        masked = ds.where(qa_mask)
    """
    def is_bad_quality(qa):
        cirrus = 0b1
        cloud = 0b10
        adjacent_cloud = 0b100
        cloud_shadow = 0b1000
        high_aerosol = 0b11000000

        return (qa & cirrus > 0) | (qa & cloud > 0) | (qa & adjacent_cloud > 0) | \
            (qa & cloud_shadow > 0) | (qa & high_aerosol == high_aerosol)
    return xr.where(is_bad_quality(qa_band), False, True)  # True where is_bad_quality is False, False where is_bad_quality is True


def fetch_band_url(tpl, chunks):
    """Fetch a given url with xarray, creating a dataset with a single data variable of the band name for the url.
    
    Args:
        tpl (Tuple[str, str]): tuple of the form (band, url) - the url to fetch and the band name for the data variable
        chunks (Dict[str, int]): How to chunk HLS input data
        
    Returns:
        xarray.Dataset: Dataset for the given HLS scene url with the data variable being named the given band
        
    """
    band, url = tpl
    da = xr.open_rasterio(url, chunks=chunks)
    da = da.squeeze().drop(labels='band')
    return da.to_dataset(name=band)


def compute_tile_median(job_id, ds, groupby, qa_name, write_store):
    """Compute QA-band-masked {groupby} median reflectance for the given dataset and save the result as zarr to `write_store`.
    
    Args:
        job_id (str): The job_id of the tile being computed
        ds (xarray.Dataset): Dataset to compute on
        groupby (str): How to group the dataset (e.g. "time.month")
        qa_name (str): Name of the QA band to use for masking
        write_store (fsspec.FSMap): The location to write the zarr
    
    Returns:
        str: The job_id that was computed and written
        
    """
    # apply QA mask
    if qa_name in ds.data_vars:
        qa_mask = get_mask(ds[qa_name])
        ds = (ds
            .drop_vars(qa_name)  # drop QA band
            .where(qa_mask)  # Apply mask
        )
    zarr = (ds
        .where(ds != -1000)  # -1000 means no data - set those entries to nan
        .groupby(groupby)
        .median()
        .chunk({'month': 1, 'y': 3660, 'x': 3660})  # groupby + median changes chunk size...lets change it back
        .to_zarr(write_store, mode='w')
    )
    return job_id


def calculate_job_median(job_id, job_df, job_groupby, bands, band_names, qa_band_name, chunks, write_store, client):
    """A job compatible with `process_catalog` which computes per-band median reflectance for the input job_df.
    
    Args:
        job_id (str): Id of the job, used for tracking purposes
        job_df (pandas.Dataframe): Dataframe of scenes to include in the computation
        job_groupby (str): How to group the dataset produced from the dataframe (e.g. "time.month")
        bands (List[HLSBand]): List of HLSBand objects to compute median reflectance on
        band_names (List[str]): List of band name strings
        qa_band_name (str): Name of the QA band to use for masking
        chunks (Dict[str, int]): How to chunk HLS input data
        write_store (fsspec.FSMap): The location to write any results
        client (dask.distributed.Client): Dask cluster client to submit tasks to
        
    Returns:
        dask.distributed.Future: Future for the computation that is being done, can be waited on.
        
    """
    
    # create partial function for concatenating individual scene ds into single ds
    concat_time_dim = partial(xr.concat, dim=pd.DatetimeIndex(job_df['dt'].tolist(), name='time'))
    
    # create partial function for applying fetch_band_url with chunks
    fetch_band_url_partial = partial(fetch_band_url, chunks=chunks)
    
    scene_ds_futures = []
    for _, row in job_df.iterrows():
        scenes = scene_to_urls(row['scene'], row['sensor'], bands)
        # list of datasets that need to be xr.merge'd (future)
        band_ds_futures = client.map(fetch_band_url_partial, list(zip(band_names, scenes)), priority=-5, retries=1)
        # single dataset with every band (future)
        scene_ds_futures.append(client.submit(xr.merge, band_ds_futures, priority=-4, retries=1))
        
    # dataset of a single index/tile with a data var for every band and dimensions: x, y, time
    job_ds_future = client.submit(concat_time_dim, scene_ds_futures, priority=-3, retries=1)
    # compute masked, monthly, median per band per pixel
    return client.submit(compute_tile_median, job_id, job_ds_future, job_groupby, qa_band_name, write_store, priority=-2, retries=1)


def process_catalog(
    catalog,
    catalog_groupby,
    job_fn,
    job_groupby,
    chunks,
    account_name,
    storage_container,
    account_key,
    client,
    concurrency,
    logger,
):
    """Process a catalog.
    
    Args:
        catalog (xarray.Dataset): catalog to process
        catalog_groupby (str): column to group the catalog in to jobs by (e.g. 'INDEX', 'tile')
        job_fn: a function to apply to each job from the grouped catalog (e.g. `calculate_job_median`)
        job_groupby (str): how to group data built within each job (e.g. 'time.month', 'time.year')
         chunks (Dict[str, int]): How to chunk HLS input data
        account_name (str): Azure storage account to write results to
        storage_container (str): Azure storage container within the `account_name` to write results to
        account_key (str): Azure account key for the `account_name` which results are written to
        client (dask.distributed.Client): Dask cluster client to submit tasks to
        concurrency (int): Number of jobs to have running on the Dask cluster at once, must be >0
        logger (logging.Logger): Logger to log info to.
        
    """
    bands = catalog.attrs['bands']
    band_names = [band.name for band in bands]
    qa_band_name = HLSBand.QA.name

    df = catalog.to_dataframe()
    first_futures = []
    start_time = time.perf_counter()
    jobs = list(df.groupby(catalog_groupby))
    num_jobs = len(jobs)
    
    while len(first_futures) < concurrency:
        job_id, job_df = jobs.pop(0)
        logger.info(f"Submitting job {job_id}")
        write_store = fsspec.get_mapper(
            f"az://{storage_container}/{job_id}.zarr",
            account_name=account_name,
            account_key=account_key
        )
        first_futures.append(
            job_fn(job_id, job_df, job_groupby, bands, band_names, qa_band_name, chunks, write_store, client)
        )
    
    ac = as_completed(first_futures)
    for future in ac:
        try:
            result = future.result()
            logger.info(f"Completed job {result}")
        except Exception as e:
            logger.exception("Exception from dask cluster")
        if len(jobs) > 0:
            job_id, job_df = jobs.pop(0)
            logger.info(f"Submitting job {job_id}")
            write_store = fsspec.get_mapper(
                f"az://{storage_container}/{job_id}.zarr",
                account_name=account_name,
                account_key=account_key
            )
            ac.add(
                job_fn(job_id, job_df, job_groupby, bands, band_names, qa_band_name, chunks, write_store, client)
            )
    logger.info(f"{num_jobs} completed in {time.perf_counter()-start_time} seconds")
