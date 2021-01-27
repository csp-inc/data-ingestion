import json
import time
from functools import partial

import  dask
import fsspec
import pandas as pd
import xarray as xr
from dask.distributed import as_completed

from utils.dask import create_cluster
from utils.dask import zip_code
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


@dask.delayed
def fetch_band_url(band, url, chunks):
    """Fetch a given url with xarray, creating a dataset with a single data variable of the band name for the url.
    
    Args:
        band (str): the band name for the data variable
        url (str): the url to fetch
        chunks (Dict[str, int]): How to chunk HLS input data
        
    Returns:
        xarray.Dataset: Dataset for the given HLS scene url with the data variable being named the given band
        
    """
    da = xr.open_rasterio(url, chunks=chunks)
    da = da.squeeze().drop_vars('band')
    # There is a bug in open_rasterio as it doesn't coerce scale_factor/add_offset to a float, but leaves it as a string.
    # If you then save this file as a zarr it will save scale_factor/add_offset as a string
    # when you try to re-open the zarr it will crash trying to apply the scale factor + add offset
    # https://github.com/pydata/xarray/issues/4784
    if 'scale_factor' in da.attrs:
        da.attrs['scale_factor'] = float(da.attrs['scale_factor'])
    if 'add_offset' in da.attrs:
        da.attrs['add_offset'] = float(da.attrs['add_offset'])
    return da.to_dataset(name=band, promote_attrs=True)


@dask.delayed
def get_scene_dataset(scene, sensor, bands, band_names, chunks):
    """For a given scene/sensor combination and list of bands + names, build a dataset using the dask client.
    
    Args:
        scene (str): String compatible with `scene_to_urls` specifying a single satellite capture of an HLS tile
        sensor (str): 'S' (Sentinel) or 'L' (Landsat) - what sensor the scene came from
        bands (List[HLSBand]): List of HLSBands to include in the dataset as data variables
        band_names (List[str]): Names of the bands, used to name each data variable
        client (dask.distributed.client): Client to submit functions to
        chunks (dict[str, int]): How to chunk the data across workers in dask
    """
    # list of datasets, one for each band, that need to be xr.merge'd (futures)
    scenes = scene_to_urls(scene, sensor, bands)
    band_datasets = [
        fetch_band_url(band, scene, chunks=chunks)
        for band, scene in zip(band_names, scenes)
    ]

    # single dataset with every band (future)
    return xr.merge(
        dask.compute(*band_datasets),
        combine_attrs='override',  # first band's attributes will be used
    )


def compute_tile_median(ds, groupby, qa_name):
    """Compute QA-band-masked {groupby} median reflectance for the given dataset.
    
    Args:
        ds (xarray.Dataset): Dataset to compute on with dimensions 'time', 'x', and 'y'
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
    return (ds
        # valid range is 0-10000 per LaSRC v3 guide: https://prd-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/atoms/files/LSDS-1368_L8_C1-LandSurfaceReflectanceCode-LASRC_ProductGuide-v3.pdf
        .where(ds <= 10000)
        .where(ds >= 0)
        .groupby(groupby)
        .median(keep_attrs=True)
        .chunk({'month': 1, 'y': 3660, 'x': 3660})  # groupby + median changes chunk size...lets change it back
    )


def save_to_zarr(ds, write_store, mode, success_value):
    """Save given dataset to zarr.
    
    Args:
        ds (xarray.Dataset): dataset to save
        write_store (fsspec.FSMap): destination to save ds to
        mode (str): what mode to use for writing, see http://xarray.pydata.org/en/stable/generated/xarray.Dataset.to_zarr.html?highlight=to_zarr
        success_value (Any): what to return when write is succesful
        
    Returns:
        Any: the provided success_value
    """
    ds.to_zarr(write_store, mode=mode)
    return success_value


def calculate_job_median(job_id, job_df, job_groupby, bands, band_names, qa_band_name, chunks, write_store):
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
        
    Returns:
        Any: Result of the computation to be passed back to process_catalog
        
    """
    
    scene_datasets = []
    for _, row in job_df.iterrows():
        # single dataset with every band
        scene_datasets.append(
            get_scene_dataset(
                scene=row['scene'],
                sensor=row['sensor'],
                bands=bands,
                band_names=band_names,
                chunks=chunks
            )
        )

    # dataset of a single index/tile with a data var for every band and dimensions: x, y, time
    job_ds_future = xr.concat(
        dask.compute(*scene_datasets),
        dim=pd.DatetimeIndex(job_df['dt'].tolist(), name='time'),
        combine_attrs='override',
    )
    # compute masked, monthly, median per band per pixel
    median = compute_tile_median(
        job_ds_future,
        job_groupby,
        qa_band_name,
    )
    # save to zarr
    return save_to_zarr(
        median,
        write_store,
        'w',
        job_id,
    )


def _read_checkpoints(path, logger):
    """
    """
    try:
        with open(path, 'r') as f:
            return set(f.read().splitlines())
    except FileNotFoundError:
        logger.warning('No checkpoint file found, creating it at %s', path)
        with open(path, 'x') as f:
            pass
        return []

    
def process_catalog(
    catalog,
    catalog_groupby,
    job_fn,
    job_groupby,
    chunks,
    account_name,
    storage_container,
    account_key,
    concurrency,
    checkpoint_path,
    logger,
    cluster_args,
    code_path=None,
    cluster_restart_freq=-1,
):
    """Process a catalog. This function handles job submission, checkpointing successful jobs, managing job concurrency, and cluster management.
    
    Args:
        catalog (xarray.Dataset): catalog to process
        catalog_groupby (str): column to group the catalog in to jobs by (e.g. 'INDEX', 'tile')
        job_fn: a function to apply to each job from the grouped catalog (e.g. `calculate_job_median`)
        job_groupby (str): how to group data built within each job (e.g. 'time.month', 'time.year')
        chunks (Dict[str, int]): How to chunk HLS input data
        account_name (str): Azure storage account to write results to
        storage_container (str): Azure storage container within the `account_name` to write results to
        account_key (str): Azure account key for the `account_name` which results are written to
        concurrency (int): Number of jobs to have running on the Dask cluster at once, must be >0
        checkpoint_path (str): Path to a local file for reading and updating checkpoints
        logger (logging.Logger): Logger to log info to.
        cluster_args (Dict[str, int]): Dict with kwargs (workers, worker_threads, worker_memory, scheduler_threads, scheduler_memory) for the create_cluster command in utils/dask.py
        code_path: Path to code to upload to cluster
        cluster_restart_freq (dask_gateway.GatewayCluster): How often to restart the cluster, <= -1 means never, must be greater than `concurrency` or -1
        
    """
    def run_job_subset(job_subset, client):
        first_futures = []
        
        # submit first set of jobs
        while len(first_futures) < concurrency and len(job_subset) > 0:
            job_id, job_df = job_subset.pop(0)
            logger.info(f"Submitting job {job_id}")
            write_store = fsspec.get_mapper(
                f"az://{storage_container}/{job_id}.zarr",
                account_name=account_name,
                account_key=account_key
            )
            first_futures.append(
                client.submit(job_fn, job_id, job_df, job_groupby, bands, band_names, qa_band_name, chunks, write_store, retries=1)
            )

        # wait on completed jobs
        ac = as_completed(first_futures)
        for future in ac:
            try:
                result = future.result()
                logger.info(f"Completed job {result}")
                metrics['job_completes'] += 1
                with open(checkpoint_path, 'a') as checkpoint_file:
                    checkpoint_file.write(str(result) + '\n')
            except Exception as e:
                logger.exception("Exception from dask cluster")
                metrics['job_errors'] += 1
            # submit another job
            if len(job_subset) > 0:
                job_id, job_df = job_subset.pop(0)
                logger.info(f"Submitting job {job_id}")
                write_store = fsspec.get_mapper(
                    f"az://{storage_container}/{job_id}.zarr",
                    account_name=account_name,
                    account_key=account_key
                )
                ac.add(
                    client.submit(job_fn, job_id, job_df, job_groupby, bands, band_names, qa_band_name, chunks, write_store, retries=1)
                )

    assert cluster_restart_freq > concurrency or cluster_restart_freq == -1, "cluster_restart_freq must be greater than concurrency or -1"
    
    # zip code if provided
    zipped_path = zip_code(code_path) if code_path else None
    
    # set up band information from catalog
    bands = catalog.attrs['bands']
    band_names = [band.name for band in bands]
    qa_band_name = HLSBand.QA.name

    # start metrics
    metrics = dict(
        job_errors=0,
        job_skips=0,
        job_completes=0
    )
    start_time = time.perf_counter()
    
    # set up catalog, jobs, and checkpoints
    df = catalog.to_dataframe()
    checkpoints = _read_checkpoints(checkpoint_path, logger)
    jobs = []
    for job_id, job in df.groupby(catalog_groupby):
        if job_id in checkpoints:
            logger.info(f"Skipping checkpointed job {job_id}")
            metrics['job_skips'] += 1
        else:
            jobs.append((job_id, job))

    if cluster_restart_freq == -1:
        cluster_restart_freq = len(jobs)
    
    for start_idx in range(0, len(jobs), cluster_restart_freq):
        subset = jobs[start_idx:start_idx+cluster_restart_freq]
        logger.info("Starting cluster")
        with create_cluster(**cluster_args) as cluster:
            logger.info("Cluster dashboard visible at %s", cluster.dashboard_link)
            cluster_client = cluster.get_client()
            if zipped_path:
                logger.info("Uploading code to cluster")
                cluster_client.upload_file(zipped_path)
            run_job_subset(subset, cluster_client)
    
    metrics['time'] = time.perf_counter()-start_time
    logger.info(f"Metrics: {json.dumps(metrics)}")
