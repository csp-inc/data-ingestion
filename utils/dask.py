import os
import tempfile
import zipfile

from dask_gateway import GatewayCluster

def create_cluster(workers, worker_threads=1, worker_memory=2, scheduler_threads=1, scheduler_memory=2):
    """Create and return a cluster with a given number of workers each with `cores` and `memory`
    
    Args:
        workers (int): Number of workers to scale the cluster to
        worker_threads (int): Number of threads for each worker to run
        worker_memory (int): Memory in GiB allocated to each worker
        scheduler_threads (int): Number of threads for the scheduler to run
        scheduler_memory (int): Memory in GiB allocated to the scheduler
        
    Returns:
        dask_gateway.GatewayCluster: started cluster
        
    """
    cluster = GatewayCluster(
        worker_cores=worker_threads,
        worker_memory=worker_memory,
        scheduler_cores=scheduler_threads,
        scheduler_memory=scheduler_memory
    )
    cluster.scale(workers)
    return cluster

def upload_source(path, client):
    """Upload the data in the given path directory to the client's workers as a zip file
    
    Args:
        path (str): Path to directory to zip + upload
        client (dask.distributed.Client): client to submit upload to
        
    """
    tmp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(tmp_dir, 'source.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        _zipdir(path, zipf)
    client.upload_file(zip_path)

def _zipdir(path, zipf):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            zipf.write(os.path.join(root, file))
