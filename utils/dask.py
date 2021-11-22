import os
import tempfile
import zipfile



def create_cluster(workers, clust_type, worker_threads=1, worker_memory=2, scheduler_threads=1, \
                   scheduler_memory=2, address = None, proxy_address = None):
    """Create and return a cluster with a given number of workers each with `cores` and `memory`
    
    Args:
        address (str): Address to gateway server
        proxy_address (str): Address of scheduler proxy server
        workers (int): Number of workers to scale the cluster to
        worker_threads (int): Number of threads for each worker to run
        worker_memory (int): Memory in GiB allocated to each worker
        scheduler_threads (int): Number of threads for the scheduler to run
        scheduler_memory (int): Memory in GiB allocated to the scheduler
        
    Returns:
        dask_gateway.GatewayCluster: started cluster
        
    """
    if clust_type == 'distributed':
        from dask_gateway import GatewayCluster
        if ((address is not None) & (proxy_address is not None)):
            cluster = GatewayCluster(
                address=address,
                proxy_address=proxy_address,
                worker_cores=worker_threads,
                worker_memory=worker_memory,
                scheduler_cores=scheduler_threads,
                scheduler_memory=scheduler_memory
            )
        else:
            cluster = GatewayCluster()
            cluster_options = {'workers':workers,'worker_cores':worker_threads, \
                            'worker_memory':worker_memory, 'scheduler_cores':scheduler_threads, \
                            'scheduler_memory':scheduler_memory}
    else:
        from dask.distributed import LocalCluster
        cluster = LocalCluster(
            threads_per_worker=1,
            memory_limit=0
        )
    cluster.scale(workers)
    return cluster


def zip_code(path):
    tmp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(tmp_dir, 'source.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        _zipdir(path, zipf)
    return zip_path


def _zipdir(path, zipf):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            zipf.write(os.path.join(root, file), \
                       os.path.relpath(os.path.join(root, file), 
                                       os.path.join(path, '..')))
