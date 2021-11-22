# CSP Azure Data Ingestion

## Introduction

Tools for efficiently ingesting, processing, and sampling data from Azure's public datasets using [Dask](https://dask.org/). \
These tools are intended for use in Microsoft's Planetary Computer [Jupyterhub environment](https://planetarycomputer.microsoft.com/compute), but are functional \
using other Dask-enabled Jupyterhub environments, or using multithreading on an Azure VM or local machine.

## Distributed Environment Setup

### Planetary Computer

Apply for Planetary Computer usage [here.](https://planetarycomputer.microsoft.com/account/request)  

If accepted, visiting [this link](https://planetarycomputer.microsoft.com/compute) and logging in will show several available environments.  
Choose the first entitled "CPU - Python", which uses a [Pangeo Notebook](https://github.com/pangeo-data/pangeo-docker-images) environment. 

### Custom Azure Jupyterhub with Kubernetes

Setting up Jupyterhub using your own Kubernetes cluster on Azure is an involved process  
and is not recommended except for those that don't qualify Planetary Computer usage.  
A guide for deploying Jupyterhub with the Pangeo Notebook environment can be found [here](https://pangeo.io/setup_guides/azure.html).

### Uploading Code

There are several options for git integration in a Jupyterhub environment, including [git codespaces](https://planetarycomputer.microsoft.com/docs/overview/ui-codespaces/),  \
and Jupyter extensions for git (search 'git' in the extension manager of Jupyterhub for several options).  
The simplest option is to clone or fork this repo locally, and upload the full contents to the Jupyterhub environment. 

## Local Multiprocessing

An anaconda environment with minimal requirements for running locally is included.  
This option is not yet diligently tested and is not recommended for use.

## Chimera Data Use Case

The use of the classes and methods provided here are demonstrated with a workflow for collecting,  
preprocessing, and sampling monthly median Harmonized Landsat Sentinel-2 (HLS) data for later use in training  
of a simplified version of a recurrent neural network called [Chimera](https://github.com/tonychangmsu/Chimera-RCNN).

Follow the sequence of four notebooks in chimera_hls_example to see the collection of monthly  
median HLS data and subsequent point sampling for later use with Azure Machine Learning. 













