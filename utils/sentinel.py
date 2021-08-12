import geopandas as gpd
import numpy as np
from pystac_client import Client as pystacClient

def search_catalogue(area_of_interest, time_of_interest):
    catalog = pystacClient.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects=area_of_interest,
        datetime=time_of_interest,
    )

    # Check how many items were returned
    return list(search.get_items())

def apply_buffer_to_points(df, buffer_distance, buffer_type='square', resolution=1):
    buffer_style = {'round':1, 'flat':2, 'square':3}
    srcdf = df.copy()
    srcdf = srcdf.to_crs('EPSG:3857')
    srcdf['geometry'] = srcdf.geometry.buffer(buffer_distance,\
                                           cap_style=buffer_style[buffer_type],\
                                           resolution=resolution)
    srcdf = srcdf.to_crs('EPSG:4326')
    df = df.set_geometry(srcdf.geometry)
    return(df)

def create_daterange(df,month=6,day=30):
    daterange = [f"{int(r['DATETIME'].strftime('%Y'))-1}-{month}-{day}/{int(r['DATETIME'].strftime('%Y'))}-{month}-{day}" for i,r in df.iterrows()]
    return daterange

def create_aoi(geometry):
    coordinates =  np.dstack(geometry.boundary.coords.xy).tolist()
    area_of_interest = {
    "type": "Polygon",
    "coordinates": coordinates,
    }
    return area_of_interest

def get_asset_href(item, asset='visual'):
    asset_href = item.assets[asset].href
    return pc.sign(asset_href)

def collect_image_chips(gdf):
    items_list = []
    for _,f in gdf.iterrows():
        area_of_interest = f['AOI']
        time_of_interest = f['DATERANGE']
        items = search_catalogue(area_of_interest, time_of_interest)
        #n_items = len(items)
        #hrefs.append([get_asset_href(items[i]) for i in range(n_items)])
        items_list.append(items)
    return items_list