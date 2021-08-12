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
