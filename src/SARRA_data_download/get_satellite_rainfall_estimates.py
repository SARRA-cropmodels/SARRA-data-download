import datetime
from datetime import datetime as dt
import gzip
import shutil
import calendar
import os
import requests
import xarray as xr 
from tqdm import tqdm as tqdm
import rioxarray as rio
import rioxarray
import numpy as np





###################
############ TAMSAT
###################





def download_TAMSAT_day(query_date, save_path):

    query_month = query_date.strftime('%m')
    query_day = query_date.strftime('%d')

    URL_filename = str(query_date.year)+"/"+str(query_month)+"/rfe"+str(query_date.year)+"_"+str(query_month)+"_"+str(query_day)+".v3.1.nc"
    URL_full = "http://www.tamsat.org.uk/public_data/data/v3.1/daily/"+URL_filename

    # if save_path does not exist, we create it
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    response = requests.get(URL_full)
    save_filename = "TAMSAT_"+URL_filename.replace("/","_")
    open(os.path.join(save_path,save_filename), "wb").write(response.content)





def crop_and_save_TAMSAT_day(query_date, area, selected_area, save_path):

    query_month = query_date.strftime('%m')
    query_day = query_date.strftime('%d')
    URL_filename = str(query_date.year)+"/"+str(query_month)+"/rfe"+str(query_date.year)+"_"+str(query_month)+"_"+str(query_day)+".v3.1.nc"
    save_filename = "TAMSAT_"+URL_filename.replace("/","_")
    nc_file_content = xr.open_dataset(os.path.join(save_path,save_filename))

    xarray_variable_name = "rfe_filled"

    # cropping
    nc_file_content = nc_file_content.where((nc_file_content.lat < area[selected_area][0])
                            & (nc_file_content.lat > area[selected_area][2])
                            & (nc_file_content.lon > area[selected_area][1])
                            & (nc_file_content.lon < area[selected_area][3])
                        ).dropna(dim='lat', how='all').dropna(dim='lon', how='all')

    bT = nc_file_content[xarray_variable_name]
    bT = bT.rio.set_spatial_dims(x_dim='lon', y_dim='lat')
    bT.rio.crs
    bT.rio.write_crs("epsg:4326", inplace=True)
    output_filename = "TAMSAT_v3.1_"+selected_area+"_"+xarray_variable_name+"_"+str(query_date.year)+"_"+str(query_month)+"_"+str(query_day)+".tif"
    output_path = "../data/3_output/TAMSAT_v3.1_"+selected_area+"_"+xarray_variable_name+"/"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    bT.rio.to_raster(os.path.join(output_path,output_filename))

    # deleting original nc file
    os.remove(os.path.join(save_path,save_filename))






def download_TAMSAT_year(query_year, area, selected_area, save_path):

    end_date = datetime.date(query_year,12,31)
    start_date = datetime.date(query_year,1,1)
    num_days = (end_date-start_date).days
    
    for num_day in tqdm(range(num_days+1)) :

        query_date_loop = datetime.date(query_year,1,1) + datetime.timedelta(days=num_day)

        #print("=== downloading TAMSAT for date",query_date_loop,"===")
        # try:
        download_TAMSAT_day(query_date_loop, save_path)
        crop_and_save_TAMSAT_day(query_date_loop, area, selected_area, save_path)
        # except:
        #     pass




###################
############ CHIRPS
###################




def build_CHIRPS_filename(query_date, selected_area=None):
    # returns filename with proper naming convention

    query_day = query_date.strftime('%d')
    query_month = query_date.strftime('%m')
    query_year = query_date.strftime('%Y')

    if selected_area == None :
        filename = "CHIRPS_v2.0_Africa_"+query_year+"_"+query_month+"_"+query_day
    else :
        filename = "CHIRPS_v2.0_Africa_"+selected_area+"_"+query_year+"_"+query_month+"_"+query_day

    return filename





def download_CHIRPS_day(query_date, download_path="../data/0_downloads/"):
    """
    query_date datetime.eate
    """
    # https://data.chc.ucsb.edu/products/CHIRPS-2.0/africa_daily/tifs/p05/2022/chirps-v2.0.2022.01.11.tif.gz

    # if download_path does not exist, we create it
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    query_day = query_date.strftime('%d')
    query_month = query_date.strftime('%m')
    query_year = query_date.strftime('%Y')

    URL_filename = "chirps-v2.0."+query_year+"."+query_month+"."+query_day+".tif.gz"
    URL_full = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/africa_daily/tifs/p05/"+query_year+"/"+URL_filename

    save_filename = build_CHIRPS_filename(query_date)+".tif.gz"

    # il file already exists, we do not download it
    if os.path.isfile(os.path.join(download_path,save_filename)) == True :
        # print("file already exists. skipping download")
        pass
    else:
        try:
            response = requests.get(URL_full)
            if response.status_code != 404:
                # if status code is different than 404, we download the file
                open(os.path.join(download_path,save_filename), "wb").write(response.content)
            else:
                # hotfix to get images that weren't gzipped during 2021
                print("download : hotfix for bad gzips")
                response = requests.get(URL_full.replace(".tif.gz",".tif"))
                open(os.path.join(download_path,save_filename.replace(".tif.gz",".tif")), "wb").write(response.content)
        except:
            print("error downloading file")





def extract_CHIRPS_data(query_date, origin_path="../data/0_downloads/", dest_path='../data/1_extraction/CHIRPS_v2.0_Africa/'):
    """
    uqery must be datetime.date
    """



    origin_filename = build_CHIRPS_filename(query_date)+".tif.gz"
    origin_full_path = os.path.join(origin_path,origin_filename)

    dest_filename = build_CHIRPS_filename(query_date)+".tif"
    dest_full_path = dest_path + dest_filename

    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    try:
        with gzip.open(origin_full_path, 'rb') as f_in:
            with open(dest_full_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    except:
        # hotfix to get images that weren't gzipped during 2021
        print("extraction : hotfix for bad gzips")
        shutil.copyfile(origin_full_path.replace(".tif.gz",".tif"), dest_full_path)

    # delete original file
    os.remove(origin_full_path)





def crop_and_save_CHIRPS_day(query_date, area, selected_area, save_path="../data/3_output/", extraction_path='../data/1_extraction/CHIRPS_v2.0_Africa/'):

    nc_file_content = rioxarray.open_rasterio(os.path.join(extraction_path,build_CHIRPS_filename(query_date)+".tif"))

    nc_file_content = nc_file_content.rio.clip_box(
        minx=area[selected_area][1],
        miny=area[selected_area][2],
        maxx=area[selected_area][3],
        maxy=area[selected_area][0],
    )

    output_filename = build_CHIRPS_filename(query_date,selected_area)+".tif"
    output_path = os.path.join(save_path,"CHIRPS_v2.0_Africa_"+selected_area)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    nc_file_content.rio.to_raster(os.path.join(output_path,output_filename))

    # delete original file
    os.remove(os.path.join(extraction_path,build_CHIRPS_filename(query_date)+".tif"))





def download_CHIRPS_year(year, area, selected_area, save_path):

    end_date = datetime.date(year,12,31)
    start_date = datetime.date(year,1,1)
    num_days = (end_date-start_date).days
    
    for num_day in tqdm(range(num_days+1)) :
        # try:
        date = datetime.date(year,1,1)+datetime.timedelta(days=num_day)
        download_CHIRPS_day(date)
        extract_CHIRPS_data(date)
        crop_and_save_CHIRPS_day(date, area, selected_area, save_path)
        # except:
        #     print("error with day",num_day)
        #     pass




###################
############ IMERG
###################

def download_IMERG_day(query_date, save_path, username, password):

    query_month = query_date.strftime('%m')
    query_day = query_date.strftime('%d')
    doy = query_date.timetuple().tm_yday


    URL_filename = ("3B-DAY-GIS.MS.MRG.3IMERG."+str(query_date.year)+query_month+query_day+"-S000000-E235959."+f"{30*(doy-1):04d}"+".V06B.tif")
    URL_full = "https://arthurhouhttps.pps.eosdis.nasa.gov/gpmdata/"+str(query_date.year)+"/"+query_month+"/"+query_day+"/gis/"+URL_filename

    # print(URL_full)

    response = requests.get(URL_full, auth=(username, password))
    save_filename = "IMERG_"+URL_filename.replace("/","_")

    # if save_path does not exist, create it
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    open(os.path.join(save_path,save_filename), "wb").write(response.content)




def crop_and_save_IMERG_day(query_date, area, selected_area, save_path):

    query_month = query_date.strftime('%m')
    query_day = query_date.strftime('%d')
    doy = query_date.timetuple().tm_yday
    URL_filename = ("3B-DAY-GIS.MS.MRG.3IMERG."+str(query_date.year)+query_month+query_day+"-S000000-E235959."+f"{30*(doy-1):04d}"+".V06B.tif")
    save_filename = "IMERG_"+URL_filename.replace("/","_")
    nc_file_content = xr.open_dataset(os.path.join(save_path,save_filename))

    xarray_variable_name = "band_data"

    # cropping
    nc_file_content = nc_file_content.where((nc_file_content.y < area[selected_area][0])
                            & (nc_file_content.y > area[selected_area][2])
                            & (nc_file_content.x > area[selected_area][1])
                            & (nc_file_content.x < area[selected_area][3])
                        ).dropna(dim='y', how='all').dropna(dim='x', how='all')

    bT = nc_file_content[xarray_variable_name]
    bT = bT.rio.set_spatial_dims(x_dim='x', y_dim='y')
    bT.rename(new_name_or_name_dict={"x":"lon", "y":"lat"})
    bT.rio.crs
    bT.rio.write_crs("epsg:4326", inplace=True)
    output_filename = "IMERG_"+selected_area+"_"+xarray_variable_name+"_"+str(query_date.year)+"_"+str(query_month)+"_"+str(query_day)+".tif"
    output_path = "../data/3_output/IMERG_"+selected_area+"_"+xarray_variable_name+"/"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    bT.rio.to_raster(os.path.join(output_path,output_filename))

    # delete original file
    os.remove(os.path.join(save_path,save_filename))




def download_IMERG_year(query_year, area, selected_area, save_path, username, password):

    end_date = datetime.date(query_year,12,31)
    start_date = datetime.date(query_year,1,1)
    num_days = (end_date-start_date).days
    
    for num_day in tqdm(range(num_days+1)) :

        query_date_loop = datetime.date(query_year,1,1) + datetime.timedelta(days=num_day)

        #print("=== downloading TAMSAT for date",query_date_loop,"===")
        # try:
        download_IMERG_day(query_date_loop, save_path, username, password)
        crop_and_save_IMERG_day(query_date_loop, area, selected_area, save_path)
        # except:
        #     print("nope")







################### get grid size

from os import listdir
from os.path import isfile, join
import pandas as pd
import rasterio

def build_rainfall_files_df(rainfall_path, date_start, duration):
    """
    This function builds a dataframe containing the list of rainfall files
    from the provided path, and the given date_start and duration.
    
    Helper function used in get_grid_size() and load_TAMSAT_data().

    Args:
        rainfall_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    rainfall_files = [f for f in listdir(rainfall_path) if isfile(join(rainfall_path, f))]
    rainfall_files_df = pd.DataFrame({"filename":rainfall_files}).sort_values("filename").reset_index(drop=True)

    rainfall_files_df["date"] = rainfall_files_df.apply(
        lambda x: datetime.date(
            int(x["filename"].replace(".tif","").split("_")[-3]),
            int(x["filename"].replace(".tif","").split("_")[-2]),
            int(x["filename"].replace(".tif","").split("_")[-1]),
        ),
        axis=1,
    )

    rainfall_files_df = rainfall_files_df[(rainfall_files_df["date"]>=date_start) & (rainfall_files_df["date"]<date_start+datetime.timedelta(days=duration))].reset_index(drop=True)

    return rainfall_files_df




def get_grid_size(rainfall_path, date_start, duration):
    """
    This function loads the list of rainfall files corresponding to the given
    date_start and duration, loads the first rainfall file, and returns its grid
    size, as dimensions of the rainfall grid define the output resolution of the
    model.

    Args:
        TAMSAT_path (_type_): _description_
        date_start (_type_): _description_
        duration (_type_): _description_

    Returns:
        _type_: _description_
    """    

    rainfall_files_df = build_rainfall_files_df(rainfall_path, date_start, duration)

    # checking coherence between date_start and duration and available rainfall data
    if rainfall_files_df["date"].iloc[-1] != date_start+datetime.timedelta(days=duration-1) :
        raise ValueError("The date range may not be covered by the available rainfall data ; please check rainfall entry files.")

    # loading the first rainfall file to get the grid size
    src = rasterio.open(os.path.join(rainfall_path,rainfall_files_df.loc[0,"filename"]))
    array = src.read(1)
    grid_width = array.shape[0]
    grid_height = array.shape[1]

    return grid_width, grid_height


### helper functions to load data into xarrays

def load_TAMSAT_data(data, TAMSAT_path, date_start, duration):
    """
    This function loops over the rainfall raster files, and loads them into a
    xarray DataArray, which is then added to the rain data dictionary. It is
    tailored to the TAMSAT rainfall data files, hence its name.

    Args:
        data (_type_): _description_
        TAMSAT_path (_type_): _description_
        date_start (_type_): _description_
        duration (_type_): _description_

    Returns:
        _type_: _description_
    """

    TAMSAT_files_df = build_rainfall_files_df(TAMSAT_path, date_start, duration)

    for i in range(len(TAMSAT_files_df)):

        dataarray = rioxarray.open_rasterio(os.path.join(TAMSAT_path,TAMSAT_files_df.loc[i,"filename"]))
        dataarray = dataarray.squeeze("band").drop_vars(["band", "spatial_ref"])
        dataarray.attrs = {}

        try:
            dataarray_full = xr.concat([dataarray_full, dataarray],"time")
        except:
            dataarray_full = dataarray

    dataarray_full.rio.write_crs(4326,inplace=True)
    data["rain"] = dataarray_full
    data["rain"].attrs = {"units":"mm", "long_name":"rainfall"}

    return data



def load_lower_res_rain_data(data, rainfall_path, date_start, duration, new_var_name):
    """
    This function loops over the rainfall raster files, and loads them into a
    xarray DataArray, which is then added to the rain data dictionary. It is
    tailored to the TAMSAT rainfall data files, hence its name.

    Args:
        data (_type_): _description_
        TAMSAT_path (_type_): _description_
        date_start (_type_): _description_
        duration (_type_): _description_

    Returns:
        _type_: _description_
    """

    rainfall_files_df = build_rainfall_files_df(rainfall_path, date_start, duration)

    for i in range(len(rainfall_files_df)):

        dataarray = rioxarray.open_rasterio(os.path.join(rainfall_path,rainfall_files_df.loc[i,"filename"]))
        dataarray = dataarray.rio.reproject_match(data, nodata=np.nan)

        
        dataarray = dataarray.squeeze("band").drop_vars(["band", "spatial_ref"])
        dataarray.attrs = {}

        try:
            dataarray_full = xr.concat([dataarray_full, dataarray],"time")
        except:
            dataarray_full = dataarray

    dataarray_full.rio.write_crs(4326,inplace=True)
    data[new_var_name] = dataarray_full
    data[new_var_name].attrs = {"units":"mm", "long_name":"rainfall"}

    return data