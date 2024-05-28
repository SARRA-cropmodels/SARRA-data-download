import cdsapi
import zipfile
import xarray as xr 
import rioxarray as rio 
import os
from tqdm import tqdm
import pandas as pd
from datetime import datetime as dt
import datetime
import rasterio
from tqdm import tqdm
import numpy as np



def download_AgERA5_data(area, selected_area, variables, mode="month", query=dt.today(), save_path="../data/"):
    """
    if mode "month", download all days of the month of the datetime.date passed in query
    if mode "year", downloads all days of the year of the datetime.date passed in query    
    by default, is in mode "month", and query is the current date
    """

    print("===== download_AgERA5_data =====")

    # try:
    if mode == "month":
        query_year = query.year # int
        query_month = query.strftime('%m') # str

        print("Mode 'month' acknowledged. Will download data for month",query.month,"/",query.year)
        print("Please note that last available date on AgERA5 should be",dt.today()-datetime.timedelta(days=8))

    elif mode == "year":
        query_year = query.year

        print("Mode 'year' acknowledged. Will download data for year",query.year)
        print("Please note that last available date on AgERA5 should be",(dt.today()-datetime.timedelta(days=8)).date())

    else :
        raise Exception("The mode passed ("+mode+") is incorrect. Please use either 'month' or 'year'")


    c = cdsapi.Client()

    if not os.path.exists(os.path.join(save_path,"0_downloads/")):
        os.makedirs(os.path.join(save_path,"0_downloads/"))

    for variable in variables :

        request = {
                'format': 'zip',
                'day': [
                    '01', '02', '03',
                    '04', '05', '06',
                    '07', '08', '09',
                    '10', '11', '12',
                    '13', '14', '15',
                    '16', '17', '18',
                    '19', '20', '21',
                    '22', '23', '24',
                    '25', '26', '27',
                    '28', '29', '30',
                    '31',
                ],
                'year': [str(query_year)],
                'variable': variable[0],
                'statistic': variable[1],
                'area': area[selected_area],
                'version':'1_1',
            }

        if mode == "month":
            zip_path = os.path.join(save_path,'0_downloads/AgERA5_'+selected_area+'_'+variable[0]+'_'+variable[1]+"_"+str(query_year)+'_'+query_month+'.zip')
            request["month"] = query_month

        if mode == "year": 
            zip_path = os.path.join(save_path,'0_downloads/AgERA5_'+selected_area+'_'+variable[0]+'_'+variable[1]+"_"+str(query_year)+'.zip')
            request["month"] = [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
            ]

        # la requête doit être adaptée pour cette variable
        if variable[0] == "solar_radiation_flux" :
            del request["statistic"]

        c.retrieve(
            'sis-agrometeorological-indicators',
            request,
            zip_path)

        print("Download OK")

    # except :
    #     print("/!\ Download NOT OK")




def extract_AgERA5_data(area, selected_area, variables, mode="month", query=dt.today(), save_path="../data/"):
    """
    mode can be 'month' or 'year'
    uqery must be datetime.date


    """

    print("===== extract_agERA5_month =====")

    try:
        query_year = query.year # int
        query_month = query.strftime('%m') # str

        for variable in tqdm(variables) :

            if mode == "month":
                zip_path = os.path.join(save_path,'0_downloads/AgERA5_'+selected_area+'_'+variable[0]+'_'+variable[1]+"_"+str(query_year)+'_'+query_month+'.zip')
                
            elif mode == "year":
                zip_path = os.path.join(save_path,'0_downloads/AgERA5_'+selected_area+'_'+variable[0]+'_'+variable[1]+"_"+str(query_year)+'.zip')
            
            extraction_path = os.path.join(save_path,'1_extraction/AgERA5_'+selected_area+'/'+str(query_year)+"/"+variable[0]+'_'+variable[1]+'/')
            
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extraction_path)
            except:
                pass
        
        print("Extraction OK")

    except :
        print("/!\ Extraction NOT OK")




def convert_AgERA5_netcdf_to_geotiff(area, selected_area, variables, query=dt.today(), save_path="../data/"):
    """
    converts netcdf to geotiff
    must define which variable to convert, and for which year, as defined with a datetime.date
    """

    print("===== convert_AgERA5_netcdf_to_geotiff =====")

    try:

        query_year = query.year # int
        query_month = query.strftime('%m') # str
        
        for variable in variables :
            
            extraction_path = os.path.join(save_path,'1_extraction/AgERA5_'+selected_area+'/'+str(query_year)+"/"+variable[0]+'_'+variable[1]+'/')
            nc_files = os.listdir(extraction_path)
            conversion_path = os.path.join(save_path,'2_conversion/AgERA5_'+selected_area+'/'+variable[0]+'_'+variable[1]+'/')

            if not os.path.exists(conversion_path):
                os.makedirs(conversion_path)

            for nc_file in tqdm(nc_files) :
                # d'après https://help.marine.copernicus.eu/en/articles/5029956-how-to-convert-netcdf-to-geotiff
                nc_file_content = xr.open_dataset(os.path.join(extraction_path, nc_file))
                xarray_variable_name = list(nc_file_content.keys())[0]
                bT = nc_file_content[xarray_variable_name]
                bT = bT.rio.set_spatial_dims(x_dim='lon', y_dim='lat')
                bT.rio.crs
                bT.rio.write_crs("epsg:4326", inplace=True)

                filename = variable[0]+"_"+variable[1]+"_"+pd.to_datetime(nc_file_content.time.values[0]).strftime('%Y')+"_"+pd.to_datetime(nc_file_content.time.values[0]).strftime('%m')+"_"+pd.to_datetime(nc_file_content.time.values[0]).strftime('%d')+".tif"
                bT.rio.to_raster(os.path.join(conversion_path, filename))


    except :
        print("/!\ Conversion to GeoTIFFs NOT OK")




def calculate_AgERA5_ET0_and_save(area, selected_area, variables, query=dt.today(), save_path="../data/", version="SARRA-Py"):

    print("===== calculate_AgERA5_ET0_and_save =====")

    # try:

    query_year = query.year
    query_month = query.strftime('%m')

    # tmin
    variable = variables[0]
    conversion_path_tmin = os.path.join(save_path,'2_conversion/AgERA5_'+selected_area+"/"+variable[0]+'_'+variable[1]+'/')
    list_files_tmin = [file for file in os.listdir(conversion_path_tmin) if str(query_year) in file]

    # tmax
    variable = variables[1]
    conversion_path_tmax = os.path.join(save_path,'2_conversion/AgERA5_'+selected_area+"/"+variable[0]+'_'+variable[1]+'/')
    list_files_tmax = [file for file in os.listdir(conversion_path_tmax) if str(query_year) in file]

    # irrad
    variable = variables[2]
    conversion_path_irrad = os.path.join(save_path,'2_conversion/AgERA5_'+selected_area+"/"+variable[0]+'_'+variable[1]+'/')
    list_files_irrad = [file for file in os.listdir(conversion_path_irrad) if str(query_year) in file]

    # vapour pressure
    variable = variables[3]
    conversion_path_vp = os.path.join(save_path,'2_conversion/AgERA5_'+selected_area+"/"+variable[0]+'_'+variable[1]+'/')
    list_files_vp = [file for file in os.listdir(conversion_path_vp) if str(query_year) in file]

    # wind
    variable = variables[4]
    conversion_path_wind = os.path.join(save_path,'2_conversion/AgERA5_'+selected_area+"/"+variable[0]+'_'+variable[1]+'/')
    list_files_wind = [file for file in os.listdir(conversion_path_wind) if str(query_year) in file]

    # tmean
    variable = variables[5]
    conversion_path_tmean = os.path.join(save_path,'2_conversion/AgERA5_'+selected_area+"/"+variable[0]+'_'+variable[1]+'/')
    list_files_tmean = [file for file in os.listdir(conversion_path_tmean) if str(query_year) in file]



    # testing if all folders have the same number of files
    len(list_files_tmin) == len(list_files_tmax) == len(list_files_irrad) == len(list_files_vp) == len(list_files_wind) == len(list_files_tmean)




    for i in tqdm(range(len(list_files_irrad))):

        ## on charge les arrays

        ######################## tmin
        img_tmin = rasterio.open(os.path.join(conversion_path_tmin,list_files_tmin[i]))
        arr_tmin = img_tmin.read()
        arr_tmin = arr_tmin - 273.15

        geotiff_path = os.path.join(save_path,'3_output/AgERA5_'+selected_area+"/2m_temperature_24_hour_minimum/")
        

        if not os.path.exists(geotiff_path):
            os.makedirs(geotiff_path)

        new_dataset = rasterio.open(
            geotiff_path+"2m_temperature_24_hour_minimum_"+'_'.join(list_files_tmin[i].split("_")[-3:]),
            'w',
            driver='GTiff',
            height=arr_tmin.shape[1],
            width=arr_tmin.shape[2],
            count=1,
            dtype=arr_tmin.dtype,
            crs=img_tmin.crs,
            transform=img_tmin.transform,
        )

        new_dataset.write(arr_tmin[0,:,:], 1)
        

        ######################## tmax
        img_tmax = rasterio.open(os.path.join(conversion_path_tmax,list_files_tmax[i]))
        arr_tmax = img_tmax.read()
        arr_tmax = arr_tmax - 273.15

        geotiff_path = os.path.join(save_path,'3_output/AgERA5_'+selected_area+"/2m_temperature_24_hour_maximum/")

        if not os.path.exists(geotiff_path):
            os.makedirs(geotiff_path)

        new_dataset = rasterio.open(
            geotiff_path+"2m_temperature_24_hour_maximum_"+'_'.join(list_files_tmax[i].split("_")[-3:]),
            'w',
            driver='GTiff',
            height=arr_tmax.shape[1],
            width=arr_tmax.shape[2],
            count=1,
            dtype=arr_tmax.dtype,
            crs=img_tmax.crs,
            transform=img_tmax.transform,
        )

        new_dataset.write(arr_tmax[0,:,:], 1)

        ######################## irrad
        # J/m²/d
        img_irrad = rasterio.open(os.path.join(conversion_path_irrad,list_files_irrad[i]))
        arr_irrad = img_irrad.read()
        # data is downloaded in J/m²/d
        # pcse needs J/m²/d, no conversion needed

        if version == "SARRA-Py":
            # however SARRA-Py needs kJ/m²/d
            arr_irrad = np.round(arr_irrad / 1000,0) # .astype(int)

        elif version =="SARRA-O":
            # according to doc, SARRA-O needs W/m² i.e. J/m²/d
            # here for the sake of experimentation, we convert to hJ/m²/d
            arr_irrad = np.round(arr_irrad / 100,0) # .astype(int)
            pass

        else:
            # raise exception  
            raise Exception("Version not recognized") 



        geotiff_path = os.path.join(save_path,'3_output/AgERA5_'+selected_area+"/solar_radiation_flux_daily/")

        if not os.path.exists(geotiff_path):
            os.makedirs(geotiff_path)

        new_dataset = rasterio.open(
            geotiff_path+"solar_radiation_flux_daily_"+'_'.join(list_files_irrad[i].split("_")[-3:]),
            'w',
            driver='GTiff',
            height=arr_irrad.shape[1],
            width=arr_irrad.shape[2],
            count=1,
            dtype=arr_irrad.dtype,
            crs=img_irrad.crs,
            transform=img_irrad.transform,
        )

        new_dataset.write(arr_irrad[0,:,:], 1)


        # hPa
        img_vp = rasterio.open(os.path.join(conversion_path_vp,list_files_vp[i]))
        arr_vp = img_vp.read()
        # pcse needs hPa, no conversion needed

        # m/s
        img_wind = rasterio.open(os.path.join(conversion_path_wind,list_files_wind[i]))
        arr_wind = img_wind.read()
        # pcse needs m/s, no conversion needed

        img_tmean = rasterio.open(os.path.join(conversion_path_tmean,list_files_tmean[i]))
        arr_tmean = img_tmean.read()
        arr_tmean = arr_tmean - 273.15

        geotiff_path = os.path.join(save_path,'3_output/AgERA5_'+selected_area+"/2m_temperature_24_hour_mean/")

        if not os.path.exists(geotiff_path):
            os.makedirs(geotiff_path)

        new_dataset = rasterio.open(
            geotiff_path+"2m_temperature_24_hour_mean_"+'_'.join(list_files_tmean[i].split("_")[-3:]),
            'w',
            driver='GTiff',
            height=arr_tmean.shape[1],
            width=arr_tmean.shape[2],
            count=1,
            dtype=arr_tmean.dtype,
            crs=img_tmean.crs,
            transform=img_tmean.transform,
        )

        new_dataset.write(arr_tmean[0,:,:], 1)

        ## on calcule le ET0

        # "When solar radiation data, relative humidity data and/or wind speed data are missing,
        # ETo can be estimated using the Hargreaves ETo equation" in FAO 56

        # providing back the good units for irradiance, from kJ/m²/d to J/m²/d
        if version == "SARRA-Py":
            # converting SARRA-Py kJ/m²/d to MJ/m²/d
            arr_irrad = np.round(arr_irrad / 1000,0) # .astype(int)

        elif version =="SARRA-O":
            # converting SARRA-O hJ/m²/d to MJ/m²/d
            # here for the sake of experimentation, we convert to hJ/m²/d
            arr_irrad = np.round(arr_irrad / 10000,0) # .astype(int)
            pass

        

        coeff = 0.0023
        arr_ET0 = coeff * (arr_tmean + 17.8) * 0.408 * arr_irrad * (abs(arr_tmax - arr_tmin))**0.5

        ## on sauvegarde les geotiffs

        geotiff_path = os.path.join(save_path,'3_output/AgERA5_'+selected_area+"/ET0Hargeaves/")

        if not os.path.exists(geotiff_path):
            os.makedirs(geotiff_path)
        
        # on utilise tmean pour récupérer la date dans le nom de fichier, le crs et le transform
        new_dataset = rasterio.open(
            geotiff_path+"ET0Hargreaves_"+'_'.join(list_files_tmean[i].split("_")[-3:]),
            'w',
            driver='GTiff',
            height=arr_ET0.shape[1],
            width=arr_ET0.shape[2],
            count=1,
            dtype=arr_ET0.dtype,
            crs=img_tmean.crs,
            transform=img_tmean.transform,
        )

        new_dataset.write(arr_ET0[0,:,:], 1)

    # except:
    #     print("/!\ Calculation of ET0 NOT OK")




variables = [
    ("2m_temperature","24_hour_minimum"),
    ("2m_temperature","24_hour_maximum"),
    ("solar_radiation_flux", "daily"),
    ("vapour_pressure", "24_hour_mean"),
    ("10m_wind_speed", "24_hour_mean"),
    ("2m_temperature","24_hour_mean"),
]

def download_AgERA5_year(query_year, area, selected_area, save_path, version):
    query_date = datetime.date(query_year,1,1)
    download_AgERA5_data(area, selected_area, variables, mode="year", query=query_date)
    extract_AgERA5_data(area, selected_area, variables, mode="year", query=query_date)
    convert_AgERA5_netcdf_to_geotiff(area, selected_area, variables, query=query_date) 
    calculate_AgERA5_ET0_and_save(area, selected_area, variables, query=query_date, version=version)
    print("===== Query date",query_date,"all done ! =====")