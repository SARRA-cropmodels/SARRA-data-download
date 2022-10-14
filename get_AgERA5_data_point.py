import cdsapi
import zipfile
import xarray as xr 
# import rioxarray as rio 
import os
from tqdm import tqdm
import pandas as pd
from datetime import datetime as dt
import datetime
# import rasterio
# from rasterio.plot import show
import numpy as np
import schedule
import requests
from pcse.util import reference_ET




def convert_point_to_area(points, selected_point):
    print("===== convert_point_to_area =====")
    #fonctionne seulement dans le nord ?
    area = points
    # area[selected_point] = [np.round(points[selected_point][0]*4,0)/4,
    #                         np.round(points[selected_point][1]*4,0)/4-0.25,
    #                         np.round(points[selected_point][0]*4,0)/4-0.25,
    #                         np.round(points[selected_point][1]*4,0)/4,]
    area[selected_point] = [np.round(np.round(points[selected_point][0]*10,0)/10+0.1,2),
                            np.round(np.round(points[selected_point][1]*10,0)/10-0.1,2),
                            np.round(np.round(points[selected_point][0]*10,0)/10-0.1,2),
                            np.round(np.round(points[selected_point][1]*10,0)/10+0.1,2),]
    return area, selected_point




def download_AgERA5_year(selected_area, variables, query_year):

    # the objective is to download the whole year

    print("===== download_AgERA5_year =====")

    # try:
    

    c = cdsapi.Client()

    if not os.path.exists('./data/0_downloads/'):
        os.makedirs('./data/0_downloads/')

    for variable in variables :
        
        print("\nDownloading values for variable",variable,"for year", query_year)

        zip_path = './data/0_downloads/AgERA5_'+selected_area+'_'+variable[0]+'_'+variable[1]+"_"+str(query_year)+'.zip'

        data_points = {}

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
                'month': [
                    '01', '02', '03',
                    '04', '05', '06',
                    '07', '08', '09',
                    '10', '11', '12',
                ],
                'year': [str(query_year)],
                'variable': variable[0],
                'statistic': variable[1],
                'area': area[selected_area],
            }

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




def extract_agERA5_year(selected_area, variables, query_year):

    print("===== extract_agERA5_year =====")

    # try:
    # query_year = query_date.year
    # query_month = query_date.strftime('%m')

    for variable in tqdm(variables) :

        zip_path = './data/0_downloads/AgERA5_'+selected_area+'_'+variable[0]+'_'+variable[1]+"_"+str(query_year)+'.zip'
        extraction_path = './data/1_extraction/AgERA5_'+selected_area+'_'+variable[0]+'_'+variable[1]+"_"+str(query_year)+'/'

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extraction_path)
        except:
            pass
    
    print("Extraction OK")

    # except :
    #     print("/!\ Extraction NOT OK")




def format_and_save_AgERA_year(selected_area, variables, query_year):

    print("===== format_and_save_AgERA_year =====")

    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    print("- Reading netCDF files to Pandas dataframe...")

    df_weather_collection = {}
    df_weather = pd.DataFrame()

    for variable in variables :
        
        df_weather_collection[variable[0]] = pd.DataFrame()

        extraction_path = './data/1_extraction/AgERA5_'+selected_area+'_'+variable[0]+'_'+variable[1]+"_"+str(query_year)+'/'
        nc_files = os.listdir(extraction_path)
        conversion_path = './data/2_conversion/AgERA5_'+selected_area+'_'+variable[0]+'_'+variable[1]+"_"+str(query_year)+'/'
        if not os.path.exists(conversion_path):
            os.makedirs(conversion_path)
        for nc_file in tqdm(nc_files) :
            # d'après https://help.marine.copernicus.eu/en/articles/5029956-how-to-convert-netcdf-to-geotiff
            nc_file_content = xr.open_dataset(os.path.join(extraction_path, nc_file))
            xarray_variable_name = list(nc_file_content.keys())[0]
            bT = nc_file_content[xarray_variable_name]

            value = nc_file_content.sel(
                lat=points[selected_point][0], 
                lon=points[selected_point][1],
                method="nearest")[xarray_variable_name].values[0]

            df_weather_collection[variable[0]] = df_weather_collection[variable[0]].append(pd.DataFrame({variable:value, "Jour":pd.Timestamp(nc_file_content["time"].values[0]).date()}, index=[0]))

        if variable == variables[0]:
            df_weather = df_weather_collection[variable[0]]
        else :
            df_weather = df_weather.merge(df_weather_collection[variable[0]], left_on = "Jour", right_on="Jour")

    df_weather = df_weather.rename(columns=variables_corresp)

    print("- Converting units...")

    # calculating variables
    df_weather[["TMin", "TMax", "TMoy"]] = df_weather[["TMin", "TMax", "TMoy"]] - 273.15
    df_weather["Rg"] = df_weather["Rg"]/1E6
    # calcul du RH depuis actual vapour pressure : https://www.weather.gov/media/epz/wxcalc/vaporPressure.pdf
    df_weather["es"] = 6.11 * 10 ** ((7.5 * df_weather["TMoy"])/(237.3 + df_weather["TMoy"]))
    df_weather["HMoy"] = (df_weather["Vap"]/df_weather["es"]) * 100

    print("- Retrieving elevation at request coordinates...")

    response = requests.get("https://api.open-elevation.com/api/v1/lookup?locations="+str(points[selected_point][0])+","+str(points[selected_point][1]))
    df_weather["ELEV"] = response.json()['results'][0]["elevation"]

    print("- Computing ET0-PM...")

    ANGSTA = 0.29
    ANGSTB = 0.49
    df_weather["ET0_PM"] = df_weather.apply(lambda x: reference_ET(x["Jour"], points[selected_point][0], x["ELEV"], x["TMin"], x["TMax"], x["Rg"]*1E6, x["Vap"], x["Vt"], ANGSTA, ANGSTB, ETMODEL="PM")[2], axis=1)

    print("- Saving...")

    df_weather.to_csv("./data/3_output/AgERA5_point_"+selected_point+"_"+str(points[selected_point][0])+"_"+str(points[selected_point][1])+"_weather_"+str(query_year)+".csv")

    print("- Done !")




def delete_AgERA5_intermediate_files(dump):
    print("===== delete_AgERA5_intermediate_files =====")
    try:
        if dump == True:
            import shutil
            shutil.rmtree( './data/0_downloads/' )
            shutil.rmtree( './data/1_extraction/' )
            shutil.rmtree( './data/2_conversion/' )
        print("Deletion of intermediate files OK")
    except:
        print("/!\ Deletion of intermediate files NOT OK")




# parameters
points = {'bobo_dioulasso': [11.18, -4.28]} #lat, lon
selected_point = "bobo_dioulasso"
variables = [
    ("2m_temperature","24_hour_minimum"),
    ("2m_temperature","24_hour_maximum"),
    ("solar_radiation_flux", "daily"),
    ("vapour_pressure", "24_hour_mean"),
    ("10m_wind_speed", "24_hour_mean"),
    ("2m_temperature","24_hour_mean"),
]

# correspondance entre noms de variables dans AgERA5 et le nom des variables souhaité dans SARRA
variables_corresp = {
    ("2m_temperature","24_hour_minimum"): "TMin",
    ("2m_temperature","24_hour_maximum"): "TMax",
    ("solar_radiation_flux", "daily"): "Rg",
    ("vapour_pressure", "24_hour_mean"): "Vap",
    ("10m_wind_speed", "24_hour_mean"): "Vt",
    ("2m_temperature","24_hour_mean"): "TMoy",
}
query_year = 2020






area, selected_area = convert_point_to_area(points, selected_point)
download_AgERA5_year(selected_area, variables, query_year)
extract_agERA5_year(selected_area, variables, query_year)
format_and_save_AgERA_year(selected_area, variables, query_year)
delete_AgERA5_intermediate_files(True)




