import cdsapi
import zipfile
import xarray as xr 
import os
from tqdm import tqdm
import pandas as pd
from datetime import datetime as dt
import datetime
import numpy as np
import schedule
import requests
from pcse.util import reference_ET
from os.path import exists




def download_AgERA5_year(area, selected_area, variable, query_year, verbose=False):

    # the objective is to download the whole year
    if verbose : print("===== download_AgERA5_year =====")
    
    c = cdsapi.Client()

    if not os.path.exists('./data/0_downloads/'):
        os.makedirs('./data/0_downloads/')

    if verbose : print("Downloading values for variable",variable,"for year", query_year)

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

    if exists(zip_path) :
        if verbose : print("File already downloaded. Skipping.")

    else :
        c.retrieve(
            'sis-agrometeorological-indicators',
            request,
            zip_path)

        if verbose : print("Download OK")

    # except :
    #     print("/!\ Download NOT OK")




def extract_agERA5_year(area, selected_area, variable, query_year, verbose=False):

    if verbose : print("===== extract_agERA5_year =====")

    # try:
    # query_year = query_date.year
    # query_month = query_date.strftime('%m')


    zip_path = './data/0_downloads/AgERA5_'+selected_area+'_'+variable[0]+'_'+variable[1]+"_"+str(query_year)+'.zip'
    extraction_path = './data/1_extraction/AgERA5_'+selected_area+'/'+str(query_year)+"/"+variable[0]+'_'+variable[1]+'/'

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extraction_path)
    except:
        pass
    
    if verbose : print("Extraction OK")

    # except :
    #     print("/!\ Extraction NOT OK")




def read_AgERA5_point_values(variables, year_begin, year_end, selected_area, points, verbose=False):

    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    df_weather = pd.DataFrame()

    for variable in tqdm(variables, position=0, desc="reading nc files") :
        
        df_weather_variable = pd.DataFrame()

        for query_year in range(year_begin, year_end+1):
            # print(variable, query_year)

            extraction_path = './data/1_extraction/AgERA5_'+selected_area+'/'+str(query_year)+"/"+variable[0]+'_'+variable[1]+'/'
            nc_files = os.listdir(extraction_path)

            for nc_file in tqdm(nc_files, position=1,  desc=variable[0]+"_"+variable[1], leave=False) :
                nc_file_content = xr.open_dataset(os.path.join(extraction_path, nc_file))
                xarray_variable_name = list(nc_file_content.keys())[0]
                bT = nc_file_content[xarray_variable_name]

                for point in points :

                    value = nc_file_content.sel(
                        lat=points[point][0], 
                        lon=points[point][1],
                        method="nearest")[xarray_variable_name].values[0]
                    df_weather_variable = df_weather_variable.append(pd.DataFrame({variable:value, "Point":point, "Jour":pd.Timestamp(nc_file_content["time"].values[0]).date()}, index=[0]))
            
        if variable == variables[0]:
            df_weather = df_weather_variable
        else :
            df_weather = df_weather.merge(df_weather_variable, left_on = ["Jour", "Point"], right_on=["Jour", "Point"])
    
    return df_weather





def format_and_save_AgERA5_point_csv(df_weather, points, year_begin, year_end, verbose=False) :

    # correspondance entre noms de variables dans AgERA5 et le nom des variables souhaité dans SARRA
    variables_corresp = {
        ("2m_temperature","24_hour_minimum"): "TMin",
        ("2m_temperature","24_hour_maximum"): "TMax",
        ("solar_radiation_flux", "daily"): "Rg",
        ("vapour_pressure", "24_hour_mean"): "Vap",
        ("10m_wind_speed", "24_hour_mean"): "Vt",
        ("2m_temperature","24_hour_mean"): "TMoy",
    }

    df_weather = df_weather.rename(columns=variables_corresp)

    if verbose : print("- Converting units...")

    # conversions
    df_weather[["TMin", "TMax", "TMoy"]] = df_weather[["TMin", "TMax", "TMoy"]] - 273.15 # K to °C
    df_weather["Rg"] = df_weather["Rg"]/1E6 # J/d to MJ/d

    # calcul du RH depuis actual vapour pressure : https://www.weather.gov/media/epz/wxcalc/vaporPressure.pdf
    df_weather["es"] = 6.11 * 10 ** ((7.5 * df_weather["TMoy"])/(237.3 + df_weather["TMoy"]))
    df_weather["HMoy"] = (df_weather["Vap"]/df_weather["es"]) * 100

    # SARRA-O day format
    df_weather["Jour_SARRA_H"] = df_weather.apply(lambda x: x["Jour"].strftime("%d/%m/%Y"), axis=1)

    if verbose : print("- Retrieving elevation at request coordinates...")
    df_elevation = pd.DataFrame()
    for point in points :
        response = requests.get("https://api.open-elevation.com/api/v1/lookup?locations="+str(points[point][0])+","+str(points[point][1]))
        df_elevation = df_elevation.append(pd.DataFrame({"Point":point, "lat":points[point][0], "lon":points[point][1], "ELEV":response.json()['results'][0]["elevation"]}, index=[0]))

    df_weather = df_weather.merge(df_elevation, left_on="Point", right_on="Point")


    if verbose : print("- Computing ET0-PM...")
    ANGSTA = 0.29
    ANGSTB = 0.49
    df_weather["ET0_PM"] = df_weather.apply(lambda x: reference_ET(x["Jour"], x["lat"], x["ELEV"], x["TMin"], x["TMax"], x["Rg"]*1E6, x["Vap"], x["Vt"], ANGSTA, ANGSTB, ETMODEL="PM")[2], axis=1)


    # saving individual files
    for point in points :
        df_weather[df_weather["Point"]==point].reset_index(drop=True).to_csv("./data/3_output/AgERA5_point_"+selected_area+"_"+point+"_"+str(points[point][0])+"_"+str(points[point][1])+"_"+str(year_begin)+"_"+str(year_end)+".csv")




# parameters
area = {
    'madagascar':[-11.3, 42.1, -26.2, 51.1],
}
selected_area = "madagascar"

points = {
    "Antsahamamy":[-18.92, 47.56],
    "Ambohimiarina":[-18.83, 47.13],
    "Ambohitsilaozana":[-17.70, 48.47],
    "Ambongabe":[-18.53, 48.03],
    "Ampitatsimo":[-17.81, 48.38],
} #lat, lon

variables = [
    ("2m_temperature","24_hour_minimum"),
    ("2m_temperature","24_hour_maximum"),
    ("solar_radiation_flux", "daily"),
    ("vapour_pressure", "24_hour_mean"),
    ("10m_wind_speed", "24_hour_mean"),
    ("2m_temperature","24_hour_mean"),
]

year_begin = 2006
year_end = 2010




for variable in tqdm(variables, position=0, desc="download/unzip AgERA5 data") :
    for query_year in tqdm(range(year_begin, year_end+1), position=1, leave=False):
        download_AgERA5_year(area, selected_area, variable, query_year)
        extract_agERA5_year(area, selected_area, variable, query_year)

df_weather = read_AgERA5_point_values(variables, year_begin, year_end, selected_area, points)
format_and_save_AgERA5_point_csv(df_weather, points, year_begin, year_end)