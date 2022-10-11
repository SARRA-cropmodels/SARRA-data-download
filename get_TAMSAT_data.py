import datetime
from datetime import datetime as dt
import requests
import xarray as xr 
import schedule
import time
import os




def download_TAMSAT_day(query_date, save_path):

    query_month = query_date.strftime('%m')
    query_day = query_date.strftime('%d')

    URL_filename = str(query_date.year)+"/"+str(query_month)+"/rfe"+str(query_date.year)+"_"+str(query_month)+"_"+str(query_day)+".v3.1.nc"
    URL_full = "http://www.tamsat.org.uk/public_data/data/v3.1/daily/"+URL_filename

    response = requests.get(URL_full)
    save_filename = "TAMSAT_"+URL_filename.replace("/","_")
    open(os.path.join(save_path,save_filename), "wb").write(response.content)




def crop_and_save_TAMSAT_day(query_date, area, selected_area):

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
    output_path = "./data/3_output/TAMSAT_v3.1_"+selected_area+"_"+xarray_variable_name+"/"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    bT.rio.to_raster(os.path.join(output_path,output_filename))




area = {
    'burkina': [16, -6, 9, 3],
    'west_africa':[29, -20, 3.5, 26]}
selected_area = "west_africa"
save_path = "./data/0_downloads/"


def run():
    query_date = dt.today() - datetime.timedelta(days=8)

    for i in range(query_date.day):

        query_date_loop = query_date - datetime.timedelta(days=i)

        print("=== downloading TAMSAT v3.1 for date",query_date_loop.date(),"===")
        download_TAMSAT_day(query_date_loop, save_path)
        crop_and_save_TAMSAT_day(query_date_loop, area, selected_area)

schedule.every().day.at("12:00").do(run)
#schedule.every().minute.do(run) # for testing purposes

# Loop so that the scheduling task
# keeps on running all time.
while True:
 
    # Checks whether a scheduled task
    # is pending to run or not
    schedule.run_pending()
    time.sleep(1)