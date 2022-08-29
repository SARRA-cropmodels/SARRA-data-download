#!/usr/bin/env python
"""
script for converting data from netcdf to geotiff
"""
# import time, sys
# from datetime import datetime, timedelta
# import cdsapi
#
from netCDF4 import Dataset, date2num, num2date
# import numpy as np
# import rioxarray
# import xarray
import sys
import os
from datetime import datetime, timedelta
from osgeo import gdal, osr, gdal_array
import xarray as xr
import numpy as np
import AGRHYMET_Tools.config as cg
import pandas as pd
from pathlib import Path
"""
Declarration des constantes
"""

def main():
    area = str(sys.argv[1])
    daily_path_data_in = Path(cg.bdir) / area  / "2_aggregation" / "2_data_ERA5_dailyAg"
    daily_path_data_out = Path(cg.bdir) / area  / "tif_data" / "daily"
    path_data_in = daily_path_data_in
    path_data_out = daily_path_data_out
    files = os.listdir(path_data_in)
    for file in files:
        if file.endswith('.nc'):
            print(file)
            mes_var = file.split('_')
            model_string = mes_var[0]
            my_date = mes_var[1]
            var_name = mes_var[2]
            end_string = mes_var[3]
            in_filename = str(path_data_in /  file) #'%s%s' % (path_data_in, file)
            var_name_folder = Path(path_data_out) /   var_name 
            os.makedirs(var_name_folder, exist_ok=True)
            # if not os.path.exists(var_name_folder):
            #     os.mkdir(var_name_folder)

            out_filename = '%s%sERA5_%s_%s.tif' % (var_name_folder, '/', var_name, my_date)
            print(in_filename)
            print(out_filename)
            # correct data
            # ncks -C -O -x -v expver ERA5_202006_accmnmx.nc ERA5_202006_accmnmx10.nc
            # ncwa -a expver ERA5_202008_accmnmx.nc ERA5_202008_accmnmx_new.nc
            # ncks -x -v var1,var2 in.nc out.nc
            # ncks -O -x -v Band,my_plottable_variable new_misr.nc mytest.nc
            os.system('gdal_translate -a_srs EPSG:4326 -if netCDF -of GTIFF ' + in_filename + ' ' + out_filename)
            # os.system('gdal_translate -a_srs EPSG:4326 -if netCDF -of GTIFF  -ot Float64' + in_filename + ' ' +
            # out_filename)

if __name__ == '__main__':
    main()


