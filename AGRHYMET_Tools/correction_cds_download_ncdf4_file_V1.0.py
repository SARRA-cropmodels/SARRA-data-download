#!/usr/bin/env python
"""
Save as file calculate-daily-tp.py and run "python calculate-daily-tp.py".
  
Input file : tp_20170101-20170102.nc
Output file: daily-tp_20170101.nc
"""
import os
import AGRHYMET_Tools.config as cg

# from nco import Nco

# nco=Nco().nco_path()
# print(nco)
# print('test')
#
#

download_path_data_in = cg.bdir + '0_download/0_data_ERA5_daily/'

download_path_data_out = cg.bdir + '0_download/0_data_ERA5_daily_corrected/'


# Affectation -dekad or daily

path_data_in = download_path_data_in
path_data_out = download_path_data_out

files = os.listdir(path_data_in)
for file in files:
    if file.endswith('.nc'):
        print(file)
        # mes_var = file.split('_')
        # model_string = mes_var[0]
        # my_date = mes_var[1]
        # var_name = mes_var[2]
        # end_string = mes_var[3]
        in_filename = '%s%s' % (path_data_in, file)
        out_filename = '%s%s' % (path_data_out, file)
        os.makedirs(path_data_out, exist_ok=True)
        os.system('ncwa -a expver ' + in_filename + ' ' + out_filename)

        # correct data
        # ncks -C -O -x -v expver ERA5_202006_accmnmx.nc ERA5_202006_accmnmx10.nc
        # ncwa -a expver ERA5_202008_accmnmx.nc ERA5_202008_accmnmx_new.nc
        # ncks -x -v var1,var2 in.nc out.nc
        # ncks -O -x -v Band,my_plottable_variable new_misr.nc mytest.nc
        # os.system('gdal_translate -a_srs EPSG:4326 -if netCDF -of GTIFF ' + in_filename + ' ' + out_filename)

        #
        # os.system('gdal_translate -a_srs EPSG:4326 -if netCDF -of GTIFF  -ot Float64' + in_filename + ' ' +
        # out_filename)
