#!/usr/bin/env python
"""
Save as file calculate-daily-tp.py and run "python calculate-daily-tp.py".
  
Input file : tp_20170101-20170102.nc
Output file: daily-tp_20170101.nc
"""

from netCDF4 import Dataset

# nco=Nco().nco_path()
# print(nco)
# print('test')
#
#


# fao56_penman_monteith(net_rad, t, ws, svp, avp, delta_svp, psy, shf=0.0)

# day1 = 20170101
# d = datetime.strptime(str(day1), '%Y%m%d')
# f_in1 = 'D:/2020/SARRA_O/python_script/era5_data/tp_%d-%s.nc' % (day1, (d + timedelta(days=1)).strftime('%Y%m%d'))
# f_out1 = 'D:/2020/SARRA_O/python_script/era5_data/daily-max_%d.nc' % day1
# ma_variable1 = 'tp'
#

path_data_in = 'D:/2020/SARRA_O/data/0_download/0_data_ERA5_daily/'
path_data_out = 'D:/2020/SARRA_O/data/2_aggregation/2_data_ERA5_dailyEto/'
# ncks -C -O -x -v expver ERA5_202006_accmnmx.nc ERA5_202006_accmnmx10.nc
# ncwa -a expver ERA5_202008_accmnmx.nc ERA5_202008_accmnmx_new.nc
# ncks -x -v var1,var2 in.nc out.nc
# ncks -O -x -v Band,my_plottable_variable new_misr.nc mytest.nc



my_date = 20200103
my_month = 202006
netcdf4_download_file = '%sERA5_%d_accmnmx.nc' % (path_data_in, my_month)
print(netcdf4_download_file)
netcdf4_download_file_cor = '%sERA5_%s_accmnmx_cor.nc' % (path_data_in, my_month)
print(netcdf4_download_file_cor)

ds_nc_download_file = Dataset(netcdf4_download_file)
#
print('ds_nc_download_file')
print(ds_nc_download_file)
longitude = ds_nc_download_file['longitude']
latitude = ds_nc_download_file['latitude']
mx2t = ds_nc_download_file['mx2t']
mn2t = ds_nc_download_file['mn2t']
ssrd = ds_nc_download_file['ssrd']
tp = ds_nc_download_file['tp']
time = ds_nc_download_file['time']
print(time)
print(ssrd)

ds_download_file_cor = Dataset(netcdf4_download_file_cor, 'w', format='NETCDF4_CLASSIC')
for name in ['latitude', 'longitude', 'time']:
    # Gestion des variables 'latitude', 'longitude'
    dim_src = ds_nc_download_file.dimensions[name]
    ds_download_file_cor.createDimension(name, dim_src.size)
    var_src = ds_nc_download_file.variables[name]
    var_dest = ds_download_file_cor.createVariable(name, var_src.datatype, (name,))
    var_dest[:] = var_src[:]
    var_dest.setncattr('units', var_src.units)
    var_dest.setncattr('long_name', var_src.long_name)

# #Gestion de la variable time
# dim_src = ds_nc_download_file.dimensions['time']
# ds_download_file_cor.createDimension(name, dim_src.size)
# var_src = ds_nc_download_file.variables['time']
# var_dest = ds_download_file_cor.createVariable('time', var_src.datatype, ('time',))
# var_dest[:] = var_src[:]
# var_dest.setncattr('units', var_src.units)
# var_dest.setncattr('long_name', var_src.long_name)

# Gestion de la variable ssrd
for name in ['ssrd', 'mx2t', 'mn2t', 'tp']:
    # dim_src = ds_nc_download_file.dimensions[name]
    # print(dim_src)
    # ds_download_file_cor.createDimension(name, dim_src.size)
    var_src = ds_nc_download_file.variables[name]
    print('var_src.datatype')
    print(var_src.datatype)
    var_dest = ds_download_file_cor.createVariable(name, var_src.datatype, ('time', 'latitude', 'longitude'))
    print('salut_a_tous')
    var_dest[:, :, :] = var_src[:, 1, :, :]
    var_dest.setncattr('units', var_src.units)
    var_dest.setncattr('long_name', var_src.long_name)

# close ncfile
ds_download_file_cor.close()
ds_nc_download_file.close()

# with Dataset(netcdf4_download_file_cor, mode='w', format='NETCDF3_64BIT_OFFSET') as ds_dest:
#     # Dimensions
#     for name in ['latitude', 'longitude']:
#         dim_src = ds_nc_download_file.dimensions[name]
#         ds_dest.createDimension(name, dim_src.size)
#         var_src = ds_nc_download_file.variables[name]
#         var_dest = ds_dest.createVariable(name, var_src.datatype, (name,))
#         var_dest[:] = var_src[:]
#         var_dest.setncattr('units', var_src.units)
#         var_dest.setncattr('long_name', var_src.long_name)
#
#     # ds_dest.createDimension('time', None)
#     # var = ds_dest.createVariable('time', np.int32, ('time',))
#     # time_units = 'hours since 1900-01-01 00:00:00'
#     # time_cal = 'gregorian'
#     # var[:] = date2num([d], units=time_units, calendar=time_cal)
#     # var.setncattr('units', time_units)
#     # var.setncattr('long_name', 'time')
#     # var.setncattr('calendar', time_cal)
#     #
#     # # Variables
#     # var = ds_dest.createVariable(var_tp.name, np.double, var_tp.dimensions)
#     # var[0, :, :] = data
#     # var.setncattr('units', var_tp.units)
#     # var.setncattr('long_name', var_tp.long_name)
#     #
#     # # # Attributes
#     # # ds_dest.setncattr('Conventions', 'CF-1.6')
#     # # ds_dest.setncattr('history', '%s %s'
#     # #                   % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#     # #                      ' '.join(time.tzname)))
#     # #
#     # # print('Done! Daily total precipitation saved in %s' % f_out)
#     # #
#     # #
#     # #
