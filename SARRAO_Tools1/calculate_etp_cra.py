#!/usr/bin/env python
"""
Save as file calculate-daily-tp.py and run "python calculate-daily-tp.py".
  
Input file : tp_20170101-20170102.nc
Output file: daily-tp_20170101.nc
"""
import time, sys
from datetime import datetime, timedelta
import cdsapi

from netCDF4 import Dataset, date2num, num2date
import numpy as np
from .fao import fao56_penman_monteith


day1 = 20170101
d = datetime.strptime(str(day1), '%Y%m%d')
f_in1 = 'D:/2020/SARRA_O/python_script/era5_data/tp_%d-%s.nc' % (day1, (d + timedelta(days=1)).strftime('%Y%m%d'))
f_out1 = 'D:/2020/SARRA_O/python_script/era5_data/daily-max_%d.nc' % day1
ma_variable1 = 'tp'


def daily_etp(day, f_in, f_out, ma_variable):
    # day = 20170101
    d = datetime.strptime(str(day), '%Y%m%d')
    # f_in = 'tp_%d-%s.nc' % (day1, (d + timedelta(days = 1)).strftime('%Y%m%d'))
    # f_out = 'daily-tp_%d.nc' % day
    #
    time_needed = []
    for i in range(1, 25):
        time_needed.append(d + timedelta(hours=i))

    with Dataset(f_in) as ds_src:
        var_time = ds_src.variables['time']
        time_avail = num2date(var_time[:], var_time.units,
                              calendar=var_time.calendar)

        indices = []
        for tm in time_needed:
            a = np.where(time_avail == tm)[0]
            if len(a) == 0:
                sys.stderr.write('Error: precipitation data is missing/incomplete - %s!\n'
                                 % tm.strftime('%Y%m%d %H:%M:%S'))
                sys.exit(200)
            else:
                print('Found %s' % tm.strftime('%Y%m%d %H:%M:%S'))
                indices.append(a[0])

        var_tp = ds_src.variables[ma_variable]
        tp_values_set = False
        for idx in indices:
            if not tp_values_set:
                data = var_tp[idx, :, :]
                tp_values_set = True
            else:
                data = np.maximum(data, var_tp[idx, :, :])

        with Dataset(f_out, mode='w', format='NETCDF3_64BIT_OFFSET') as ds_dest:
            # Dimensions
            for name in ['latitude', 'longitude']:
                dim_src = ds_src.dimensions[name]
                ds_dest.createDimension(name, dim_src.size)
                var_src = ds_src.variables[name]
                var_dest = ds_dest.createVariable(name, var_src.datatype, (name,))
                var_dest[:] = var_src[:]
                var_dest.setncattr('units', var_src.units)
                var_dest.setncattr('long_name', var_src.long_name)

            ds_dest.createDimension('time', None)
            var = ds_dest.createVariable('time', np.int32, ('time',))
            time_units = 'hours since 1900-01-01 00:00:00'
            time_cal = 'gregorian'
            var[:] = date2num([d], units=time_units, calendar=time_cal)
            var.setncattr('units', time_units)
            var.setncattr('long_name', 'time')
            var.setncattr('calendar', time_cal)

            # Variables
            var = ds_dest.createVariable(var_tp.name, np.double, var_tp.dimensions)
            var[0, :, :] = data
            var.setncattr('units', var_tp.units)
            var.setncattr('long_name', var_tp.long_name)

            # Attributes
            ds_dest.setncattr('Conventions', 'CF-1.6')
            ds_dest.setncattr('history', '%s %s'
                              % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                 ' '.join(time.tzname)))

            print('Done! Daily total precipitation saved in %s' % f_out)



daily_max(day1, f_in1, f_out1,ma_variable1)
