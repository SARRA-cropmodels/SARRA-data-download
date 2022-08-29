#!/usr/bin/env python
# export PYTHONPATH="${PYTHONPATH}:/SARRAH/agrhymet_tools/py_tools/cra_era5_tools/"
# !/home/sarrah/.conda/envs/cra_era5_tools/bin/python
import sys
import os
print('Number of arguments: {}'.format(len(sys.argv)))
print('Argument(s) passed: {}'.format(str(sys.argv)))

import datetime as dt
import pandas as pd
import cdsapi
# from AGRHYMET_Tools import config as cg
import AGRHYMET_Tools.config as cg
from pathlib import Path
from dateutil.relativedelta import relativedelta
# import config as cg

c = cdsapi.Client()
# c = cdsapi.Client(timeout=600,quiet=False,debug=True)

def get_first_last_dayofmonth(year, month):
    if month == 12:
        startday = dt.date(year, month, 1)
        endday = dt.date(year, month, 31)
    else:
        startday = dt.date(year, month, 1)
        endday = dt.date(year, month + 1, 1) - dt.timedelta(days=1)
    return startday.strftime('%Y%m%d'), endday.strftime('%Y%m%d')

start, end = '2017-08', '2017-09'
start, end = str(sys.argv[1]), str(sys.argv[2])


newt_start = dt.datetime.strptime(start, '%Y-%m') - relativedelta(months=1)
start = newt_start.strftime("%Y-%m")

newt_end = dt.datetime.strptime(end, '%Y-%m') + relativedelta(months=1)
end = newt_end.strftime("%Y-%m")

area = str(sys.argv[3])

outdir = Path(cg.bdir) / area  / "0_download" / "0_data_ERA5_daily"
if not os.path.exists(outdir):
    os.makedirs(outdir)

print((pd.date_range(start, end, freq='M')))

for mon in pd.date_range(start, end, freq='M'):
    period = get_first_last_dayofmonth(mon.year, mon.month)
    period_fmt = f"{period[0]}/to/{period[1]}"
    # period_fmt= f"20151231/to/20160131"
    print((pd.date_range(start, end, freq='M')))
    # period_fmt = '20200701/to/20200720'
    print(period)
    print(period_fmt)


    print("min max temp rain")
    fn = f'ERA5_{period_fmt[0:6]}_accmnmx.nc'
    print(outdir, fn)
    if os.path.exists(f'{outdir / fn}'):
        print("data allready downloaded")
    else:
        r = c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'variable': ['maximum_2m_temperature_since_previous_post_processing',
                            'minimum_2m_temperature_since_previous_post_processing',
                            'surface_solar_radiation_downwards', 'total_precipitation'],
                'product_type': 'reanalysis',
                'dates': period_fmt,
                "area": cg.area[area],
                'time': [
                    '00:00', '01:00', '02:00',
                    '03:00', '04:00', '05:00',
                    '06:00', '07:00', '08:00',
                    '09:00', '10:00', '11:00',
                    '12:00', '13:00', '14:00',
                    '15:00', '16:00', '17:00',
                    '18:00', '19:00', '20:00',
                    '21:00', '22:00', '23:00'
                ],
                'grid': '0.25/0.25',
                'format': 'netcdf'
            })
        r.download(f'{outdir / fn}')

    print("wind dewpoint temp and co")
    fn = f'ERA5_{period_fmt[0:6]}_inst1.nc'
    print(outdir, fn)
    if os.path.exists(f'{outdir / fn}'):
        print("data allready downloaded")
    else:
        r = c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'variable': ['10m_u_component_of_wind', '10m_v_component_of_wind', '2m_dewpoint_temperature',
                            '2m_temperature', 'total_cloud_cover'],
                'product_type': 'reanalysis',
                'dates': period_fmt,
                "area": cg.area[area],
                'time': [
                    '00:00',
                    '03:00',
                    '06:00',
                    '09:00',
                    '12:00',
                    '15:00',
                    '18:00',
                    '21:00',
                ],
                'grid': '0.25/0.25',
                'format': 'netcdf'
            })
        r.download(f'{outdir / fn}')
    

print("Download finit")

exit()
