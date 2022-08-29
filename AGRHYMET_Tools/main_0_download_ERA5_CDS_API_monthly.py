#!/usr/bin/env python
# export PYTHONPATH="${PYTHONPATH}:/SARRAH/agrhymet_tools/py_tools/cra_era5_tools/"
# !/home/sarrah/.conda/envs/cra_era5_tools/bin/python
import sys

print('Number of arguments: {}'.format(len(sys.argv)))
print('Argument(s) passed: {}'.format(str(sys.argv)))


import datetime as dt
import pandas as pd
import cdsapi
# from AGRHYMET_Tools import config as cg
import AGRHYMET_Tools.config as cg
# import config as cg


c = cdsapi.Client()

def get_first_last_dayofmonth(year, month):
    if month == 12:
        startday = dt.date(year, month, 1)
        endday = dt.date(year, month, 31)
    else:
        startday = dt.date(year, month, 1)
        endday = dt.date(year, month + 1, 1) - dt.timedelta(days=1)
    return startday.strftime('%Y%m%d'), endday.strftime('%Y%m%d')


# outdir = '/Volumes/Flux_v2/marsop5/'
# outdir = '/Volumes/VolumeWork/Marsop5/ERA5_CDSAPI_mir_monthly/'
# outdir = '/Volumes/FETT_v2/ERA5_monthly/'


outdir = cg.bdir + '0_download/0_data_ERA5_daily/'

start, end = '2017-08', '2017-09'
start, end = str(sys.argv[1]), str(sys.argv[2])

for mon in pd.date_range(start, end, freq='M'):
    period = get_first_last_dayofmonth(mon.year, mon.month)
    breakpoint()
    period_fmt = f"{period[0]}/to/{period[1]}"

    print((pd.date_range(start, end, freq='M')))
    # period_fmt = '20200701/to/20200720'
    print(period)
    print(period_fmt)
    # "area": cg.africa_zone,
    #
    breakpoint()
    r = c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'variable': ['10m_u_component_of_wind', '10m_v_component_of_wind', '2m_dewpoint_temperature',
                         '2m_temperature', 'total_cloud_cover'],
            'product_type': 'reanalysis',
            'dates': period_fmt,
            "area": cg.africa_zone,
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
    fn = f'ERA5_{period_fmt[0:6]}_inst1.nc'
    print(outdir, fn)
    r.download(f'{outdir}{fn}')

    r = c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'variable': ['snow_density', 'snow_depth'],
            'product_type': 'reanalysis',
            'dates': period_fmt,
            "area": cg.africa_zone,
            'time': [
                '00:00',
                '06:00',
                '12:00',
            ],
            'grid': '0.25/0.25',
            'format': 'netcdf'
        })
    fn = f'ERA5_{period_fmt[0:6]}_inst2.nc'
    print(outdir, fn)
    r.download(f'{outdir}{fn}')

    r = c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'variable': ['maximum_2m_temperature_since_previous_post_processing',
                         'minimum_2m_temperature_since_previous_post_processing',
                         'surface_solar_radiation_downwards', 'total_precipitation'],
            'product_type': 'reanalysis',
            'dates': period_fmt,
            "area": cg.africa_zone,
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
    fn = f'ERA5_{period_fmt[0:6]}_accmnmx.nc'
    print(outdir, fn)
    r.download(f'{outdir}{fn}')

    r = c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'variable': ['land_sea_mask', ],
            'product_type': 'reanalysis',
            'dates': period_fmt,
            "area": cg.africa_zone,
            'time': [
                '00:00'
            ],
            'grid': '0.25/0.25',
            'format': 'netcdf'
        })
    fn = f'ERA5_{period_fmt[0:6]}_landsea.nc'
    print(outdir, fn)
    r.download(f'{outdir}{fn}')

    r = c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'variable': ['orography', ],
            'product_type': 'reanalysis',
            'dates': period_fmt,
            "area": cg.africa_zone,
            'time': [
                '00:00'
            ],
            'grid': '0.25/0.25',
            'format': 'netcdf'
        })
    fn = f'ERA5_{period_fmt[0:6]}_orography.nc'
    print(outdir, fn)
    r.download(f'{outdir}{fn}')
    exit()
