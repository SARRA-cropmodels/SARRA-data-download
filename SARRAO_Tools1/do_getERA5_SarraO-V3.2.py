#!/usr/bin/env python

# Les variables
# '10m_u_component_of_wind':'u10'
# '10m_v_component_of_wind':'v10'
# '2m_dewpoint_temperature':'d2m'
# '2m_temperature': 't2m'
# 'maximum_2m_temperature_since_previous_post_processing':'mx2t'
# 'minimum_2m_temperature_since_previous_post_processing':'mn2t'
# 'surface_solar_radiation_downwards':'ssrd'
# 'total_cloud_cover':'tcc'
# 'total_precipitation':'tp'
# Africa
# "area":"39/-19/-35/52",
africa_zone="39/-19/-35/52"
# WAF
# "area":"30/-19/0/26",
waf_zone="30/-19/0/26"

import cdsapi
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import os
c = cdsapi.Client()


# Africa
# "area":"39/-19/-35/52",
my_zone="39/-19/-35/52"
# WAF
# "area":"30/-19/0/26",
my_zone="30/-19/0/26"


def days_of_month(y, m):
    d0 = datetime(y, m, 1)
    d1 = d0 + relativedelta(months=1)
    out = list()
    while d0 < d1:
        out.append(d0.strftime('%Y%m%d'))
        d0 += timedelta(days=1)
    return out


def download_data_sarrao(nom_varibale, short_name, start_date, end_date, data_path):
    delta = timedelta(days=1)
    while start_date <= end_date:
        d = start_date.strftime("%Y%m%d")
        start_date += delta
        d2 = start_date.strftime("%Y%m%d")
        my_period = "%s/%s" % (d, d2)
        print("date1 :", d)
        print("date2 :", d2)
        print("ma periode: :", my_period)
        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "variable": nom_varibale,
                "area": africa_zone,
                "product_type": "reanalysis",
                "date":d,
                "grid": "0.25/0.25",

                "time": [
                    '00:00', '01:00', '02:00',
                    '03:00', '04:00', '05:00',
                    '06:00', '07:00', '08:00',
                    '09:00', '10:00', '11:00',
                    '12:00', '13:00', '14:00',
                    '15:00', '16:00', '17:00',
                    '18:00', '19:00', '20:00',
                    '21:00', '22:00', '23:00'
                ],
                "format": "netcdf"
            },
            # "%sera5_africa_%s_%s_%s.nc" % (data_path,short_name,d,d2)
            "%sera5_africa_%s_%s.nc" % (data_path, short_name, d)
        )


# Period to download
start_date = datetime(2019, 5, 1)
end_date = datetime(2019, 5, 31)
my_path="D:/2020/SARRA_O/python_script/era5_data/"
#my_path="/SARRAH/ERA5_DATA/"


#Download 2m_temperature
# '2m_temperature': 't2m'
nom_varibale1 = "2m_temperature"
short_name1 = "t2m"
data_path1 = "%s%s/" % (my_path,short_name1)
if not os.path.exists(data_path1):
    os.makedirs(data_path1)
#download_data_sarrao(nom_varibale1, short_name1, start_date, end_date, data_path1)

start_date = datetime(2019,1, 1)
end_date = datetime(2019, 3, 1)
# '2m_dewpoint_temperature':'d2m'
nom_varibale1 = "2m_dewpoint_temperature"
short_name1 = "d2m"
data_path1 = "%s%s/" % (my_path,short_name1)
if not os.path.exists(data_path1):
    os.makedirs(data_path1)
#download_data_sarrao(nom_varibale1, short_name1, start_date, end_date, data_path1)

# 'surface_solar_radiation_downwards':'ssrd'
nom_varibale1 = "surface_solar_radiation_downwards"
short_name1 = "ssrd"
data_path1 = "%s%s/" % (my_path,short_name1)
if not os.path.exists(data_path1):
    os.makedirs(data_path1)
#download_data_sarrao(nom_varibale1, short_name1, start_date, end_date, data_path1)

# '10m_u_component_of_wind':'u10'
nom_varibale1 = "10m_u_component_of_wind"
short_name1 = "u10"
data_path1 = "%s%s/" % (my_path,short_name1)
if not os.path.exists(data_path1):
    os.makedirs(data_path1)
download_data_sarrao(nom_varibale1, short_name1, start_date, end_date, data_path1)

# '10m_v_component_of_wind':'v10'
nom_varibale1 = "10m_v_component_of_wind"
short_name1 = "v10"
data_path1 = "%s%s/" % (my_path,short_name1)
if not os.path.exists(data_path1):
    os.makedirs(data_path1)
download_data_sarrao(nom_varibale1, short_name1, start_date, end_date, data_path1)

# 'total_precipitation':'tp'
nom_varibale1 = "total_precipitation"
short_name1 = "tp"
data_path1 = "%s%s/" % (my_path,short_name1)

if not os.path.exists(data_path1):
    os.makedirs(data_path1)
download_data_sarrao(nom_varibale1, short_name1, start_date, end_date, data_path1)

#python  "d:2020\SARRA_O\python_script\agrhymet_era5_tools\do_getERA5_SarraO-V3.2.py"