

import rasterio
import glob
from pathlib import Path
import calendar
import pandas as pd
import math
from datetime import timedelta,datetime
import numpy as np
from dateutil.relativedelta import relativedelta
import sys

def dekad2day(year, month, dekad):
    """Gets the day of a dekad.
    Parameters
    ----------
    year : int
        Year of the date.
    month : int
        Month of the date.
    dekad : int
        Dekad of the date.
    Returns
    -------
    day : int
        Day value for the dekad.
    """
    if dekad == 1:
        day = 10
    elif dekad == 2:
        day = 20
    elif dekad == 3:
        day = calendar.monthrange(year, month)[1]
    return day

def runningdekad2date(year, rdekad):
    """Gets the date of the running dekad of a spacifc year.
    Parameters
    ----------
    year : int
        Year of the date.
    rdekad : int
        Running dekad of the date.
    Returns
    -------
    datetime.datetime
        Date value for the running dekad.
    """
    month = int(math.ceil(rdekad / 3.))
    dekad = rdekad - month * 3 + 3
    day = dekad2day(year, month, dekad)

    return datetime(year, month, day)

def check_dekad(date):
    """Checks the dekad of a date and returns the dekad date.
    Parameters
    ----------
    date : datetime
        Date to check.
    Returns
    -------
    new_date : datetime
        Date of the dekad.
    """
    if date.day < 11:
        dekad = 10
    elif date.day > 10 and date.day < 21:
        dekad = 20
    else:
        dekad = calendar.monthrange(date.year, date.month)[1]
    new_date = datetime(date.year, date.month, dekad)
    return new_date
    
def day2dekad(day):
    """Returns the dekad of a day.
    Parameters
    ----------
    day : int
        Day of the date.
    Returns
    -------
    dekad : int
        Number of the dekad in a month.
    """
    if day < 11:
        dekad = 1
    elif day > 10 and day < 21:
        dekad = 2
    else:
        dekad = 3
    return dekad


def get_dekad_period(dates):
    """Checks number of the dekad in the current year for dates given as list.
    Parameters
    ----------
    dates : list of datetime
        Dates to check.
    Returns
    -------
    period : list of int
        List of dekad periods.
    """
    period = []
    for dat in dates:
        d = check_dekad(dat)
        dekad = day2dekad(d.day)
        per = dekad + ((d.month - 1) * 3)
        period.append((per,dat.year))
    return period
    
area = str(sys.argv[1])
pathIn = f"output/{area}/tif_data/daily"
output_path = Path(pathIn).parent / "dekad"
output_path.mkdir(exist_ok=True)

for para_ in glob.glob(pathIn +"/*"):

    param = Path(para_).stem # name of the parameter for istance RR SH ETO

    output_path = Path(pathIn).parent / "dekad" / param
    output_path.mkdir(exist_ok=True)

    daterange = [datetime.strptime(str(Path(x).stem).split("_")[-1], '%Y%m%d') for x in glob.glob(f"{para_}/*.tif")]

    dekad = get_dekad_period(daterange) # convert date range to list of (dekad, year)
    dekad = set(dekad)

    for dek_ in dekad: 
        start_day = runningdekad2date(dek_[1],dek_[0])
        end_day = start_day + timedelta(days=9)
        daterange = pd.date_range(start_day, end_day, freq='D')

        array = []
        for date_ in daterange:
            x_name = Path(pathIn) / param / f"ERA5_{param}_{date_.strftime('%Y%m%d')}.tif"
            if x_name.exists()==False:
                print(f"{Path(x_name).stem} dont exist")
                break

            with rasterio.open(x_name) as ds:
                ts_array = ds.read()
                profile = ds.profile
                array.append(ts_array[0])

        if x_name.exists():
            array = np.stack(array,axis=-1)
            dek_array = np.mean(array,2)
 
            new_name = Path(pathIn).parent / "dekad" / param / f"ERA5_{param}_year{dek_[1]}_dekad{dek_[0]}.tif"

            with rasterio.open(new_name, 'w', **profile) as dst:
                dst.write(dek_array, 1)

