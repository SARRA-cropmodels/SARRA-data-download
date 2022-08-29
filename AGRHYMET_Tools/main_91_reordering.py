import os
import logging
from datetime import datetime
import pandas as pd
import xarray as xr
from multiprocessing.pool import Pool
# from multiprocessing.pool import ThreadPool

import config
import AGRHYMET_Tools.lib.tools


def main():
    """
    Main function of main_91_reordering.py

        This function will reorder the ERA5/EChres netCDF files:
            Before: each file contains global data of one day
            After:  each file contains 5deg of longitudes of the whole 2 years period
    """

    # Configuration
    model           = 'ERA5'                                               # EChres / ERA5

    # 2year_models
    start           = '20180101'                                           # Start date of processing period (20160401)
    end             = '20191231'                                           # End date of processing period (20180331)

    # # 1year_outsample
    # start           = '20161101'                                           # Start date of processing period (20160401)
    # end             = '20171030'                                           # End date of processing period (20180331)

    # # 3year_models
    # start           = '20161101'                                           # Start date of processing period (20160401)
    # end             = '20191030'                                           # End date of processing period (20180331)

    # Get model configuration
    parameters      = config.Base[model].keys()
    daterange       = pd.date_range(datetime.strptime(start, '%Y%m%d'),
                                    datetime.strptime(end,   '%Y%m%d'), freq='D')

    dir_fileserver  = config.dirs['step_reordering'][model]
    dir_in = dir_fileserver['data_source']
    dir_out = dir_fileserver['data_target']

    # Iterate through all parameters
    for parameter in parameters:
        logging.info(parameter)

        param_name = config.Base[model][parameter]['ParamResult']['NcParamName']
        if param_name in config.excl_params_biascorrection:
            continue

        queue_exec_tasks(daterange, param_name, model, parameter, dir_in, dir_out)
        # break


def queue_exec_tasks(daterange, param_name, model, parameter, dir, dir_out):
    """
    Iterate through all 5deg longitudinal bands and collect reordering tasks
    """

    pool = Pool(2)
    # pool = ThreadPool(2)

    lats = slice(70, 30)
    # lats = None

    # step = 1440
    step = 60
    slices = [(x, x + step) for x in range(0, 1440, step)]
    # print(len(slices))
    # print(slices)
    # exit()

    queue_tasks = []
    for lon1, lon2 in slices:
        # Add tasks to queue
        queue_tasks.append([daterange, param_name, model, parameter, dir, dir_out, lon1, lon2, step, lats])

    # Start processing tasks in queue
    pool.map(task_reordering, queue_tasks)
    pool.close()


def task_reordering(args):
    """
    Reorder task
    """
    daterange = args[0]
    param_name = args[1]
    model = args[2]
    # parameter = args[3]
    dir = args[4]
    dir_out = args[5]
    lon1 = args[6]
    lon2 = args[7]
    step = args[8]
    # lats = args[9]

    logging.info('Processing {}, {} , {}'.format(model, lon1, lon2))
    all = []

    fntag = ''
    fn_out = '{}/{}_{}_aggregated_{}degLons_{}.nc'.format(dir_out, model, param_name, int(step / 4), int(lon1 / step))
    if os.path.isfile(fn_out):
        return

    # Read longitudinal bands for all files in period
    for day in daterange:
        fn = '{}/{}_{}_{}_aggregated{}.nc'.format(dir, model, day.strftime('%Y%m%d'), param_name, fntag)
        # print(fn)
        # ds = xr.open_dataset(fn).isel(lon=slice(lon1, lon2)).sel(lat=lats)
        ds = xr.open_dataset(fn).isel(lon=slice(lon1, lon2))
        all.append(ds)

    # Concat all days and export to netCDF
    res = xr.concat(all, 'time')
    res.to_netcdf(fn_out, engine='netcdf4', format='NETCDF4', encoding={param_name: {'zlib': True, '_FillValue': -9999}})


if __name__ == '__main__':
    start0 = lib.tools.timer_start()
    lib.tools.init_logging()

    main()

    lib.tools.timer_end(start0, tab=0)
