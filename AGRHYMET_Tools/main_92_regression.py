import os
import logging
import pandas as pd
import xarray as xr
from datetime import datetime
from multiprocessing.pool import Pool

import config
import AGRHYMET_Tools.lib.tools
import AGRHYMET_Tools.lib.regression.regmodel


def main():
    """
    Main function of main_92_regression.py

        This function will calculate the bias-correction equations from all prepared ERA5/EChres data

    """

    # Configuration
    start           = '20180101'                                           # Start date of processing period (20160401)
    end             = '20191231'                                           # End date of processing period (20180331)
    daterange       = pd.date_range(datetime.strptime(start, '%Y%m%d'),
                                    datetime.strptime(end,   '%Y%m%d'), freq='D')

    parameters = config.Base['EChres'].keys()
    dir_in_EChres = config.dirs['step_regression']['EChres']['data_source']
    dir_in_ERA5   = config.dirs['step_regression']['ERA5']['data_source']
    dir_out       = config.dirs['step_regression']['data_target']

    # Set regression model type
    mymodel = lib.regression.regmodel.Model_2SelSeasonals

    # Number longitudial chunks (360deg / 5deg = 72)
    # Number longitudial chunks (360deg / 15deg = 24)
    n = 24
    lat_sel = None

    # Create seasonal features for complete time period
    features_seas = lib.tools.calc_seasonal_features(daterange)

    # Iterate through all parameters
    for parameter in parameters:
        logging.info(parameter)

        param_name = config.Base['EChres'][parameter]['ParamResult']['NcParamName']
        if param_name in config.excl_params_biascorrection:
            continue

        # Set if multiprocessing should be used
        # pool = Pool(4)
        pool = None

        if pool:
            # Multiprocessing
            list_args = []
            for x in range(n):
                list_args.append([param_name, x, lat_sel, features_seas, mymodel, dir_in_EChres, dir_in_ERA5, dir_out])
            models_fns = pool.map(multiprc_wrapper, list_args)
            pool.close()
        else:
            # One process at the time (for verification and testing purposes)
            models_fns = []
            for x in range(n):
                fns = mymodel(param_name, x, lat_sel, features_seas, dir_in_EChres, dir_in_ERA5, dir_out).do_it()
                models_fns.append(fns)

        print(models_fns)

        # Merge all longitudinal model files into global files
        fns_eval, fns_eq = zip(*models_fns)
        ds_eval = merge_model_files(fns_eval, 'lon')
        ds_eq = merge_model_files(fns_eq, 'lon')

        ds_eval.to_netcdf(fns_eval[0].replace('tmp_', '').replace('_0_', '_'), mode='w')
        ds_eq.to_netcdf(fns_eq[0].replace('tmp_', '').replace('_0_', '_'), mode='w')
        [os.remove(x) for x in fns_eval + fns_eq]


def multiprc_wrapper(myargs):
    param_name = myargs[0]
    x = myargs[1]
    lat_sel = myargs[2]
    features_seas = myargs[3]
    mymodel = myargs[4]
    dir_in_EChres = myargs[5]
    dir_in_ERA5 = myargs[6]
    dir_out = myargs[7]

    fns = mymodel(param_name, x, lat_sel, features_seas, dir_in_EChres, dir_in_ERA5, dir_out).do_it()
    return fns


def merge_model_files(fns, dim):
    ds_all = []
    for fn in fns:
        ds_all.append(xr.open_dataset(fn))
    ds_all = xr.concat(ds_all, dim)
    return ds_all


if __name__ == '__main__':
    start0 = lib.tools.timer_start()
    lib.tools.init_logging()
    main()
    lib.tools.timer_end(start0, tab=0)
