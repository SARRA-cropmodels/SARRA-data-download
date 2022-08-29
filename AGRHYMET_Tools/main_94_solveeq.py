import pandas as pd
import xarray as xr
import numpy as np
from datetime import datetime
import config
import os


def main():
    """
    Simple script to apply the equations for a given day and a given parameter
    Script is straped down to the basic dependecies
    """

    # days = ['20160428', '20160429', '20160430', '20160501', '20160502']
    # # days = ['20190701', '20190702']

    start           = '19800101'                                           # Start date of processing period (20160401)
    end             = '19991231'                                           # End date of processing period (20180331)
    daterange       = pd.date_range(datetime.strptime(start, '%Y%m%d'),
                                    datetime.strptime(end,   '%Y%m%d'), freq='D')
    parameters      = config.Base['ERA5'].keys()

    outfiles = []
    for day in daterange:

        day_fmt = day.strftime('%Y%m%d')
        print(day_fmt)
        da_day = []
        for parameter in parameters:
            
            cfg_parameter = config.Base['ERA5'][parameter]
            param_name = cfg_parameter['ParamResult']['NcParamName']
            parameter = param_name
            model       = './data/4_data_RegModels/models_{}_Model2SelSeasonals_eq.nc'.format(parameter)
            ERA5        = './data/2_aggregation/2_data_ERA5_dailyAg/ERA5_{}_{}_aggregated.nc'.format(day_fmt, parameter)
            ERA5        = '/Volumes/VolumeWork/Marsop5/2_data_ERA5_dailyAg/ERA5_{}_{}_aggregated.nc'.format(day_fmt, parameter)
            ERA5corr    = './data/6_solveEq/ERA5_{}_{}_corrected.nc'.format(day_fmt, parameter)
            
            if param_name in config.excl_params_biascorrection:
                # import pdb; pdb.set_trace()
                res     = xr.open_dataset(ERA5)[parameter].squeeze()

            else:

                # Read model equation and ERA5 data
                model_eq = xr.open_dataset(model)
                model_eq.lon.data = model_eq.lon.data.astype(np.float32)
                era5     = xr.open_dataset(ERA5)[parameter].squeeze()

                # Solve equation
                res = solve_model_DayRegion(day_fmt, era5, model_eq, cfg_parameter)

                # Write data to file
                res.data = res.data.astype(np.float32)
                res.name = parameter
                # res.to_netcdf(ERA5corr)
            
            res = do_range_checks(res, cfg_parameter)

            da_day.append(res)

        dir_out = './data/7_solveEq_merge'
        fn_out = '{}/ERA5_{}.nc'.format(dir_out, day_fmt)

        xr.merge(da_day).to_netcdf(fn_out, engine='netcdf4', format='NETCDF4', encoding={parameter: {'zlib': True, '_FillValue': -9999}})
        outfiles.append(fn_out.split('/')[-1])


        if day.strftime('%m%d') == '1231':
            year = day.strftime('%Y')

            os.chdir('/Users/js/Documents/code/JRC_Marsop5_ERA5/processing_pipeline_marsop5/data/7_solveEq_merge/')

            cmd = f'tar zcvf ERA5_{year}.tar.gz {" ".join(outfiles)}'
            print(cmd)
            os.system(cmd)

            cmd = f'rm {" ".join(outfiles)}'
            print(cmd)
            os.system(cmd)

            outfiles = []
            os.chdir('/Users/js/Documents/code/JRC_Marsop5_ERA5/processing_pipeline_marsop5/')


def do_range_checks(parameter_value, cfg_parameter):
    """Carries our range checks on the given parameter_value

    Values outside the range are replaced with the lower/upper values that range.
    A range of "*" means unbounded.

    :param parameter_value: a DataArray
    :param lower_range: the lowest value allowed
    :param upper_range: the highest value allowed
    :return:
    """

    lower_range = cfg_parameter['ParamResult']['LowerRange']
    upper_range = cfg_parameter['ParamResult']['UpperRange']

    # import pdb; pdb.set_trace()
    if None in (lower_range, upper_range):
        msg = f"Lower and/or upper range undefined: ({lower_range} - {upper_range})"
        raise RuntimeError(msg)

    if lower_range != "*":
        ix = parameter_value.data < lower_range
        if np.any(ix):
            parameter_value.data[ix] = lower_range
    if upper_range != "*":
        ix = parameter_value.data > upper_range
        if np.any(ix):
            parameter_value.data[ix] = upper_range

    return parameter_value


def solve_model_DayRegion(day, x0_day, model_eq, cfg_parameter):

    features_seas = calc_seasonal_features(day)

    model_eq.coef_T1.data[np.isnan(model_eq.coef_T1.data)] = 0
    model_eq.coef_T2.data[np.isnan(model_eq.coef_T2.data)] = 0
    model_eq.coef_T3.data[np.isnan(model_eq.coef_T3.data)] = 0
    model_eq.coef_T4.data[np.isnan(model_eq.coef_T4.data)] = 0

    res = (model_eq.coef_X * x0_day) + \
          (model_eq.coef_T1 * features_seas['T1']) + (model_eq.coef_T2 * features_seas['T2']) + \
          (model_eq.coef_T3 * features_seas['T3']) + (model_eq.coef_T4 * features_seas['T4']) + model_eq.intercept

    res.attrs['long_name'] = cfg_parameter['ParamResult']['LongName']
    res.attrs['aggregation'] = cfg_parameter['ParamResult']['AggrName']
    res.attrs['units'] = cfg_parameter['ParamResult']['Unit']

    decimals = cfg_parameter['ParamResult']['Decimals']
    if decimals is not None:
        attrs = res.attrs
        res = res.round(decimals)
        res.attrs = attrs

    return res


def calc_T1(doy):
    return 1 * np.sin(2 * np.pi * ((doy - 21) / 365.))


def calc_T2(doy):
    return 1 * np.sin(2 * np.pi * ((doy - 81) / 365.))


def calc_T3(doy):
    return 1 * np.sin(2 * np.pi * ((doy - 111) / 365.))


def calc_T4(doy):
    return 1 * np.sin(2 * np.pi * ((doy - 141) / 365.))


def calc_seasonal_features(day):
    doy = int(pd.to_datetime(day).strftime('%j'))
    T1 = calc_T1(doy)
    T2 = calc_T2(doy)
    T3 = calc_T3(doy)
    T4 = calc_T4(doy)

    features_seas = {'T1': np.round(T1, 2), 'T2': np.round(T2, 2),
                     'T3': np.round(T3, 2), 'T4': np.round(T4, 2)}

    return features_seas


if __name__ == '__main__':
    main()
