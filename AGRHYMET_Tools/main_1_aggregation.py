import logging
from datetime import timedelta, datetime
import pandas as pd

import AGRHYMET_Tools.config as config
import AGRHYMET_Tools.lib.tools
from AGRHYMET_Tools.lib.aggregation.data import NetCDFData
from AGRHYMET_Tools.lib.aggregation.daily_aggregation import ConfigLogger
import sys
from pathlib 

print('Number of arguments: {}'.format(len(sys.argv)))
print('Argument(s) passed: {}'.format(str(sys.argv)))

def main():
    """
    Main function of main_1_aggregation.py

        This function implements the daily aggregation of 1/3-hourly ERA5/EChres data

        Steps:
            - Parse configuration for aggregation
            - Collect needed files
            - If needed: Unzip and/or convert to netCDF files
            - Read netCDF files
            - Calculate derived parameters
            - Aggregate the data to daily values following the aggregation scheme
            - Write resulting files as netCDF to disc

    """

    # Configuration
    model           = 'ERA5'                                               # EChres / ERA5
    start           = '20171002'                                           # Start date of processing period (20160401)
    end             = '20171031'                                           # End date of processing period (20180331)
    start, end = str(sys.argv[1]), str(sys.argv[2])
    # Get model configuration
    cfg_dirs        = Path(config.bdir) / config.area[area]  / config.dirs['step_aggregation']
    parameters      = config.Base[model].keys()
    daterange       = pd.date_range(datetime.strptime(start, '%Y%m%d'),
                                    datetime.strptime(end,   '%Y%m%d'), freq='D')

    # Define config logger object
    config_logger = ConfigLogger(model, cfg_dirs)

    logging.info('#########################################################')
    logging.info('Start daily aggregation of model data')
    logging.info('Model:       {}'.format(model))
    logging.info('Period:      {} - {}'.format(start, end))
    logging.info('Parameters:  {}'.format(', '.join(parameters)))
    logging.info('#########################################################')

    # Iterate through days and parameters // netCDF
    for day in daterange:
        for parameter in parameters:
            logging.info('Process day / parameter: {} / {}'.format(day, parameter))

            # Ignore elements not beeing bias corrected
            param_name = config.Base[model][parameter]['ParamResult']['NcParamName']
            # if param_name not in config.excl_params_biascorrection:
            #     continue

            # Collect needed source files and timesteps
            cfg_parameter = config.Base[model][parameter]
            parse_func = parse_config_WebAPI_MIR
            if model is 'ERA5':
                # parse_func = parse_config_WebAPI_MIR_ERA5
                parse_func = parse_config_CDSAPI_MIR_2
            cfg_timeslots = parse_func(cfg_parameter, day, cfg_dirs)

            # Read from netCDF files and organize data as xr.Dataset
            print('mes data sont:')
            print(cfg_parameter)
            print(cfg_timeslots)
            print(cfg_dirs)
            print(day)
            print(config_logger)
            
            data = NetCDFData(cfg_parameter, cfg_timeslots, cfg_dirs, day, config_logger)
            data.read_files()
            data.combine_to_dataset()

            # If needed, calculate derived parameter from source parameters
            data.calc_derived_parameter()

            # Do daily aggregation
            data.aggregate_to_daily_data()
            # import pdb; pdb.set_trace()
            # Write final file to disk
            data.write_result_netcdf()

            # Cleanup
            del data

    config_logger.print_dict()
    config_logger.export_to_file()


def main_external(day, model, in_file):
    # Get model configuration
    cfg_dirs        = config.dirs['step_aggregation']

    parameter      = '2t_davg'
    daterange       = pd.date_range(datetime.strptime(day, '%Y%m%d'),
                                    datetime.strptime(day,   '%Y%m%d'), freq='D')

    # Iterate through days and parameters // netCDF
    day = daterange[0]
    logging.info('Process day / parameter: {} / {}'.format(day, parameter))

    # Collect needed source files and timesteps
    cfg_parameter = config.Base[model][parameter]
    cfg_timeslots = parse_config_CDS(cfg_parameter, day, cfg_dirs, in_file)

    # Read from netCDF files and organize data as xr.Dataset
    data = NetCDFData(cfg_parameter, cfg_timeslots, cfg_dirs, day, None)
    data.read_files()
    data.combine_to_dataset()
    # If needed, calculate derived parameter from source parameters
    data.calc_derived_parameter()
    # Do daily aggregation
    data.aggregate_to_daily_data()
    # Write final file to disk
    outfile = data.write_result_netcdf()

    return outfile


def parse_config(config_param, day, cfg_dirs):
    """
    Parse configuration and create dict containing all needed source parameters, timesteps, files, etc
    """
    logging.info('  Parse timeslot config...')

    timeslots = config_param['Aggregation']['Timeslots']
    file_tag = config_param['GrbFileTag']
    data_type = config_param['GrbDataType']
    model = config_param['Model']

    cfg_timeslots = {}
    for i, ts in enumerate(timeslots):
        issue_date, forecasthour, valid_date = parse_timestep_tag(ts, day, data_type)

        path_dir = '{}{}/'.format(cfg_dirs[model]['data_source'], issue_date.strftime('%Y/%m'))
        if data_type == 'FC':
            path_file = '{}_{}_{}_{}_{}.nc'.format(model, issue_date.strftime('%Y%m%d'), data_type, issue_date.strftime('%H%M'), file_tag)
        elif data_type == 'AN':
            path_file = '{}_{}_{}_{}.nc'.format(model, issue_date.strftime('%Y%m%d'), data_type, file_tag)

        logging.info('    {} / {} \t -> {} +{}h \t -> {}'.format(day.strftime('%Y-%m-%d'), ts, issue_date, forecasthour, valid_date))
        cfg_timeslots[i] = {'timeslot': ts,
                            'issue_date': issue_date,
                            'forecasthour': forecasthour,
                            'valid_date': valid_date,
                            'file_path': path_dir + path_file,
                            'data_type': data_type,
                            }

    return cfg_timeslots


def parse_config_WebAPI_MIR_ERA5(config_param, day, cfg_dirs):
    """
    Parse configuration and create dict containing all needed source parameters, timesteps, files, etc
    """
    logging.info('  Parse timeslot config...')

    timeslots = config_param['Aggregation']['Timeslots']
    file_tag = config_param['GrbFileTag']
    data_type = config_param['GrbDataType']
    model = config_param['Model']

    cfg_timeslots = {}
    for i, ts in enumerate(timeslots):
        issue_date, forecasthour, valid_date = parse_timestep_tag(ts, day, data_type)
        path_dir = '{}'.format(cfg_dirs[model]['data_source'])
        if data_type == 'FC':
            path_file = '{}_{}_{}_{}_{}_mir_01gridNN.nc'.format(model, issue_date.strftime('%Y%m%d'), data_type, issue_date.strftime('%H%M'), file_tag)
        elif data_type == 'AN':
            path_file = '{}_{}_AN_{}_mir_01gridNN.nc'.format(model, issue_date.strftime('%Y%m%d'), file_tag)

        logging.info('    {} / {} \t -> {} +{}h \t -> {} \t -> {}'.format(day.strftime('%Y-%m-%d'), ts, issue_date, forecasthour, valid_date, path_file))
        cfg_timeslots[i] = {'timeslot': ts,
                            'issue_date': issue_date,
                            'forecasthour': forecasthour,
                            'valid_date': valid_date,
                            'file_path': path_dir + path_file,
                            'data_type': data_type,
                            }

    return cfg_timeslots


def parse_config_CDS(config_param, day, cfg_dirs, in_file):
    """
    Parse configuration and create dict containing all needed source parameters, timesteps, files, etc
    """
    logging.info('  Parse timeslot config...')
    import os

    timeslots = config_param['Aggregation']['Timeslots']
    file_tag = config_param['GrbFileTag']
    data_type = config_param['GrbDataType']
    model = config_param['Model']

    cfg_timeslots = {}
    for i, ts in enumerate(timeslots):
        issue_date, forecasthour, valid_date = parse_timestep_tag(ts, day, data_type)
        path_dir = '{}'.format(cfg_dirs[model]['data_source'])
        if data_type == 'FC':
            path_file = '{}_{}_{}_{}_{}_mir_01gridNN.nc'.format(model, issue_date.strftime('%Y%m%d'), data_type, issue_date.strftime('%H%M'), file_tag)
        elif data_type == 'AN':
            path_file = os.path.basename(in_file)

        logging.info('    {} / {} \t -> {} +{}h \t -> {} \t -> {}'.format(day.strftime('%Y-%m-%d'), ts, issue_date, forecasthour, valid_date, path_file))
        cfg_timeslots[i] = {'timeslot': ts,
                            'issue_date': issue_date,
                            'forecasthour': forecasthour,
                            'valid_date': valid_date,
                            'file_path': path_dir + path_file,
                            'data_type': data_type,
                            }

    return cfg_timeslots


def parse_config_WebAPI_MIR(config_param, day, cfg_dirs):
    """
    Parse configuration and create dict containing all needed source parameters, timesteps, files, etc
    """
    logging.info('  Parse timeslot config...')

    timeslots = config_param['Aggregation']['Timeslots']
    file_tag = config_param['GrbFileTag']
    data_type = config_param['GrbDataType']
    model = config_param['Model']

    cfg_timeslots = {}
    for i, ts in enumerate(timeslots):
        issue_date, forecasthour, valid_date = parse_timestep_tag(ts, day, data_type)
        path_dir = '{}'.format(cfg_dirs[model]['data_source'])
        path_file = '{}_{}_{}_mir.netcdf'.format(model, issue_date.strftime('%Y%m'), file_tag)

        logging.info('    {} / {} \t -> {} +{}h \t -> {} \t -> {}'.format(day.strftime('%Y-%m-%d'), ts, issue_date, forecasthour, valid_date, path_file))
        cfg_timeslots[i] = {'timeslot': ts,
                            'issue_date': issue_date,
                            'forecasthour': forecasthour,
                            'valid_date': valid_date,
                            'file_path': path_dir + path_file,
                            'data_type': data_type,
                            }

    return cfg_timeslots


def parse_config_CDSAPI_MIR(config_param, day, cfg_dirs):
    """
    Parse configuration and create dict containing all needed source parameters, timesteps, files, etc
    """
    logging.info('  Parse timeslot config...')

    timeslots = config_param['Aggregation']['Timeslots']
    file_tag = config_param['GrbFileTag']
    data_type = config_param['GrbDataType']
    model = config_param['Model']

    cfg_timeslots = {}
    for i, ts in enumerate(timeslots):
        issue_date, forecasthour, valid_date = parse_timestep_tag(ts, day, data_type)
        path_dir = '{}'.format(cfg_dirs[model]['data_source'])
        path_file = '{}_{}_all.nc'.format(model, valid_date.strftime('%Y%m'))

        # import pdb; pdb.set_trace()

        logging.info('    {} / {} \t -> {} +{}h \t -> {} \t -> {}'.format(day.strftime('%Y-%m-%d'), ts, issue_date, forecasthour, valid_date, path_file))
        cfg_timeslots[i] = {'timeslot': ts,
                            'issue_date': issue_date,
                            'forecasthour': forecasthour,
                            'valid_date': valid_date,
                            'file_path': path_dir + path_file,
                            'data_type': data_type,
                            }
    # exit()
    return cfg_timeslots

def parse_config_CDSAPI_MIR_2(config_param, day, cfg_dirs):
    """
    Parse configuration and create dict containing all needed source parameters, timesteps, files, etc
    """
    logging.info('  Parse timeslot config...')

    timeslots = config_param['Aggregation']['Timeslots']
    file_tag = config_param['GrbFileTag']
    data_type = config_param['GrbDataType']
    model = config_param['Model']

    cfg_timeslots = {}
    for i, ts in enumerate(timeslots):
        issue_date, forecasthour, valid_date = parse_timestep_tag(ts, day, data_type)
        path_dir = '{}'.format(cfg_dirs[model]['data_source'])
        path_file = '{}_{}_{}.nc'.format(model, valid_date.strftime('%Y%m'), file_tag)

        # import pdb; pdb.set_trace()

        logging.info('    {} / {} \t -> {} +{}h \t -> {} \t -> {}'.format(day.strftime('%Y-%m-%d'), ts, issue_date, forecasthour, valid_date, path_file))
        cfg_timeslots[i] = {'timeslot': ts,
                            'issue_date': issue_date,
                            'forecasthour': forecasthour,
                            'valid_date': valid_date,
                            'file_path': path_dir + path_file,
                            'data_type': data_type,
                            }
    # exit()
    return cfg_timeslots

def parse_timestep_tag(ts, day, data_type):
    """
    Derive issueDate, validDate and forecasthour from timestep tag
    Eg:
        (timestep tag, day, data_type)  ->  (issueDate, forecasthour, validDate)
        ------------------------------      ----------------------------------------------
        (06p_07, 2017-06-02, FC) 	    ->  (2017-06-01 06:00:00, +7h, 2017-06-01 13:00:00)
        (00p_15, 2017-06-02, AN) 	    ->  (2017-06-01 15:00:00, +0h, 2017-06-01 15:00:00)
    """
    issue_tag, forecasthour = ts.split('_')
    forecasthour = int(forecasthour)

    issue_hour = int(issue_tag[0:2])
    issue_offset = 0
    if 'p' in issue_tag:
        issue_offset = -24
    elif 'f' in issue_tag:
        issue_offset = +24

    if data_type == 'FC':
        issue_date = day + timedelta(hours=issue_offset) + timedelta(hours=issue_hour)
        valid_date = issue_date + timedelta(hours=forecasthour)
    elif data_type == 'AN':
        issue_date = day + timedelta(hours=issue_offset) + timedelta(hours=forecasthour)
        valid_date = issue_date
        forecasthour = 0

    return issue_date, forecasthour, valid_date


if __name__ == '__main__':
    start0 = AGRHYMET_Tools.lib.tools.timer_start()
    AGRHYMET_Tools.lib.tools.init_logging()

    main()

    AGRHYMET_Tools.lib.tools.timer_end(start0, tab=0)
