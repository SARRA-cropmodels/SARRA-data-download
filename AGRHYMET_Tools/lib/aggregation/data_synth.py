import xarray as xr
import numpy as np
import logging
import os
import subprocess

from AGRHYMET_Tools.lib.aggregation.daily_aggregation import Aggregator
from AGRHYMET_Tools.lib.aggregation.parameter_calculation import ParameterCalculator
import AGRHYMET_Tools.lib.tools


class NetCDFData():

    def __init__(self, cfg_parameter, cfg_timeslots, cfg_dirs, start_day, end_day, config_logger):
        self.day = start_day
        self.cfg_timeslots = cfg_timeslots
        self.cfg_parameter = cfg_parameter
        self.cfg_dirs = cfg_dirs
        self.config_logger = config_logger

        # Filepaths of all needed files
        self.source_files = self.__get_source_files()

        # Contains raw data from files
        self.data_files = None

        # Contains raw data from files
        self.end_day = end_day

        # xr.DataSet containing all source parameters
        self.data_in = None

        # xr.DataArray containing the derived parameter
        self.data_der = None

        # xr.DataArray containing the daily aggregated result parameter
        self.data_aggr = None

    def read_files(self):
        """
        Read all needed parameters from source netCDF files. Hold data as list of xr.Datasets in self.data_files
        """
        logging.info('  Read netCDF files...')

        filter_param_name = self.cfg_parameter['ParamSource']['NcParamName']
        all_datasets = {}
        for source_file in self.source_files:
            asd = self.__check_create_nc_file(source_file)
            tmp = xr.open_dataset(asd, engine='netcdf4')
            all_datasets[source_file] = tmp[filter_param_name]
        self.data_files = all_datasets

    def read_files_2(self):
        """
        Read all needed parameters from source netCDF files. Hold data as list of xr.Datasets in self.data_files
        """
        logging.info('  Read netCDF files 2...')

        filter_param_name = self.cfg_parameter['ParamSource']['NcParamName']
        logging.info(filter_param_name)
        all_datasets = {}
        for source_file in self.source_files:
            asd = self.__check_create_nc_file(source_file)
            tmp = xr.open_dataset(asd, engine='netcdf4')
            filter_param_name = tmp.data_vars.__str__().split(' ')[5]
            all_datasets[source_file] = tmp[filter_param_name]
        self.data_files = all_datasets

    def read_files_3(self):
        """
        Read all needed parameters from source netCDF files. Hold data as list of xr.Datasets in self.data_files
        """
        logging.info('  Read netCDF files...')

        filter_param_name = self.cfg_parameter['ParamSource']['NcParamName']
        all_datasets = {}
        for source_file in self.source_files:
            asd = self.__check_create_nc_file(source_file)
            tmp = xr.open_dataset(asd, engine='netcdf4')
            all_datasets[source_file] = tmp[filter_param_name]
        self.data_files = all_datasets

    def combine_to_dataset(self):
        """
        Combine imported data into one xr.Dataset according to predefined timeslots
        Also:
            - Set FillValues
            - rename dimensions
            - Fix lat/lon grid of ERA5 data
        """
        logging.info('  Combine data to xr.Dataset...')

        merged = []

        for slot_id, slot_info in self.cfg_timeslots.items():
            print('icciiii')
            print(slot_info['file_path'])
            print(slot_info['valid_date'])
            print(slot_id)
            file = slot_info['file_path']
            this = self.data_files[file].sel(time=slot_info['valid_date'])
            this['ts'] = slot_id
            merged.append(this)

        merged = xr.concat(merged, dim='ts')
        if 'longitude' in merged.variables.keys():
            merged = merged.rename({'longitude': 'lon', 'latitude': 'lat'})

        # merged.name = self.cfg_parameter['GrbParamName']
        merged.attrs['_FillValue'] = -9999
        merged.encoding['_FillValue'] = -9999

        # Replace lat/lon values, floating point error introduced somewhere at ECMWF, inaccurate already in
        # downloaded gribs import pdb; pdb.set_trace() if self.cfg_parameter['Model'] == 'ERA5': merged.lon.data =
        # np.arange(0, 360, 0.28125) merged.lat.data = np.arange(90, -90.001, -0.28125)

        self.data_in = merged

    def combine_to_dataset_2(self):
        """
        Combine imported data into one xr.Dataset according to predefined timeslots
        Also:
            - Set FillValues
            - rename dimensions
            - Fix lat/lon grid of ERA5 data
        """
        logging.info('  Combine data to xr.Dataset 2...')

        merged = []
        for slot_id, slot_info in self.cfg_timeslots.items():
            file = slot_info['file_path']
            this = self.data_files[file]
            merged.append(this)

        merged = xr.merge(merged)

        # merged = xr.concat(merged)
        if 'longitude' in merged.variables.keys():
            merged = merged.rename({'longitude': 'lon', 'latitude': 'lat'})

        # merged.name = self.cfg_parameter['GrbParamName']
        merged.attrs['_FillValue'] = -9999
        merged.encoding['_FillValue'] = -9999
        # # Replace lat/lon values, floating point error introduced somewhere at ECMWF, inaccurate already in
        # downloaded gribs import pdb; pdb.set_trace() if self.cfg_parameter['Model'] == 'ERA5': merged.lon.data =
        # np.arange(0, 360, 0.28125) merged.lat.data = np.arange(90, -90.001, -0.28125)

        self.data_in = merged

    def combine_to_dataset_3(self):
        """
        Combine imported data into one xr.Dataset according to predefined timeslots
        Also:
            - Set FillValues
            - rename dimensions
            - Fix lat/lon grid of ERA5 data
        """
        logging.info('  Combine 3 data to xr.Dataset 2...')

        merged = []
        for slot_id, slot_info in self.cfg_timeslots.items():
            file = slot_info['file_path']
            this = self.data_files[file]
            merged.append(this)

        merged = xr.merge(merged)
        # merged = xr.concat(merged)
        if 'longitude' in merged.variables.keys():
            merged = merged.rename({'longitude': 'lon', 'latitude': 'lat'})

        # merged.name = self.cfg_parameter['GrbParamName']
        merged.attrs['_FillValue'] = -9999
        merged.encoding['_FillValue'] = -9999
        # # Replace lat/lon values, floating point error introduced somewhere at ECMWF, inaccurate already in
        # downloaded gribs import pdb; pdb.set_trace() if self.cfg_parameter['Model'] == 'ERA5': merged.lon.data =
        # np.arange(0, 360, 0.28125) merged.lat.data = np.arange(90, -90.001, -0.28125)

        self.data_in = merged

    def calc_derived_parameter(self):
        """
        Calculate derived parameter, if needed. Otherwise just pass through data.
        """

        if self.cfg_parameter['ParamFunc']:
            myCalc = ParameterCalculator(self.cfg_parameter)
            self.data_der = myCalc.do_it(self.data_in)
        else:
            self.data_der = self.data_in

    def calc_derived_parameter_2(self):
        """
        Calculate derived parameter, if needed. Otherwise just pass through data.
        """

        if self.cfg_parameter['ParamFunc']:
            myCalc = ParameterCalculator(self.cfg_parameter)
            self.data_der = myCalc.do_it(self.data_in)
            print('good_job_start')
            print(self.data_der['ETO'])
        else:
            self.data_der = self.data_in

    def calc_derived_parameter_3(self):
        """
        Calculate derived parameter, if needed. Otherwise just pass through data.
        """

        if self.cfg_parameter['ParamFunc']:
            myCalc = ParameterCalculator(self.cfg_parameter)
            self.data_der = myCalc.do_it(self.data_in)
            print('self.data_der')
            print(self.data_der)
            print('good_job_start')
            print(self.data_der['T2M'])
        else:
            self.data_der = self.data_in

    def aggregate_to_daily_data(self):
        """
        Do temporal aggregation from 1-/3-hourly data to daily data according to predefined scheme
        """
        myAggr = Aggregator(self.cfg_parameter, self.day, self.config_logger)
        self.data_aggr = myAggr.do_it(self.data_der)

    def aggregate_to_dekadly_data(self):
        """
        Do temporal aggregation from 1-/3-hourly data to daily data according to predefined scheme
        """
        myAggr = Aggregator(self.cfg_parameter, self.day, self.config_logger)
        # self.data_aggr = myAggr.do_it_2(self.data_der)
        self.data_der = myAggr.do_it_2(self.data_in)

    def write_result_netcdf(self):
        """
        Write processed data to netCDF file
        """
        logging.info('  Write xr.Dataset to file...')

        parameter = self.cfg_parameter['ParamResult']['NcParamName']
        decimals = self.cfg_parameter['ParamResult']['Decimals']
        model = self.cfg_parameter['Model']

        # Create output grib file from template file
        out_dir = self.cfg_dirs[model]['data_target']
        out_filename = '{}{}_{}_{}_aggregated'.format(out_dir, model, self.day.strftime('%Y%m%d'), parameter)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        if decimals is not None:
            attrs = self.data_aggr.attrs
            self.data_aggr = self.data_aggr.round(decimals)
            self.data_aggr.attrs = attrs

        outfile = out_filename + '.nc'
        self.data_aggr.to_netcdf(outfile, engine='netcdf4', format='NETCDF4',
                                 encoding={parameter: {'zlib': True, '_FillValue': -9999}})

        return outfile

    def write_result_netcdf_2(self):
        """
        Write processed data to netCDF file
        """
        logging.info('  Write xr.Dataset to file...')

        parameter = self.cfg_parameter['ParamResult']['NcParamName']
        decimals = self.cfg_parameter['ParamResult']['Decimals']
        model = self.cfg_parameter['Model']

        # Create output grib file from template file
        out_dir = self.cfg_dirs[model]['data_target']
        out_filename = '{}{}_{}_{}_10daggregated'.format(out_dir, model, self.day.strftime('%Y%m%d'), parameter)
        print('out_filename debut')
        print(out_filename)
        # self.data_der=self.data_in
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        if decimals is not None:
            attrs = self.data_der.attrs
            self.data_der = self.data_der.round(decimals)
            self.data_der.attrs = attrs

        outfile = out_filename + '.nc'
        self.data_der.to_netcdf(outfile, engine='netcdf4', format='NETCDF4',
                                encoding={parameter: {'zlib': True, '_FillValue': -9999}})

        return outfile

    def __get_source_files(self):
        """
        get list of unique source file names
        """
        all_files = AGRHYMET_Tools.lib.tools.deep_dict_get('file_path', self.cfg_timeslots)
        return set(all_files)

    def __get_source_files2(self):
        """
        get list of unique source file names
        """
        all_files = AGRHYMET_Tools.lib.tools.deep_dict_get('file_path', self.cfg_timeslots)
        return set(all_files)

    def __check_create_nc_file(self, nc_source_file):
        """
        Check if requested netCDF file already exists. If not, try to convert from grib file.
        """

        choice1 = '/Volumes/VolumeWork/Marsop5/ERA5_CDSAPI_mir_monthly/'
        choice2 = '/Volumes/Flux_v2/marsop5/'
        choice3 = '/Volumes/FETT_v2/ERA5_monthly/'

        if os.path.isfile(nc_source_file):
            return nc_source_file
        elif os.path.isfile(nc_source_file.replace(choice1, choice2)):
            return nc_source_file.replace(choice1, choice2)
        elif os.path.isfile(nc_source_file.replace(choice1, choice3)):
            return nc_source_file.replace(choice1, choice3)
        else:
            print('nc_source_file')
            print(nc_source_file)
            print( nc_source_file.split('.')[0])
            print('salut')
            print(self.cfg_parameter['GrbVersion'])

            grb_source_file = nc_source_file.split('.')[0] + '.' + self.cfg_parameter['GrbVersion']
            grb_source_file_gz = grb_source_file + '.gz'

            # Gunzip grib file if needed
            if os.path.isfile(grb_source_file_gz):
                command = 'gunzip ' + grb_source_file_gz
                subprocess.call(command, shell=True)
                if (self.cfg_parameter['GrbFileTag'] == 'inst1') and (self.cfg_parameter['Model'] == 'EChres'):
                    command = 'gunzip ' + grb_source_file_gz.replace('inst1', 'inst2')
                    subprocess.call(command, shell=True)

            # Convert grib to netCDF
            if os.path.isfile(grb_source_file):
                if (self.cfg_parameter['GrbFileTag'] == 'inst1') and (self.cfg_parameter['Model'] == 'EChres'):
                    self.__convert_grib_to_netcdf_WORKAROUND(grb_source_file, nc_source_file)
                else:
                    self.__convert_grib_to_netcdf(grb_source_file, nc_source_file)
                return nc_source_file
            else:
                logging.error('Expected source file does not exist! {}'.format(nc_source_file))
                exit(1)

    def __convert_grib_to_netcdf(self, source_file_grb, target_file_nc):
        """
        Convert given grib file to netCDF file
        """
        # command = 'grib_to_netcdf -D NC_INT {} -o {}'
        command = 'grib_to_netcdf {} -o {}'
        command = command.format(source_file_grb, target_file_nc)
        print(command)
        subprocess.call(command, shell=True)

        command = 'rm ' + source_file_grb
        subprocess.call(command, shell=True)

    def __convert_grib_to_netcdf_WORKAROUND(self, source_file_grb, target_file_nc):
        """
        Convert given grib file to netCDF file

        WORKAROUND
        For some strange reason it is for EChres currently not possible to retrieve parameter SD togther with all other instantaneus
        parameters from the MARS archive. Therefore we download them seperately and now merge them together into one file, before converting to netCDF.
        """

        command = 'grib_copy {} {} {}'
        command = command.format(source_file_grb, source_file_grb.replace('inst1', 'inst2'), target_file_nc + '.tmp')
        print(command)
        subprocess.call(command, shell=True)

        command = 'grib_to_netcdf {} -o {}'
        command = command.format(target_file_nc + '.tmp', target_file_nc)
        print(command)
        subprocess.call(command, shell=True)

        command = 'rm {} {} {}'
        command = command.format(target_file_nc + '.tmp', source_file_grb.replace('inst1', 'inst2'), source_file_grb)
        print(command)
        subprocess.call(command, shell=True)
