import numpy as np
import xarray as xr
import pandas as pd
from pprint import pprint
import copy
import logging
import AGRHYMET_Tools.config as config

from datetime import datetime


class Aggregator():
    def __init__(self, cfg_parameter, day, config_logger):

        self.day = day
        self.cfg_parameter = cfg_parameter
        self.aggr_function = self.cfg_parameter['Aggregation']['AggrFunc']
        self.logger = config_logger

    def do_it(self, indata):
        """
        Do daily aggragation according to predefined scheme
        Also:
            - Log configuration with ConfigLogger
            - Fix time information in resulting xr.DataArray
        """
        logging.info('  Aggregate from 3-/1-hourly to daily data...')

        aggr_function = self.aggr_function
        param = self.cfg_parameter['ParamResult']['NcParamName']
        slots = self.cfg_parameter['Aggregation']['Timeslots']
        cfg_zones = config.zones

        # Rename xr.DataSet parameter if needed
        indata = self.__rename_data_var(indata)

        # Translate indata from xr.DataSet to xr.DataArray and create empty result field
        indata = indata[param]
        res = self.__get_empty_result_da(indata)

        # Iterate through 8 meridional aggregation zones
        for i, (zone, zone_def) in enumerate(cfg_zones.items()):
            logging.info('    Zone: {} {}'.format(zone, zone_def))

            # Get config this zone
            zone_timesteps = self.cfg_parameter['Aggregation'][zone]
            zone_lon = slice(zone_def[0], zone_def[1])

            # Aggregate timesteps for this zone
            tmp_data, docu = aggr_function(indata, zone_lon, zone_timesteps, slots)
            res.loc[dict(lon=zone_lon)] = tmp_data

            # Special case for CENTRAL zone, defined in 2 parts
            if len(zone_def) == 4:
                zone_lon = slice(zone_def[2], zone_def[3])
                tmp_data, foo = aggr_function(indata, zone_lon, zone_timesteps, slots)
                res.loc[dict(lon=zone_lon)] = tmp_data

            # Write to ConfigLogger
            self.logger.add_docu(param, zone, docu, self.day)

        # Fix time information
        res = res.drop('ts')
        res = res.expand_dims('time')
        res.time.data[0] = self.day.to_datetime64()

        print('salut_tous')
        print(self.day)
        print(self.day.to_datetime64())

        res.attrs['long_name'] = self.cfg_parameter['ParamResult']['LongName']
        res.attrs['aggregation'] = self.cfg_parameter['ParamResult']['AggrName']
        res.attrs['units'] = self.cfg_parameter['ParamResult']['Unit']

        return res

    def do_it_2(self, indata):
        """
        Do dekadly aggragation according to predefined scheme
        Also:
            - Log configuration with ConfigLogger
            - Fix time information in resulting xr.DataArray
        """
        logging.info('  Aggregate to dekadalu data...')
        aggr_function = self.aggr_function
        param = self.cfg_parameter['ParamResult']['NcParamName']
        slots = self.cfg_parameter['Aggregation']['Timeslots']
        # slots=indata.time.
        cfg_zones = config.zones
        z=config.zones


        # Rename xr.DataSet parameter if needed
        indata = self.__rename_data_var(indata)
        # Translate indata from xr.DataSet to xr.DataArray and create empty result field
        indata = indata[param]
        # res = copy.deepcopy(indata)
        # res = self.__get_empty_result_da(indata)
        # Iterate through 8 meridional aggregation zones
        res, docu = aggr_function(indata)
        # Fix time information
        # res = res.drop('ts')

        # Fix time information
        # res = res.drop('ts')
        res = res.expand_dims('time')
        res.time.data[0] = self.day.to_datetime64()
        res.attrs['long_name'] = self.cfg_parameter['ParamResult']['LongName']
        res.attrs['aggregation'] = self.cfg_parameter['ParamResult']['AggrName']
        res.attrs['units'] = self.cfg_parameter['ParamResult']['Unit']
        return res

    def __get_empty_result_da(self, indata):
        """
        Create an empty xr.DataArray using the indata as a template
        """
        res = copy.deepcopy(indata.sel(ts=0))
        res.data.fill(np.NaN)
        return res

    def __rename_data_var(self, indata):
        """
        If xr.DataSet parameter still has not yet the final result name, rename it. Thats the case for non-derived parameters.
        """
        param_source = self.cfg_parameter['ParamSource']['NcParamName'][0]
        param_result = self.cfg_parameter['ParamResult']['NcParamName']
        ds_name = list(indata.data_vars.keys())[0]

        if (ds_name == param_source) & (ds_name != param_result):
            indata = indata.rename({param_source: param_result})
        return indata


def mean_func_synth(da):
    """
    Aggregation function for calculating the MEAN of a given sequence of timeslots
    """

    da_sel = da
    # ts_n = len(da_sel.ts.data)
    ts_n = len(da_sel.time.data)
    print('da_sel')
    print(da_sel)
    print('ts_n')
    print(ts_n)

    da_aggr = da_sel.mean(dim='time')
    print('da_aggr')
    print(da_aggr)
    # Write string describing operation
    # slots_selected = [slots[x] for x in da_sel.time.data]
    slots_selected = da_sel.time.data
    print(slots_selected)
    # explanation = ', '.join(slots_selected)
    # explanation = 'mean({})/{}'.format(explanation, ts_n)
    explanation = 'mean'
    print('viens_ici')
    print(da_aggr)
    return da_aggr, explanation


def sum_func_synth(da):
    """
    Aggregation function for calculating the MEAN of a given sequence of timeslots
    """

    da_sel = da
    # ts_n = len(da_sel.ts.data)
    ts_n = len(da_sel.time.data)
    print('da_sel')
    print(da_sel)
    print('ts_n')
    print(ts_n)

    da_aggr = da_sel.sum(dim='time')
    print('da_aggr')
    print(da_aggr)
    # Write string describing operation
    # slots_selected = [slots[x] for x in da_sel.time.data]
    slots_selected = da_sel.time.data
    print(slots_selected)
    # explanation = ', '.join(slots_selected)
    # explanation = 'mean({})/{}'.format(explanation, ts_n)
    explanation = 'mean'
    print('viens_ici')
    print(da_aggr)
    return da_aggr, explanation


def mean_func(da, lon_range, zone_timesteps, slots):
    """
    Aggregation function for calculating the MEAN of a given sequence of timeslots
    """

    ts_start, ts_end = zone_timesteps
    ts_range = slice(ts_start, ts_end)
    # da_sel = da.sel(ts=ts_range, lon=lon_range)
    da_sel = da.sel(ts=ts_range, lon=lon_range)
    ts_n = len(da_sel.ts.data)
    da_aggr = da_sel.mean(dim='ts')

    # Write string describing operation
    slots_selected = [slots[x] for x in da_sel.ts.data]
    explanation = ', '.join(slots_selected)
    explanation = 'sum({})/{}'.format(explanation, ts_n)

    print('suite_songoti')
    print(slots_selected)
    return da_aggr, explanation


def min_func(da, lon_range, zone_timesteps, slots):
    """
    Aggregation function for calculating the MINIMUM of a given sequence of timeslots
    """

    ts_start, ts_end = zone_timesteps
    ts_range = slice(ts_start, ts_end)
    da_sel = da.sel(ts=ts_range, lon=lon_range)
    da_aggr = da_sel.min(dim='ts')

    # Write string describing operation
    slots_selected = [slots[x] for x in da_sel.ts.data]
    explanation = ', '.join(slots_selected)
    explanation = 'min({})'.format(explanation)

    return da_aggr, explanation


def max_func(da, lon_range, zone_timesteps, slots):
    """
    Aggregation function for calculating the MAXIMUM of a given sequence of timeslots
    """

    ts_start, ts_end = zone_timesteps
    ts_range = slice(ts_start, ts_end)
    da_sel = da.sel(ts=ts_range, lon=lon_range)
    da_aggr = da_sel.max(dim='ts')

    # Write string describing operation
    slots_selected = [slots[x] for x in da_sel.ts.data]
    explanation = ', '.join(slots_selected)
    explanation = 'max({})'.format(explanation)

    return da_aggr, explanation


def sum_func(da, lon_range, zone_timesteps, slots):
    """
    Aggregation function for calculating the SUM of a given sequence of timeslots
    """

    ts_start, ts_end = zone_timesteps
    ts_range = slice(ts_start, ts_end)
    da_sel = da.sel(ts=ts_range, lon=lon_range)
    da_aggr = da_sel.sum(dim='ts')

    # Write string describing operation
    slots_selected = [slots[x] for x in da_sel.ts.data]
    explanation = ', '.join(slots_selected)
    explanation = 'sum({})'.format(explanation)

    return da_aggr, explanation


def sum_func_ssrd(da, lon_range, zone_timesteps, slots):
    """
    Aggregation function for calculating the SUM of a given sequence of timeslots
    ssrd en  Joule
    in ERA5 data ssrd is the accumulation or timeintegral over last hour, with units of Wm-2s or Jm-2, so to have the
    average or mean flux over hour you should divide by 3600 seconds

    short ssrd(time, latitude, longitude) ;
	ssrd:scale_factor = 64.8701570201273 ;
	ssrd:add_offset = 2125535.56492149 ;
	ssrd:_FillValue = -32767s ;
	ssrd:missing_value = -32767s ;
	ssrd:units = "J m**-2" ;
	ssrd:long_name = "Surface solar radiation downwards" ;
	ssrd:standard_name = "surface_downwelling_shortwave_flux_in_air" ;

	Resultat en KJ/m**2/day


    """

    ts_start, ts_end = zone_timesteps
    ts_range = slice(ts_start, ts_end)
    da_sel = da.sel(ts=ts_range, lon=lon_range)
    da_aggr = da_sel.sum(dim='ts')
    # da_aggr = (da_aggr / 3600.)
    da_aggr = (da_aggr / 1000.)

    # Write string describing operation
    slots_selected = [slots[x] for x in da_sel.ts.data]
    explanation = ', '.join(slots_selected)
    explanation = 'sum({})'.format(explanation)
    print('ssrd here')
    print(explanation)

    return da_aggr, explanation


def selTS_func(da, lon_range, zone_timesteps, slots):
    """
    Dummy aggregation function for selecting only one timestep per zone
    """

    da_aggr = da.sel(ts=zone_timesteps, lon=lon_range)

    # Write string describing operation
    slots_selected = slots[zone_timesteps]
    explanation = 'sel({})'.format(slots_selected)

    return da_aggr, explanation


def sum_diff_func(da, lon_range, zone_timesteps, slots):
    """
    Aggregation function for calculating the SUM of a given sequence of timeslots.
    Additionally calculates differences between timeslots handed over as tuples.
    This function is used for the accumulated parameter tp and ssrd of EChres.
    """

    tmp = []
    explanation = []
    for slice in zone_timesteps:
        if isinstance(slice, tuple):
            explanation.append('({} - {})'.format(slots[slice[0]], slots[slice[1]]))
            tmp.append(da.sel(ts=slice[0], lon=lon_range) - da.sel(ts=slice[1], lon=lon_range))
        elif isinstance(slice, int):
            explanation.append('{}'.format(slots[slice]))
            tmp.append(da.sel(ts=slice, lon=lon_range).drop(['ts', 'time']))
        else:
            exit(1)

    explanation = ', '.join(explanation)
    explanation = 'sum({})'.format(explanation)
    tmp = xr.concat(tmp, dim='ts')
    asd = tmp.sum(dim='ts')

    return asd, explanation


def sum_diff_func_ssrd_mega_joule(da, lon_range, zone_timesteps, slots):
    """
    ssrd is calculated in MJ ==> j/1000.000
    Aggregation function for calculating the SUM of a given sequence of timeslots.
    Additionally calculates differences between timeslots handed over as tuples.
    This function is used for the accumulated parameter tp and ssrd of EChres.
    """

    tmp = []
    explanation = []
    for slice in zone_timesteps:
        if isinstance(slice, tuple):
            explanation.append('({} - {})'.format(slots[slice[0]], slots[slice[1]]))
            tmp.append(da.sel(ts=slice[0], lon=lon_range) - da.sel(ts=slice[1], lon=lon_range))
        elif isinstance(slice, int):
            explanation.append('{}'.format(slots[slice]))
            tmp.append(da.sel(ts=slice, lon=lon_range).drop(['ts', 'time']))
        else:
            exit(1)

    explanation = ', '.join(explanation)
    explanation = 'sum({})'.format(explanation)
    tmp = xr.concat(tmp, dim='ts')
    asd = tmp.sum(dim='ts')

    print('mes_ssrd_joule')
    print(asd)
    asd = asd * 0.000001
    print('mes_ssrd_Mega_joule')
    print(asd)

    return asd, explanation


def count_rain_func(da, lon_range, zone_timesteps, slots):
    """
    Aggregation function for calculating the rain fraction of the day
    """
    ts_start, ts_end = zone_timesteps
    ts_range = slice(ts_start, ts_end)
    da_sel = da.sel(ts=ts_range, lon=lon_range)

    # For some strange reason, there are some float values at the poles. Therefore create mask indicating faulty
    # locations
    faulty_mask = (np.mod(da_sel.round(1), 1) != 0).any(dim='ts')

    # Round data, due to floating point inaccuracy
    da_sel = da_sel.round(0)

    # Calculate rain fraction
    ts_n = float(len(da_sel.ts.data))
    da_aggr = (da_sel.where(da_sel == 1).count(dim='ts') / ts_n)

    # Replace faulty locations with np.NaN
    da_aggr = da_aggr.where(~faulty_mask, np.nan)

    # Write string describing operation
    slots_selected = [slots[x] for x in da_sel.ts.data]
    explanation = ', '.join(slots_selected)
    explanation = 'count rain({})/{}'.format(explanation, ts_n)

    return da_aggr, explanation


def count_solid_func(da, lon_range, zone_timesteps, slots):
    """
    Aggregation function for calculating the solid prec fraction of the day
    """
    ts_start, ts_end = zone_timesteps
    ts_range = slice(ts_start, ts_end)
    da_sel = da.sel(ts=ts_range, lon=lon_range)

    # For some strange reason, there are some float values at the poles. Therefore create mask indicating faulty locations
    faulty_mask = (np.mod(da_sel.round(1), 1) != 0).any(dim='ts')

    # Round data, due to floating point inaccuracy
    da_sel = da_sel.round(0)

    # Calculate solid prec fraction
    ts_n = float(len(da_sel.ts.data))
    da_aggr = (da_sel.where((da_sel == 3) | (da_sel == 5) | (da_sel == 6) | (da_sel == 7) | (da_sel == 8)).count(
        dim='ts') / ts_n)

    # Replace faulty locations with np.NaN
    da_aggr = da_aggr.where(~faulty_mask, np.nan)

    # Write string describing operation
    slots_selected = [slots[x] for x in da_sel.ts.data]
    explanation = ', '.join(slots_selected)
    explanation = 'count solid({})/{}'.format(explanation, ts_n)

    return da_aggr, explanation


class ConfigLogger():
    def __init__(self, model, cfg_dirs):
        self.doku = {}
        self.day = None
        self.model = model
        self.cfg_dirs = cfg_dirs

    def add_docu(self, param, zone, docu_str, day):

        # Only add docu if we are working on the first day of the processing Period. No need to write down config
        # more than one time
        if not self.day:
            self.day = day
        if self.day != day:
            return

        if param not in self.doku:
            self.doku[param] = {zone: docu_str}
        else:
            self.doku[param][zone] = docu_str

    def print_dict(self):
        pprint(self.day)
        pprint(self.doku)

    def export_to_file(self):
        fn = '{}/config_aggr_docu_{}.csv'.format(self.cfg_dirs[self.model]['data_target'], self.model)
        pd.DataFrame(self.doku).T.to_csv(fn)
