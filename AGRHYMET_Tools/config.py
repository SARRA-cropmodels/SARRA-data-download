# flake8: noqa
import collections
import AGRHYMET_Tools.lib.tools as tools
import AGRHYMET_Tools.lib.aggregation.daily_aggregation as daily_aggregation
import AGRHYMET_Tools.lib.aggregation.parameter_calculation as parameter_calculation

# import AGRHYMET_Tools.lib.aggregation.parameter_calculation
"""
Definition of meridional aggregation zones
"""

area = {
    # top_lat////left_long//bottom_lat/right_long
    'india_zone': "26/67/7/89",
    'india': "26/67/7/89",
    'burkina': "16/-6/9/3",  
    'waf_zone' : "30/-19/0/26",
    'africa_zone': "39/-19/-35/52"
}

zones = collections.OrderedDict()

# ZONES A REMPLIR

# zones['C'] = (330, -19, -35, 52)
# zones['E'] = (330, 180)
# zones['W'] = (180, 330)
# zones['test'] = (-30, 0)
# zones['W'] =  (-30, 0)
zones['I'] = (330, 67, 7, 89)
# zones['B'] = (330, 67, 7, 89)
# zones['W'] = (-6, 3) # longitude (start_long, end_long)

"""
Source and target locations of model data
"""

bdir = '/home/simon/DATA/cra_era5_tools_clean/output/'

dirs = {
    'step_aggregation': {
        'ERA5': {
            'data_source': '0_download/0_data_ERA5_daily/',
            'data_target':'2_aggregation/2_data_ERA5_dailyAg/',
        },
    },
    'step_evapotranspiration': {
        'ERA5': {
            'data_source': '2_aggregation/2_data_ERA5_dailyAg/',
            'data_target': '2_aggregation/2_data_ERA5_dailyAg/',
        },
    },
    'step_dekadly_aggregation': {
        'ERA5': {
            'data_source': bdir + '2_aggregation/2_data_ERA5_dailyAg/',
            'data_target': bdir + '2_aggregation/2_data_ERA5_dekadlyAg/',
        },
    },
    'step_reordering': {
        'ERA5': {
            'data_source': bdir + '2_aggregation/2_data_ERA5_dailyAg/',
            'data_target': bdir + '3_reordering/3_data_ERA5_dailyAg_reordered/'
        },
    },
    'step_regression': {
        'ERA5': {
            'data_source': bdir + '3_reordering/3_data_ERA5_dailyAg_reordered/',
        },
        'data_target': bdir + '4_data_RegModels/',
    },
    'step_verification': {
        'ERA5': {
            'data_source_global': bdir + '2_aggregation/2_data_ERA5_dailyAg/',
            'data_source_period': bdir + '3_reordering/3_data_ERA5_dailyAg_reordered/',
        },
        'regmodels': {
            'data_source': bdir + '4_data_RegModels/',
        },
        'data_target': bdir + '5_plots_Verification/',
    },
}


"""
List of parameters not to be bias-corrected
"""
excl_params_biascorrection = ['RR',
                              'SN',
                              'SH']


"""
Base: Holds all parameter configurations
"""
Base = {

    'ERA5': {
        '2t_davg': {'ParamSource': {
            'NcParamName': ['t2m'],
        },
            'ParamResult': {
                'NcParamName': 'T2M',
                'LongName': '2 metre temperature (00-00LT)',
                'AggrName': 'Mean 00-00LT',
                'Unit': 'degC',
                'Decimals': 1,
                'LowerRange': '*',
                'UpperRange': '*',
            },
            'ParamFunc': parameter_calculation.unit_temp_K_to_degC,
            'Model': 'ERA5',
            'GrbDataType': 'AN',
            'GrbFileTag': 'inst1',
            'Aggregation': {
                'AggrFunc': daily_aggregation.mean_func,
                'Timeslots': ('00p_21',
                              '00_00', '00_03', '00_06', '00_09',
                              '00_12', '00_15', '00_18', '00_21',
                              '00f_00', '00f_03', '00f_06'),
                'E': (0, 7),
                'C': (2, 9),
                'W': (4, 11),
                'I': (0, 7),
            }
        },
        '2d_davg': {'ParamSource': {
            'NcParamName': ['d2m'],
        },
            'ParamResult': {
                'NcParamName': 'TD',
                'LongName': '2 metre dewpoint temperature (00-00LT)',
                'AggrName': 'Mean 00-00LT',
                'Unit': 'degC',
                'Decimals': 1,
                'LowerRange': '*',
                'UpperRange': '*',
            },
            'ParamFunc': parameter_calculation.unit_temp_K_to_degC,
            'Model': 'ERA5',
            'GrbDataType': 'AN',
            'GrbFileTag': 'inst1',
            'Aggregation': {
                'AggrFunc': daily_aggregation.mean_func,
                'Timeslots': ('00p_21',
                              '00_00', '00_03', '00_06', '00_09',
                              '00_12', '00_15', '00_18', '00_21',
                              '00f_00', '00f_03', '00f_06'),
                'E': (0, 7),
                'C': (2, 9),
                'W': (4, 11),
                'I': (0, 7),
            }
        },
        'mx2t_dmax': {'ParamSource': {
            'NcParamName': ['mx2t'],
        },
            'ParamResult': {
                'NcParamName': 'TX',
                'LongName': 'Maximum temperature at 2 metres (06-18LT)',
                'AggrName': 'Max 06-18LT',
                'Unit': 'degC',
                'Decimals': 1,
                'LowerRange': '*',
                'UpperRange': '*',
            },
            'ParamFunc': parameter_calculation.unit_temp_K_to_degC,
            'Model': 'ERA5',
            'GrbDataType': 'FC',
            'GrbFileTag': 'accmnmx',
            'Aggregation': {
                'AggrFunc': daily_aggregation.max_func,
                'Timeslots': ('18p_07', '18p_08', '18p_09', '18p_10', '18p_11', '18p_12',
                              '06_01', '06_02', '06_03', '06_04', '06_05', '06_06', '06_07', '06_08', '06_09', '06_10',
                              '06_11', '06_12',
                              '18_01', '18_02', '18_03', '18_04', '18_05', '18_06'),
                'E': (0, 11),
                'C': (6, 17),
                'W': (12, 23),
                'I': (0, 11),
            }
        },
        'mn2t_dmin': {'ParamSource': {
            'NcParamName': ['mn2t'],
        },
            'ParamResult': {
                'NcParamName': 'TN',
                'LongName': 'Minimum temperature at 2 metres (18-06LT)',
                'AggrName': 'Min 18-06LT',
                'Unit': 'degC',
                'Decimals': 1,
                'LowerRange': '*',
                'UpperRange': '*',
            },
            'ParamFunc': parameter_calculation.unit_temp_K_to_degC,
            'Model': 'ERA5',
            'GrbDataType': 'FC',
            'GrbFileTag': 'accmnmx',
            'Aggregation': {
                'AggrFunc': daily_aggregation.min_func,
                'Timeslots': ('06p_07', '06p_08', '06p_09', '06p_10', '06p_11', '06p_12',
                              '18p_01', '18p_02', '18p_03', '18p_04', '18p_05', '18p_06', '18p_07', '18p_08', '18p_09',
                              '18p_10', '18p_11', '18p_12',
                              '06_01', '06_02', '06_03', '06_04', '06_05', '06_06'),
                'E': (0, 11),
                'C': (6, 17),
                'W': (12, 23),
                'I': (0, 11),
            }
        },

        'tp_dsum': {'ParamSource': {
            'NcParamName': ['tp'],
        },
            'ParamResult': {
                'NcParamName': 'RR',
                'LongName': 'Total precipitation (06-06LT)',
                'AggrName': 'Sum 06-06LT',
                'Unit': 'mm d-1',
                'Decimals': 1,
                'LowerRange': 0,
                'UpperRange': '*',
            },
            'ParamFunc': parameter_calculation.unit_tp_m_to_mm,
            'Model': 'ERA5',
            'GrbDataType': 'FC',
            'GrbFileTag': 'accmnmx',
            'Aggregation': {
                'AggrFunc': daily_aggregation.sum_func,
                'Timeslots': ('18p_07', '18p_08', '18p_09', '18p_10', '18p_11', '18p_12',
                              '06_01', '06_02', '06_03', '06_04', '06_05', '06_06', '06_07', '06_08', '06_09', '06_10',
                              '06_11', '06_12',
                              '18_01', '18_02', '18_03', '18_04', '18_05', '18_06', '18_07', '18_08', '18_09', '18_10',
                              '18_11', '18_12',
                              '06f_01', '06f_02', '06f_03', '06f_04', '06f_05', '06f_06'),
                'E': (0, 23),
                'C': (6, 29),
                'W': (12, 35),
                'I': (0, 23),
            }
        },
        'ssrd_dsum': {'ParamSource': {
            'NcParamName': ['ssrd'],
        },
            'ParamResult': {
                'NcParamName': 'SSRD',
                'LongName': 'Surface solar radiation downwards (00-00LT)',
                'AggrName': 'Sum 00-00LT',
                'Unit': 'MJ m-2d-1',
                'Decimals': 0,
                'LowerRange': 0,
                'UpperRange': '*',
            },
            'ParamFunc': None,
            'Model': 'ERA5',
            'GrbDataType': 'FC',
            'GrbFileTag': 'accmnmx',
            'Aggregation': {
                'AggrFunc': daily_aggregation.sum_func_ssrd,
                'Timeslots': (
                    '18p_01', '18p_02', '18p_03', '18p_04', '18p_05', '18p_06', '18p_07', '18p_08', '18p_09', '18p_10',
                    '18p_11', '18p_12',
                    '06_01', '06_02', '06_03', '06_04', '06_05', '06_06', '06_07', '06_08', '06_09', '06_10', '06_11',
                    '06_12',
                    '18_01', '18_02', '18_03', '18_04', '18_05', '18_06', '18_07', '18_08', '18_09', '18_10', '18_11',
                    '18_12'),
                'E': (0, 23),
                'C': (6, 29),
                'W': (12, 35),
                'I': (0, 23),
            }
        },
        'ff_davg': {'ParamSource': {
            'NcParamName': ['u10', 'v10'],
        },
            'ParamResult': {
                'NcParamName': 'FFM',
                'LongName': '10 metre wind component (00-00LT)',
                'AggrName': 'Mean 00-00LT',
                'Unit': 'm s-1',
                'Decimals': 1,
                'LowerRange': 0,
                'UpperRange': '*',
            },
            'ParamFunc': parameter_calculation.calc_10u10v_to_ff,
            'Model': 'ERA5',
            'GrbDataType': 'AN',
            'GrbFileTag': 'inst1',
            'Aggregation': {
                'AggrFunc': daily_aggregation.mean_func,
                'Timeslots': ('00p_21',
                              '00_00', '00_03', '00_06', '00_09',
                              '00_12', '00_15', '00_18', '00_21',
                              '00f_00', '00f_03', '00f_06'),
                'E': (0, 7),
                'C': (2, 9),
                'W': (4, 11),
                'I': (0, 7),
            }
        },
    },
}


Climat = {
    'ERA5': {
        'eto_davg': {'ParamSource': {
            'NcParamName': ['SSRD', 'T2M', 'TN', 'TX', 'TD', 'FFM'],
        },
            'ParamResult': {
                'NcParamName': 'ETO',
                'LongName': 'Estimate evapotranspiration (ETo) grass reference surface',
                'AggrName': 'Sum 00-00LT',
                'Unit': 'mm d-1',
                'Decimals': 1,
                'LowerRange': 0,
                'UpperRange': '*',
            },
            'ParamFunc': parameter_calculation.fao56_penman_monteith,
            'Model': 'ERA5',
            'GrbDataType': 'AN',
            'GrbFileTag': ['SSRD_aggregated', 'T2M_aggregated', 'TN_aggregated', 'TX_aggregated', 'TD_aggregated',
                           'FFM_aggregated'],
            'Aggregation': {
                'AggrFunc': daily_aggregation.mean_func,
                'Timeslots': ('00_00'),
                'E': (0, 7),
                'C': (2, 9),
                'W': (4, 11),
                'I': (0, 7),
            }
        },
    },
}
