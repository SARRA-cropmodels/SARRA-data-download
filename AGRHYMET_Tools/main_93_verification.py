import logging
import pandas as pd
import xarray as xr
import numpy as np
from datetime import datetime

import config
import AGRHYMET_Tools.lib.tools
import AGRHYMET_Tools.lib.verification.plotting


"""
List of interesting locations for further analysis.
FORMAT: (lon, lat): 'LocationName'
"""
pois = {(-16.75, 28.0):  'Tenerife_CoastSea',
        (-16.75, 28.25):  'Tenerife_CoastLand',
        (-3.75, 36.5):   'Spain_CoastSea',
        (-3.75, 36.75):   'Spain_CoastLand',
        (13.5, 52.5):   'Berlin',
        (5.5, 52.0):    'Wageningen',
        (6.5, 44.25):    'Alps_central',
        (7.5, 46.25):    'Alps_lower',
        (13.5, 59.0):   'Sweden_Lake',
        (22.5, 67.0):   'Finland_North',
        (28.0, 20.0):   'Sudan',
        (16.0, -4.0):   'Congo_West',
        (29.75, -5.75):   'Congo_East',
        (26.0, 26.0):   'Egypt',
        (-40.0, 75.0):  'Greenland',
        (60.0, -89.0):  'Antartica',
        (60.0, 89.0):   'Arctic',
        (-94.0, 42.0):  'USIowa',
        (-103.0, 50.0): 'Canada',
        (-60.0, -36.0): 'Argentina',
        (35.0, 46.0):   'Ukraine',
        (78.0, 29.0):   'India',
        (115.0, 35.0):  'China',
        (105.0, -4.0):  'Sumatra',
        (145.0, -37.0): 'Australia',
        (22.5, 50.0):   'OnAggrZone',
        (-75.75, 3.75):   'Columbia_mountains',
        (146.0, -6.75):  'PapuaNeuguinea_mountains',
        }

"""
Define days and periods for a global analysis
"""
days_global     = ['20190715']
periods_global  = [
                #    ['20170701', '20170731', 'July2017'],
                #    ['20180101', '20180131', 'January2018'],
                   ['20190701', '20190731', 'July2019'],
                   ['20190101', '20190131', 'January2019']
                  ]


def main():
    """
    Create wide analysis and verification of bias-corrected parameters
     - Draw model metrics and equation components as global maps
     - Solve model, calc metrics and draw plots, for:
        - one point in time / globally
        - Complete period   / one location
        - Calculate region average of eval metrics (eg land, EU, high elev, tropics, ...))
    """

    # Select model type
    #       ('ModelOneReg4Seasonal', 'ModelOneReg2Seasonal', 'ModelOneRegSimple', 'LassoFixedAlpha',
    #        'ModelStepwiseSimple0', 'ModelFRegression', 'Model2SelSeasonals')
    model = 'Model2SelSeasonals'

    parameters    = config.Base['EChres'].keys()
    dir_cfg       = config.dirs['step_verification']
    dir_out       = dir_cfg['data_target']
    # parameters = ['2t_davg']
    for parameter in parameters:

        logging.info(parameter)

        # Get parameter name
        param_name = config.Base['EChres'][parameter]['ParamResult']['NcParamName']
        parameters_ignore = ['RR', 'SN', 'SH']
        if param_name in parameters_ignore:
            continue

        # Set result directory
        out_cfg = {'dir_out': dir_out + '/' + param_name,
                   'param_name': param_name,
                   'parameter': parameter,
                   'model': model}
        lib.tools.make_dirs_avalable([out_cfg['dir_out']])

        # Read model equation and evaluation files
        model_eq, model_eval = read_models(param_name, model, dir_cfg)

        # Draw global and Europe maps of model equations and model metrics
        create_maps_model(model_eq, model_eval, out_cfg)

        # # # Calculate average metrics over predefined regions, return as table
        # # create_metricstable_model(model_eval, out_cfg)

        # Solve equation for given day/region, calc metrics, draw map
        for day in days_global:
            logging.info('... solve equation for day {}'.format(day))
            x0_day, y_day = read_data_DayRegion(day, param_name, dir_cfg)
            x0_day, y_day = cutout_model_data_region(model_eq, x0_day, y_day)
            yhat_day = solve_model_DayRegion(day, x0_day, model_eq)

            resid, resid_before, correction = calc_metrics(x0_day, y_day, yhat_day)
            create_maps_DayRegion(day, x0_day, y_day, yhat_day, resid, resid_before, correction, out_cfg)

        # Solve equation for given period/region, calc metrics, draw map
        for period in periods_global:
            start, end, descr = period
            logging.info('... solve equation for period {}-{}'.format(start, end))
            days = pd.date_range(datetime.strptime(start, '%Y%m%d'),
                                 datetime.strptime(end,   '%Y%m%d'), freq='D')
            days = [x.strftime('%Y%m%d') for x in days]
            x0_days, y_days = read_data_DaysRegion(days, param_name, dir_cfg)
            x0_days, y_days = cutout_model_data_region(model_eq, x0_days, y_days)
            yhat_days = solve_model_DaysRegion(days, x0_days, model_eq)

            resid, resid_before, correction = calc_metrics(x0_days, y_days, yhat_days)
            resid = resid.mean(dim='time')
            resid_before = resid_before.mean(dim='time')
            correction = correction.mean(dim='time')
            x0_days = x0_days.mean(dim='time')
            y_days = y_days.mean(dim='time')
            yhat_days = yhat_days.mean(dim='time')
            create_maps_DayRegion(descr, x0_days, y_days, yhat_days, resid, resid_before, correction, out_cfg)

        # Solve equation for complete period/gridpoint, calc metrics, draw scatter/ts/table
        for gridpoint, gp_name in pois.items():
            logging.info('... solve equation and draw plot for gridpoint: {} / {}'.format(gp_name, gridpoint))
            x0_gp, y_gp = read_data_PeriodGP(gridpoint, param_name, dir_cfg)
            model_eq_gp, model_eval_gp, x0_gp, y_gp = cutout_model_data_gridpoint(gridpoint, model_eq, model_eval, x0_gp, y_gp)
            yhat_gp, eq_txt = solve_model_PeriodGP(gridpoint, x0_gp, model_eq_gp)

            resid, resid_before, correction = calc_metrics(x0_gp, y_gp, yhat_gp)
            lib.verification.plotting.plot_scatter_timeseries(gridpoint, gp_name, x0_gp, y_gp, yhat_gp, resid, resid_before, correction, out_cfg, eq_txt, model_eval_gp)

            # export_data(gridpoint, gp_name, x0_gp, y_gp, yhat_gp, model_eq_gp, resid, resid_before, correction, out_cfg)


def create_metricstable_model(model_eval, out_cfg):

    parameter = out_cfg['parameter']
    regions = {'Europe':     {'lat': [slice(72, 36)],
                              'lon': [slice(350, 359.9), slice(0, 40)]},
               'N-America':  {'lat': [slice(70, 10)],
                              'lon': [slice(200, 300)]},
               'S-America':  {'lat': [slice(10, -55)],
                              'lon': [slice(275, 325)]},
               'Africa':     {'lat': [slice(36, -35)],
                              'lon': [slice(340, 359.9), slice(0, 45)]},
               'Asia':       {'lat': [slice(75, 8)],
                              'lon': [slice(40, 170)]},
               'Australia':  {'lat': [slice(8, -50)],
                              'lon': [slice(90, 160)]}}

    # Read landsea-mask and elevation data
    model_static = xr.open_dataset('./data/EChres_201701_landsea2_mir_01gridNN.netcdf')
    model_static.lat.values = [round(x, 1) for x in model_static.lat.values.tolist()]
    model_static.lon.values = [round(x, 1) for x in model_static.lon.values.tolist()]

    mae_gt800m = model_eval.mae.where(model_static.z > 800 * 9.81)
    mae_le800m = model_eval.mae.where((model_static.z <= 800 * 9.81) & (model_static.lsm >= 0.5))
    mae_land  = model_eval.mae.where(model_static.lsm >= 0.5)
    mae_coast  = model_eval.mae.where((model_static.lsm >= 0.1) & (model_static.lsm <= 0.9))
    mae_before_gt800m = model_eval.mae_before.where(model_static.z > 800 * 9.81)
    mae_before_le800m = model_eval.mae_before.where((model_static.z <= 800 * 9.81) & (model_static.lsm >= 0.5))
    mae_before_land  = model_eval.mae_before.where(model_static.lsm >= 0.5)
    mae_before_coast  = model_eval.mae_before.where((model_static.lsm >= 0.1) & (model_static.lsm <= 0.9))

    metrics = {}

    def myfmt(x):
        return '{:.4f}'.format(float(x))

    def mymean(x):
        return x.mean().data

    def mymean_prctImpr(x, x_before):
        return 1 - (x.mean().data / x_before.mean().data)

    for region in regions:

        this_mae_gt800m = sel_subregion(mae_gt800m, regions[region])
        this_mae_le800m = sel_subregion(mae_le800m, regions[region])
        this_mae_land = sel_subregion(mae_land, regions[region])
        this_mae_coast = sel_subregion(mae_coast, regions[region])
        this_mae_before_gt800m = sel_subregion(mae_before_gt800m, regions[region])
        this_mae_before_le800m = sel_subregion(mae_before_le800m, regions[region])
        this_mae_before_land = sel_subregion(mae_before_land, regions[region])
        this_mae_before_coast = sel_subregion(mae_before_coast, regions[region])

        # this_mae_land.to_netcdf('./data/test_mae_land_{}.nc'.format(region))
        # this_mae_gt800m.to_netcdf('./data/test_mae_gt800m_{}.nc'.format(region))
        # this_mae_le800m.to_netcdf('./data/test_mae_le800m_{}.nc'.format(region))
        # this_mae_coast.to_netcdf('./data/test_mae_coast_{}.nc'.format(region))

        metrics[region] = {'Parameter': parameter,
                           'MAE land':           myfmt(mymean(this_mae_land)),
                           'MAE before land ':   myfmt(mymean(this_mae_before_land)),
                           'MAE Impr land ':     myfmt(mymean_prctImpr(this_mae_land, this_mae_before_land)),
                           'MAE >800m':          myfmt(mymean(this_mae_gt800m)),
                           'MAE before >800m':   myfmt(mymean(this_mae_before_gt800m)),
                           'MAE Impr >800m':     myfmt(mymean_prctImpr(this_mae_gt800m, this_mae_before_gt800m)),
                           'MAE <=800m':         myfmt(mymean(this_mae_le800m)),
                           'MAE before <=800m':  myfmt(mymean(this_mae_before_le800m)),
                           'MAE Impr <=800m':    myfmt(mymean_prctImpr(this_mae_le800m, this_mae_before_le800m)),
                           'MAE coast':          myfmt(mymean(this_mae_coast)),
                           'MAE before coast':   myfmt(mymean(this_mae_before_coast)),
                           'MAE Impr coast':     myfmt(mymean_prctImpr(this_mae_coast, this_mae_before_coast))}
    metrics = pd.DataFrame.from_dict(metrics, orient='index')

    fn = '{}/data_{}_{}_metrics.csv'.format(out_cfg['dir_out'], out_cfg['param_name'], out_cfg['model'])
    metrics.to_csv(fn)


def sel_subregion(da, region):
    lat = region['lat'][0]
    lons = region['lon']

    subs = []
    for lon in lons:
        subs.append(da.sel(lat=lat, lon=lon))
    subs = xr.concat(subs, dim='lon')
    return subs


def read_models(param_name, model, dir_cfg):
    logging.info('... read_models()')

    # Read files
    dir_models    = dir_cfg['regmodels']['data_source']
    model_eq = xr.open_dataset('{}/models_{}_{}_eq.nc'.format(dir_models, param_name, model))
    model_eval = xr.open_dataset('{}/models_{}_{}_eval.nc'.format(dir_models, param_name, model))

    # # Correct floating points of coordindates
    # model_eq.lat.values = [round(x, 1) for x in model_eq.lat.values.tolist()]
    # model_eq.lon.values = [round(x, 1) for x in model_eq.lon.values.tolist()]
    # model_eval.lat.values = [round(x, 1) for x in model_eval.lat.values.tolist()]
    # model_eval.lon.values = [round(x, 1) for x in model_eval.lon.values.tolist()]

    return model_eq, model_eval


def read_data_DayRegion(day, param_name, dir_cfg):
    # Read org ERA5 data
    dir_in = dir_cfg['ERA5']['data_source_global']
    era5 = xr.open_dataset('{}/ERA5_{}_{}_aggregated.nc'.format(dir_in, day, param_name))[param_name].squeeze()
    # era5.lat.values = [round(x, 1) for x in era5.lat.values.tolist()]
    # era5.lon.values = [round(x, 1) for x in era5.lon.values.tolist()]

    # Read org EChres data
    dir_in = dir_cfg['EChres']['data_source_global']
    echres = xr.open_dataset('{}/EChres_{}_{}_aggregated.nc'.format(dir_in, day, param_name))[param_name].squeeze()
    # echres.lat.values = [round(x, 1) for x in echres.lat.values.tolist()]
    # echres.lon.values = [round(x, 1) for x in echres.lon.values.tolist()]

    return era5, echres


def read_data_DaysRegion(days, param_name, dir_cfg):
    era5, echres = [], []
    for day in days:
        era5_tmp, echres_tmp = read_data_DayRegion(day, param_name, dir_cfg)
        era5.append(era5_tmp)
        echres.append(echres_tmp)

    era5 = xr.concat(era5, dim='time')
    echres = xr.concat(echres, dim='time')

    return era5, echres


def read_data_PeriodGP(gridpoint, param_name, dir_cfg):

    offset = 0
    if gridpoint[0] < 0:
        offset = 360
    lon_slice = int((gridpoint[0] + offset) // 15)

    # Read org ERA5 data
    # dir_in     = './data/4_data_ERA5_dailyAg_01grid_reordered'
    dir_in = dir_cfg['ERA5']['data_source_period']
    file_in    = '{}/ERA5_{}_aggregated_15degLons_{}.nc'.format(dir_in, param_name, lon_slice)
    era5_lon = xr.open_dataset(file_in)[param_name]
    # era5_lon.lat.values = [round(x, 1) for x in era5_lon.lat.values.tolist()]
    # era5_lon.lon.values = [round(x, 1) for x in era5_lon.lon.values.tolist()]

    # Read org EChres data
    # dir_in     = './data/4_data_EChres_dailyAg_reordered'
    dir_in = dir_cfg['EChres']['data_source_period']
    file_in    = '{}/EChres_{}_aggregated_15degLons_{}.nc'.format(dir_in, param_name, lon_slice)
    echres_lon = xr.open_dataset(file_in)[param_name]
    # echres_lon.lat.values = [round(x, 1) for x in echres_lon.lat.values.tolist()]
    # echres_lon.lon.values = [round(x, 1) for x in echres_lon.lon.values.tolist()]

    return era5_lon, echres_lon


def cutout_model_data_region(model_eq, x0_day, y_day):
    # Cutout data
    def get_grid_extend(data):
        eq_ext = [data.lat.values.tolist()[0], data.lat.values.tolist()[-1], data.lon.values.tolist()[0], data.lon.values.tolist()[-1]]
        eq_ext = [round(x, 1) for x in eq_ext]
        return eq_ext
    eq_ext, era5_ext, echres_ext = get_grid_extend(model_eq), get_grid_extend(x0_day), get_grid_extend(y_day)

    if echres_ext != era5_ext:
        print('ERA5 and EChres should have the same grid extend!')
        exit(1)

    if eq_ext != era5_ext:
        print('Models grid and data grid are different, will cutout the extend!')

    x0_day = x0_day.sel(lat=slice(eq_ext[0], eq_ext[1]), lon=slice(eq_ext[2], eq_ext[3]))
    y_day = y_day.sel(lat=slice(eq_ext[0], eq_ext[1]), lon=slice(eq_ext[2], eq_ext[3]))

    return x0_day, y_day


def cutout_model_data_gridpoint(gridpoint, model_eq, model_eval, x0_gp, y_gp):

    offset = 0
    if gridpoint[0] < 0:
        offset = 360

    gp_lat, gp_lon = gridpoint[1], gridpoint[0] + offset

    # import pdb; pdb.set_trace()
    ts_era5_gp = x0_gp.sel(lat=gp_lat, lon=gp_lon)
    ts_echres_gp = y_gp.sel(lat=gp_lat, lon=gp_lon)
    model_eq_gp = model_eq.sel(lat=gp_lat, lon=gp_lon)
    model_eval_gp = model_eval.sel(lat=gp_lat, lon=gp_lon)

    return model_eq_gp, model_eval_gp, ts_era5_gp, ts_echres_gp


def solve_model_DayRegion(day, x0_day, model_eq):

    features_seas = lib.tools.calc_seasonal_features(day)

    model_eq.coef_T1.data[np.isnan(model_eq.coef_T1.data)] = 0
    model_eq.coef_T2.data[np.isnan(model_eq.coef_T2.data)] = 0
    model_eq.coef_T3.data[np.isnan(model_eq.coef_T3.data)] = 0
    model_eq.coef_T4.data[np.isnan(model_eq.coef_T4.data)] = 0

    res = (model_eq.coef_X * x0_day) + \
          (model_eq.coef_T1 * features_seas['T1']) + (model_eq.coef_T2 * features_seas['T2']) + \
          (model_eq.coef_T3 * features_seas['T3']) + (model_eq.coef_T4 * features_seas['T4']) + model_eq.intercept
    return res


def solve_model_DaysRegion(days, x0_day, model_eq):
    res = []
    for day in days:
        res_tmp = solve_model_DayRegion(day, x0_day.sel(time=day), model_eq)
        res.append(res_tmp)
    res = xr.concat(res, dim='time')
    return res


def solve_model_PeriodGP(gridpoint, x0_gp, model_eq_gp):

    features_seas = lib.tools.calc_seasonal_features(x0_gp.time.data)

    if np.isnan(model_eq_gp.coef_T1.data).all():
        model_eq_gp.coef_T1.data = 0
    if np.isnan(model_eq_gp.coef_T2.data).all():
        model_eq_gp.coef_T2.data = 0
    if np.isnan(model_eq_gp.coef_T3.data).all():
        model_eq_gp.coef_T3.data = 0
    if np.isnan(model_eq_gp.coef_T4.data).all():
        model_eq_gp.coef_T4.data = 0

    yhat_gp = (model_eq_gp.coef_X * x0_gp) + \
              (model_eq_gp.coef_T1.data * features_seas['T1']) + (model_eq_gp.coef_T2.data * features_seas['T2']) + \
              (model_eq_gp.coef_T3.data * features_seas['T3']) + (model_eq_gp.coef_T4.data * features_seas['T4']) + model_eq_gp.intercept
    eq_txt = get_eq_textrepr(model_eq_gp)

    return yhat_gp, eq_txt


def get_eq_textrepr(model_eq_gp):
    def fmt_coef(coef):
        if coef.data == 0:
            return '0'
        else:
            return '{:.2f}'.format(float(coef.data))

    eq_txt = []
    eq_txt.append('y^ = {}*X + {}*T1 + {}*T2'.format(fmt_coef(model_eq_gp.coef_X), fmt_coef(model_eq_gp.coef_T1), fmt_coef(model_eq_gp.coef_T2)))
    eq_txt.append('+ {}*T3 + {}*T4 + {}'.format(fmt_coef(model_eq_gp.coef_T3), fmt_coef(model_eq_gp.coef_T4), fmt_coef(model_eq_gp.intercept)))

    return eq_txt


def calc_metrics(x0, y, yhat):
    resid = y - yhat
    resid_before = y - x0
    correction = yhat - x0
    return resid, resid_before, correction


def create_maps_model(model_eq, model_eval, out_cfg):
    logging.info('... create_maps_model()')

    pn = out_cfg['param_name']
    plot = lib.verification.plotting.plot_map_global
    plot(model_eq.intercept, vrange=get_vrange(pn, 'intercept'), cmap='seismic', out_cfg=out_cfg)
    plot(model_eq.coef_X, vrange=get_vrange(pn, 'coef_X'), cmap='seismic', out_cfg=out_cfg)
    plot(model_eq.coef_T1, vrange=get_vrange(pn, 'coef_T'), cmap='seismic', out_cfg=out_cfg, pois=pois)
    plot(model_eq.coef_T2, vrange=get_vrange(pn, 'coef_T'), cmap='seismic', out_cfg=out_cfg)
    plot(model_eq.coef_T3, vrange=get_vrange(pn, 'coef_T'), cmap='seismic', out_cfg=out_cfg)
    plot(model_eq.coef_T4, vrange=get_vrange(pn, 'coef_T'), cmap='seismic', out_cfg=out_cfg)

    plot(model_eval.mae, vrange=get_vrange(pn, 'mae'), cmap='jet', out_cfg=out_cfg)
    plot(model_eval.mae_before, vrange=get_vrange(pn, 'mae'), cmap='jet', out_cfg=out_cfg)
    plot(model_eval.rmse, vrange=get_vrange(pn, 'rmse'), cmap='jet', out_cfg=out_cfg)
    plot(model_eval.rmse_before, vrange=get_vrange(pn, 'rmse'), cmap='jet', out_cfg=out_cfg)
    plot(model_eval.rsq, vrange=get_vrange(pn, 'rsq'), cmap_label='[0-1]', cmap='jet', title='R_square', out_cfg=out_cfg)
    plot(model_eval.maxabs, vrange=get_vrange(pn, 'maxabs'), out_cfg=out_cfg)
    plot(model_eval.bic, vrange=get_vrange(pn, 'bic'), out_cfg=out_cfg)
    plot(model_eval.nnonzero, vrange=get_vrange(pn, 'nnonzero'), cmap='jet_r', out_cfg=out_cfg)
    plot(model_eval.bias_before, vrange=get_vrange(pn, 'corval'), cmap='seismic', out_cfg=out_cfg)

    # plot = lib.verification.plotting.plot_map_europe
    # plot(model_eq.intercept, vrange=get_vrange(pn, 'intercept'), cmap='seismic', out_cfg=out_cfg)
    # plot(model_eq.coef_X, vrange=get_vrange(pn, 'coef_X'), cmap='seismic', out_cfg=out_cfg)
    # plot(model_eq.coef_T1, vrange=get_vrange(pn, 'coef_T'), cmap='seismic', out_cfg=out_cfg)
    # plot(model_eq.coef_T2, vrange=get_vrange(pn, 'coef_T'), cmap='seismic', out_cfg=out_cfg)
    # plot(model_eq.coef_T3, vrange=get_vrange(pn, 'coef_T'), cmap='seismic', out_cfg=out_cfg)
    # plot(model_eq.coef_T4, vrange=get_vrange(pn, 'coef_T'), cmap='seismic', out_cfg=out_cfg)
    #
    # plot(model_eval.mae, vrange=get_vrange(pn, 'mae'), cmap='jet', out_cfg=out_cfg)
    # plot(model_eval.rmse, vrange=get_vrange(pn, 'rmse'), cmap='jet', out_cfg=out_cfg)
    # plot(model_eval.rsq, vrange=get_vrange(pn, 'rsq'), cmap_label='[0-1]', cmap='jet', title='R_square', out_cfg=out_cfg)
    # plot(model_eval.bic)
    # plot(model_eval.nnonzero)


def create_maps_DayRegion(day, x0_day, y_day, yhat_day, resid, resid_before, correction, out_cfg):

    pn = out_cfg['param_name']
    plot = lib.verification.plotting.plot_map_global
    plot(x0_day, vrange=get_vrange(pn, 'absval'), title='{}: x0 / ERA5'.format(day), fntag='{}_ERA5'.format(day), out_cfg=out_cfg)
    plot(y_day, vrange=get_vrange(pn, 'absval'), title='{}: y / EChres'.format(day), fntag='{}_EChres'.format(day), out_cfg=out_cfg)
    plot(yhat_day, vrange=get_vrange(pn, 'absval'), title='{}: yhat / ERA5 corrected'.format(day), fntag='{}_ERA5corrected'.format(day), out_cfg=out_cfg)

    plot(resid, vrange=get_vrange(pn, 'corval'), title='{}: Residual (y - yhat)'.format(day), fntag='{}_Residual'.format(day), cmap='bwr', out_cfg=out_cfg)
    plot(resid_before, vrange=get_vrange(pn, 'corval'), title='{}: Residual before (y - x)'.format(day), fntag='{}_ResidualBefore'.format(day), cmap='bwr', out_cfg=out_cfg)
    plot(correction, vrange=get_vrange(pn, 'corval'), title='{}: Correction effect (yhat - x)'.format(day), fntag='{}_CorrectionEffect'.format(day), cmap='bwr', out_cfg=out_cfg)

    # plot = lib.verification.plotting.plot_map_europe
    # plot(x0_day, vmin=vmin, vmax=vmax, title='{}: x0 / ERA5'.format(day), fntag='{}_ERA5'.format(day), out_cfg=out_cfg)
    # plot(y_day, vmin=vmin, vmax=vmax, title='{}: y / EChres'.format(day), fntag='{}_EChres'.format(day), out_cfg=out_cfg)
    # plot(yhat_day, vmin=vmin, vmax=vmax, title='{}: yhat / ERA5 corrected'.format(day), fntag='{}_ERA5corrected'.format(day), out_cfg=out_cfg)
    #
    # plot(resid, vmin=-10, vmax=10, title='{}: Residual (y - yhat)'.format(day), fntag='{}_Residual'.format(day), cmap='bwr', out_cfg=out_cfg)
    # plot(resid_before, vmin=-10, vmax=10, title='{}: Residual before (y - x)'.format(day), fntag='{}_ResidualBefore'.format(day), cmap='bwr', out_cfg=out_cfg)
    # plot(correction, vmin=-10, vmax=10, title='{}: Correction effect (yhat - x)'.format(day), fntag='{}_CorrectionEffect'.format(day), cmap='bwr', out_cfg=out_cfg)


def export_data(gridpoint, gp_name, x0_gp, y_gp, yhat_gp, model_eq_gp, resid, resid_before, correction, out_cfg):

    features_seas = lib.tools.calc_seasonal_features(x0_gp.time.data)

    labels = ['X_ERA5', 'T1', 'T2', 'T3', 'T4', 'Y_ERA5corr', 'EChres']
    df = [x0_gp.data, features_seas['T1'], features_seas['T2'], features_seas['T3'], features_seas['T4'], yhat_gp.data, y_gp.data]
    df = pd.DataFrame(data=df).T
    df.index = x0_gp.time.data
    df.columns = labels

    df['eq_coef_X'] = model_eq_gp.coef_X.data
    df['eq_coef_T1'] = model_eq_gp.coef_T1.data
    df['eq_coef_T2'] = model_eq_gp.coef_T2.data
    df['eq_coef_T3'] = model_eq_gp.coef_T3.data
    df['eq_coef_T4'] = model_eq_gp.coef_T4.data
    df['eq_intercept'] = model_eq_gp.intercept.data

    fn = '{}/data_{}_{}_gridpoint_{}.csv'.format(out_cfg['dir_out'], out_cfg['param_name'], out_cfg['model'], gp_name)
    df.to_csv(fn)


def get_vrange(param_name, metric):

    vcfg = None
    if param_name in ['T2M', 'TD', 'TX', 'TN']:
        vcfg = {'intercept': (-150, 150), 'coef_X': (0.3, 1.7), 'coef_T': (-6, 6),
                'mae': (0.2, 4), 'rmse': (0.2, 4), 'rsq': (0.2, 1), 'maxabs': (0, 20), 'nnonzero': (500, 730), 'bic': (-1000, 4000),
                'absval': (-20, 30), 'corval': (-10, 10)}
    elif 'TCC' in param_name:
        vcfg = {'intercept': (-70, 70), 'coef_X': (0.2, 1.8), 'coef_T': (-20, 20),
                'mae': (0, 20), 'rmse': (0, 20), 'rsq': (0.2, 1), 'maxabs': (0, 100), 'nnonzero': (500, 730), 'bic': (2000, 3500),
                'absval': (0, 10), 'corval': (-50, 50)}
    # elif 'Relative_Humidity' in param_name:
    #     vcfg = {'intercept': (-100, 100), 'coef_X': (0.2, 1.8), 'coef_T': (-15, 15),
    #             'mae': (0, 12), 'rmse': (0, 12), 'rsq': (0.2, 1), 'maxabs': (0, 60), 'nnonzero': (500, 730), 'bic': (2000, 6000),
    #             'absval': (0, 100), 'corval': (-30, 30)}
    elif 'SSRD' in param_name:
        vcfg = {'intercept': (-1.5e7, 1.5e7), 'coef_X': (0.2, 1.8), 'coef_T': (-5e6, 5e6),
                'mae': (0, 5e6), 'rmse': (0, 5e6), 'rsq': (0.2, 1), 'maxabs': (0, 20), 'nnonzero': (500, 730), 'bic': (20000, 24000),
                'absval': (0, 3.5e7), 'corval': (-1.5e7, 1.5e7)}
    # elif 'Vapour_Pressure' in param_name:
    #     vcfg = {'intercept': (-20, 20), 'coef_X': (0.2, 1.8), 'coef_T': (-6, 6),
    #             'mae': (0, 2), 'rmse': (0, 2), 'rsq': (0.2, 1), 'maxabs': (0, 20), 'nnonzero': (500, 730), 'bic': (-1000, 4000),
    #             'absval': (0, 40), 'corval': (-10, 10)}
    elif 'FFM' in param_name:
        vcfg = {'intercept': (-5, 5), 'coef_X': (0.2, 1.8), 'coef_T': (-2, 2),
                'mae': (0, 2), 'rmse': (0, 2), 'rsq': (0.2, 1), 'maxabs': (0, 10), 'nnonzero': (500, 730), 'bic': (-1000, 4000),
                'absval': (0, 25), 'corval': (-5, 5)}
    else:
        vcfg = {'intercept': (-50, 50), 'coef_X': (0.2, 1.8), 'coef_T': (-6, 6),
                'mae': (0.2, 4), 'rmse': (0.2, 4), 'rsq': (0.2, 1), 'maxabs': (0, 20), 'nnonzero': (500, 730), 'bic': (-1000, 4000),
                'absval': (0, 25), 'corval': (-5, 5)}

    return (vcfg[metric][0], vcfg[metric][1])


if __name__ == '__main__':
    start0 = lib.tools.timer_start()
    lib.tools.init_logging()

    main()

    lib.tools.timer_end(start0, tab=0)
