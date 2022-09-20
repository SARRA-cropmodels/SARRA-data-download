import logging
import copy
import xarray as xr
import numpy as np
import statsmodels.api as sm
from statsmodels.tools import eval_measures
from scipy.stats.stats import pearsonr
import operator

import lib.tools


def read_files(param_name, n, lat_sel, dir_in_EChres, dir_in_ERA5):
    """
    Helper function for reading ERA5/ECHres files for a given lat chunk
    """
    logging.info('    {} - read_files()'.format(n))
    fn_EChres = '{}/EChres_{}_aggregated_15degLons_{}.nc'.format(dir_in_EChres, param_name, n)
    fn_ERA5   = '{}/ERA5_{}_aggregated_15degLons_{}.nc'.format(dir_in_ERA5, param_name, n)

    if lat_sel:
        da_echres = xr.open_dataset(fn_EChres).sel(lat=lat_sel)[param_name]
        da_era5 = xr.open_dataset(fn_ERA5).sel(lat=lat_sel)[param_name]
    else:
        da_echres = xr.open_dataset(fn_EChres)[param_name]
        da_era5 = xr.open_dataset(fn_ERA5)[param_name]

    da_echres.lat.values = np.round(da_echres.lat.values, 2)
    da_era5.lat.values = np.round(da_era5.lat.values, 2)

    da_echres = da_echres
    da_era5 = da_era5

    return da_echres, da_era5


class Model():
    """
    Abstract parant class representing the regression model.

    The following methods need to be implemented by the child class:
        - calc_model()      implements the regression algorthm and writes variable 'self.models_raw'

    """

    def __init__(self, param_name, x, lat_sel, features_seas, dir_in_EChres, dir_in_ERA5, dir_out):
        self.x = x
        self.param_name = param_name
        self.lat_sel = lat_sel
        self.dir_in_EChres = dir_in_EChres
        self.dir_in_ERA5 = dir_in_ERA5
        self.dir_out = dir_out
        self.features_seas = features_seas

        self.da_echres, self.da_era5 = None, None
        self.models_raw = None
        self.models_eval = None
        self.models_eq = None

        self.verif_data = None

    def calc_model(self):
        """
        calc_model() needs to be implemented by child class.

        This methods implements the regression algorithm and writes variable self.models_raw
        self.models_raw is an xr.DataArray containing named dict at each grid point.
        The dict needs to have entries for all defined eq and eval parameters.
        """

        pass

    def do_it(self):
        """
        This function will trigger the regression calculation pipeline
        """
        start = lib.tools.timer_start()
        logging.info('    {} - calculate regression()'.format(self.x))

        # Read ERA5/EChres data
        self.read_data()

        # Train regression model
        self.calc_model()

        # Convert the model equations and metrics to xarray objects and export to netCDF files
        self.convert_model()
        fns = self.write_model_to_file()

        lib.tools.timer_end(start, tab=2)

        return fns

    def read_data(self):
        """
        Read ERA5/EChres data from files
        """

        self.da_echres, self.da_era5 = read_files(self.param_name, self.x, self.lat_sel, self.dir_in_EChres, self.dir_in_ERA5)

        # Compare data to seasonal features
        len_ts_echres = len(self.da_echres.time.data)
        len_ts_feat = len(self.features_seas['T1'])
        if len_ts_echres != len_ts_feat:
            logging.error('model data and seasonal features should contain same amount of timesteps! Check the processing period!')
            exit(1)

    def convert_model(self):
        """
        Convert the raw model equations and metrics to xarray data objects
        """

        logging.info('    {} - convert_model()'.format(self.x))
        extract_evals = ['mae', 'mae_before', 'rmse', 'rmse_before', 'rsq', 'bic', 'nnonzero', 'maxabs', 'bias', 'bias_before']
        extract_eq = ['intercept', 'coef_X', 'coef_T1', 'coef_T2', 'coef_T3', 'coef_T4']

        def extract_keys(param_name):
            vfunc = np.vectorize(lambda x: x[param_name])
            res_rsq = vfunc(self.models_raw.values)

            da_res = copy.deepcopy(self.models_raw)
            da_res.name = param_name
            da_res.values = res_rsq
            return da_res
        
        all_eval = []
        for param_name in extract_evals:
            all_eval.append(extract_keys(param_name))
        all_eval = xr.merge(all_eval)

        all_eq = []
        for param_name in extract_eq:
            all_eq.append(extract_keys(param_name))
        all_eq = xr.merge(all_eq)

        self.models_eval = all_eval
        self.models_eq = all_eq

    def write_model_to_file(self):
        """
        Write model equations and metrics to file
        """
        logging.info('    {} - write_model_to_file()'.format(self.x))

        fn0 = '{}/tmp_models_{}_{}_{}_eval.nc'.format(self.dir_out, self.param_name, self.model_name, self.x)
        self.models_eval.to_netcdf(fn0)

        fn1 = '{}/tmp_models_{}_{}_{}_eq.nc'.format(self.dir_out, self.param_name, self.model_name, self.x)
        self.models_eq.to_netcdf(fn1)

        return fn0, fn1


class Model_2SelSeasonals(Model):
    """
    This class implements the abstract Model()-class using a simple OLS regression using the 2 best seasonal features
    """

    def __init__(self, param_name, x, lat_sel, features_seas, dir_in_EChres, dir_in_ERA5, dir_out):
        Model.__init__(self, param_name, x, lat_sel, features_seas, dir_in_EChres, dir_in_ERA5, dir_out)

        self.model_name = 'Model2SelSeasonals'
        self.model_desc = 'Simple OLS regression using the main feature, a y-intercept and the 2 best seasonal features (based on corr-coef)'

    def calc_model(self):
        logging.info('    {} - calc_model()'.format(self.x))

        global T1, T2, T3, T4
        T1, T2, T3, T4 = self.features_seas['T1'], self.features_seas['T2'], self.features_seas['T3'], self.features_seas['T4']

        def reg_xarray_2SelSeasonals(x, y):
            """
            Function implementing the model training
                x -> Feature/Predictor
                y -> Target/Predictand
            """

            # Organize features
            map_feat_ind = {'const': 0, 'X': 1, 'T1': 2, 'T2': 3, 'T3': 4, 'T4': 5}
            np_X = sm.add_constant(np.column_stack((x, T1, T2, T3, T4)), has_constant='add')

            def get_sub_np_features(np_X, list_feat):
                ind = [map_feat_ind[x] for x in list_feat]
                return np.take(np_X, ind, axis=1)

            def get_sorted_corrs_np(Y, X, features_names):
                test2 = {y: abs(pearsonr(Y, X[:, x])[0]) for (x, y) in enumerate(features_names)}
                sorted_x = sorted(test2.items(), key=operator.itemgetter(1), reverse=True)
                return [x[0] for x in sorted_x], [x[1] for x in sorted_x]

            # Find best features (based on correlation coeficient)
            list_feat = ['T1', 'T2', 'T3', 'T4']
            best_X, corr_vals = get_sorted_corrs_np(y, get_sub_np_features(np_X, list_feat), list_feat)

            def myreg_np(Y, np_X):
                return sm.OLS(Y, np_X).fit()

            # Build and train model
            list_feat = ['const', 'X'] + best_X[:2]
            mod = myreg_np(y, get_sub_np_features(np_X, list_feat))

            # Reorganize coefs of features
            def get_single_coefs(best_mod, best_feat):
                coefs = []
                for feat in ['const', 'X', 'T1', 'T2', 'T3', 'T4']:
                    if feat in best_feat:
                        coefs.append(best_mod.params[best_feat.index(feat)])
                    else:
                        coefs.append(None)
                return coefs
            coefs = get_single_coefs(mod, list_feat)

            res2 = {'mae': eval_measures.meanabs(mod.fittedvalues, y),
                    'mae_before': eval_measures.meanabs(x, y),
                    'rmse': eval_measures.rmse(mod.fittedvalues, y),
                    'rmse_before': eval_measures.rmse(x, y),
                    'maxabs': eval_measures.maxabs(mod.fittedvalues, y),
                    'bias': eval_measures.bias(mod.fittedvalues, y),
                    'bias_before': eval_measures.bias(x, y),
                    'rsq': mod.rsquared,
                    # 'rsq_adj': None,
                    'bic': mod.bic,
                    'nnonzero': np.count_nonzero(x),
                    # 'nfeat': None,
                    'intercept': coefs[0],
                    'coef_X': coefs[1],
                    'coef_T1': coefs[2],
                    'coef_T2': coefs[3],
                    'coef_T3': coefs[4],
                    'coef_T4': coefs[5]}
            return res2

        self.models_raw = xr.apply_ufunc(reg_xarray_2SelSeasonals,
                                         self.da_era5, self.da_echres,
                                         vectorize=True,
                                         input_core_dims=[['time'], ['time']],
                                         output_core_dims=[[]])
