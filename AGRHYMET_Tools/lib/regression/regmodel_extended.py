import logging
import copy
import xarray as xr
import numpy as np
import statsmodels.api as sm
from statsmodels.tools import eval_measures
from sklearn.feature_selection import f_regression
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats.stats import pearsonr
import operator

import lib.tools


def read_files(param_name, n, lat_sel, dir_in_EChres, dir_in_ERA5):
    """
    Helper function for reading ERA5/ECHres files for a given lat chunk
    """
    logging.info('    {} - read_files()'.format(n))
    fn_EChres = '{}/EChres_{}_aggregated_5degLons_{}.nc'.format(dir_in_EChres, param_name, n)
    fn_ERA5   = '{}/ERA5_{}_aggregated_5degLons_{}.nc'.format(dir_in_ERA5, param_name, n)

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


class Model_LassoFixedAlpha(Model):
    """
    This class implements the abstract Model()-class using a Lasso regression
    """

    def __init__(self, param_name, x, lat_sel, features_seas, dir_in_EChres, dir_in_ERA5, dir_out):
        Model.__init__(self, param_name, x, lat_sel, features_seas, dir_in_EChres, dir_in_ERA5, dir_out)

        self.model_name = 'LassoFixedAlpha'
        self.model_desc = 'Regression regularized by L1 norm (Lasso regression). The hyperparameter alpha is fixed at 1.0'

    def calc_model(self):
        logging.info('    {} - calc_model()'.format(self.x))

        global T1, T2, T3, T4
        T1, T2, T3, T4 = self.features_seas['T1'], self.features_seas['T2'], self.features_seas['T3'], self.features_seas['T4']

        def reg_xarray_LassoFixedAlpha(x, y):
            """
            Function implementing the model training
                x -> Feature/Predictor
                y -> Target/Predictand
            """

            alpha = 1
            X1 = np.column_stack((x, T1, T2, T3, T4))
            clf = linear_model.Lasso(alpha=alpha, fit_intercept=True).fit(X1, y)

            y_pred = clf.predict(X1)
            mae = mean_absolute_error(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            rsq = r2_score(y, y_pred)

            res = {'mae': mae,
                   'rmse': rmse,
                   'rsq': rsq,
                   'rsq_adj': None,
                   'bic': None,
                   'nnonzero': np.count_nonzero(y_pred),
                   'nfeat': None,
                   'intercept': clf.intercept_,
                   'coef_X':  clf.coef_[0],
                   'coef_T1': clf.coef_[1],
                   'coef_T2': clf.coef_[2],
                   'coef_T3': clf.coef_[3],
                   'coef_T4': clf.coef_[4]}
            return res

        self.models_raw = xr.apply_ufunc(reg_xarray_LassoFixedAlpha,
                                         self.da_era5, self.da_echres,
                                         vectorize=True,
                                         input_core_dims=[['time'], ['time']],
                                         output_core_dims=[[]])


class Model_OneReg4Seasonal(Model):
    """
    This class implements the abstract Model()-class using a simple OLS regression
    """

    def __init__(self, param_name, x, lat_sel, features_seas, dir_in_EChres, dir_in_ERA5, dir_out):
        Model.__init__(self, param_name, x, lat_sel, features_seas, dir_in_EChres, dir_in_ERA5, dir_out)

        self.model_name = 'ModelOneReg4Seasonal'
        self.model_desc = 'Simple OLS regression with 4 seasonal features, no explicit feature selection'

    def calc_model(self):
        logging.info('    {} - calc_model()'.format(self.x))

        global T1, T2, T3, T4
        T1, T2, T3, T4 = self.features_seas['T1'], self.features_seas['T2'], self.features_seas['T3'], self.features_seas['T4']

        def reg_xarray_OneReg4Seasonal(x, y):
            """
            Function implementing the model training
                x -> Feature/Predictor
                y -> Target/Predictand
            """

            X1 = np.column_stack((x, T1, T2, T3, T4))
            X2 = sm.add_constant(X1)
            mod = sm.OLS(y, X2)
            res = mod.fit()

            res2 = {'mae': eval_measures.meanabs(res.fittedvalues, y),
                    'rmse': np.sqrt(res.mse_total),
                    'rsq': res.rsquared,
                    'rsq_adj': res.rsquared_adj,
                    'bic': res.bic,
                    'nnonzero': np.count_nonzero(res.fittedvalues),
                    'nfeat': 6,
                    'intercept': res.params[0],
                    'coef_X':  res.params[1],
                    'coef_T1': res.params[2],
                    'coef_T2': res.params[3],
                    'coef_T3': res.params[4],
                    'coef_T4': res.params[5],
                    }
            return res2

        self.models_raw = xr.apply_ufunc(reg_xarray_OneReg4Seasonal,
                                         self.da_era5, self.da_echres,
                                         vectorize=True,
                                         input_core_dims=[['time'], ['time']],
                                         output_core_dims=[[]])


class Model_OneReg2Seasonal(Model):
    """
    This class implements the abstract Model()-class using a simple OLS regression
    """

    def __init__(self, param_name, x, lat_sel, features_seas, dir_in_EChres, dir_in_ERA5, dir_out):
        Model.__init__(self, param_name, x, lat_sel, features_seas, dir_in_EChres, dir_in_ERA5, dir_out)

        self.model_name = 'ModelOneReg2Seasonal'
        self.model_desc = 'Simple OLS regression with 2 preselected seasonal features, no explicit feature selection'

    def calc_model(self):
        logging.info('    {} - calc_model()'.format(self.x))

        global T1, T2, T3, T4
        T1, T2, T3, T4 = self.features_seas['T1'], self.features_seas['T2'], self.features_seas['T3'], self.features_seas['T4']

        def reg_xarray_OneReg2Seasonal(x, y):
            """
            Function implementing the model training
                x -> Feature/Predictor
                y -> Target/Predictand
            """

            X1 = np.column_stack((x, T1, T3))
            X2 = sm.add_constant(X1)
            mod = sm.OLS(y, X2)
            res = mod.fit()

            res2 = {'mae': eval_measures.meanabs(res.fittedvalues, y),
                    'rmse': np.sqrt(res.mse_total),
                    'rsq': res.rsquared,
                    'rsq_adj': res.rsquared_adj,
                    'bic': res.bic,
                    'nnonzero': np.count_nonzero(res.fittedvalues),
                    'nfeat': 4,
                    'intercept': res.params[0],
                    'coef_X':  res.params[1],
                    'coef_T1': res.params[2],
                    'coef_T2': None,
                    'coef_T3': res.params[3],
                    'coef_T4': None}
            return res2

        self.models_raw = xr.apply_ufunc(reg_xarray_OneReg2Seasonal,
                                         self.da_era5, self.da_echres,
                                         vectorize=True,
                                         input_core_dims=[['time'], ['time']],
                                         output_core_dims=[[]])


class Model_OneRegSimple(Model):
    """
    This class implements the abstract Model()-class using a simple OLS regression
    """

    def __init__(self, param_name, x, lat_sel, features_seas, dir_in_EChres, dir_in_ERA5, dir_out):
        Model.__init__(self, param_name, x, lat_sel, features_seas, dir_in_EChres, dir_in_ERA5, dir_out)

        self.model_name = 'ModelOneRegSimple'
        self.model_desc = 'Simple OLS regression without any additional features'

    def calc_model(self):
        logging.info('    {} - calc_model()'.format(self.x))

        def reg_xarray_OneRegSimple(x, y):
            """
            Function implementing the model training
                x -> Feature/Predictor
                y -> Target/Predictand
            """

            X2 = sm.add_constant(x)
            mod = sm.OLS(y, X2)
            res = mod.fit()

            res2 = {'mae': eval_measures.meanabs(res.fittedvalues, y),
                    'rmse': np.sqrt(res.mse_total),
                    'rsq': res.rsquared,
                    'rsq_adj': res.rsquared_adj,
                    'bic': res.bic,
                    'nnonzero': np.count_nonzero(res.fittedvalues),
                    'nfeat': 2,
                    'intercept': res.params[0],
                    'coef_X': res.params[1],
                    'coef_T1': None,
                    'coef_T2': None,
                    'coef_T3': None,
                    'coef_T4': None}
            return res2

        self.models_raw = xr.apply_ufunc(reg_xarray_OneRegSimple,
                                         self.da_era5, self.da_echres,
                                         vectorize=True,
                                         input_core_dims=[['time'], ['time']],
                                         output_core_dims=[[]])


class Model_FRegression(Model):
    """
    This class implements the abstract Model()-class using a F-regression
    """

    def __init__(self, param_name, x, lat_sel, features_seas, dir_in_EChres, dir_in_ERA5, dir_out):
        Model.__init__(self, param_name, x, lat_sel, features_seas, dir_in_EChres, dir_in_ERA5, dir_out)

        self.model_name = 'ModelFRegression'
        self.model_desc = 'F-Regression is used to train the final regression with 2 most significant seasonal features'

    def calc_model(self):
        logging.info('    {} - calc_model()'.format(self.x))

        global T1, T2, T3, T4
        T1, T2, T3, T4 = self.features_seas['T1'], self.features_seas['T2'], self.features_seas['T3'], self.features_seas['T4']

        def reg_xarray_FRegression(x, y):
            """
            Function implementing the model training
                x -> Feature/Predictor
                y -> Target/Predictand
            """

            # Organize features and run f-regression
            X = np.column_stack((x, T1, T2, T3, T4))
            f_test, _ = f_regression(X, y)
            f_test /= np.max(f_test)
            f_test = np.abs(f_test)

            # Get best 2 seasonal features
            n_feat = 2
            best_feat = np.argsort(f_test)[::-1][0:n_feat + 1]
            X = X[:, best_feat]

            # Calculate final regression
            X = sm.add_constant(X)
            mod = sm.OLS(y, X)
            mod = mod.fit()

            # Reorder coefs of features
            def get_single_coefs(best_mod, best_feat):
                coefs = []
                for feat in ['const', 'X', 'T1', 'T2', 'T3', 'T4']:
                    if feat in best_feat:
                        coefs.append(best_mod.params[best_feat.index(feat)])
                    else:
                        coefs.append(None)
                return coefs
            feats = ['X', 'T1', 'T2', 'T3', 'T4']
            feats = ['const'] + np.array(feats)[best_feat].tolist()
            coefs = get_single_coefs(mod, feats)

            res2 = {'mae': eval_measures.meanabs(mod.fittedvalues, y),
                    'rmse': eval_measures.rmse(mod.fittedvalues, y),
                    'maxabs': eval_measures.maxabs(mod.fittedvalues, y),
                    'bias': eval_measures.bias(mod.fittedvalues, y),
                    'rsq': mod.rsquared,
                    # 'rsq_adj': mod.rsquared_adj,
                    'bic': mod.bic,
                    'nnonzero': np.count_nonzero(mod.fittedvalues),
                    'nfeat': len(best_feat),
                    'intercept': coefs[0],
                    'coef_X': coefs[1],
                    'coef_T1': coefs[2],
                    'coef_T2': coefs[3],
                    'coef_T3': coefs[4],
                    'coef_T4': coefs[5]}
            return res2

        self.models_raw = xr.apply_ufunc(reg_xarray_FRegression,
                                         self.da_era5, self.da_echres,
                                         vectorize=True,
                                         input_core_dims=[['time'], ['time']],
                                         output_core_dims=[[]])


class Model_StepwiseSimple(Model):
    """
    This class implements the abstract Model()-class using a forward stepwise regression approach
    """

    def __init__(self, param_name, x, lat_sel, features_seas, dir_in_EChres, dir_in_ERA5, dir_out):
        Model.__init__(self, param_name, x, lat_sel, features_seas, dir_in_EChres, dir_in_ERA5, dir_out)

        self.model_name = 'ModelStepwiseSimple'
        self.model_desc = 'Forward stepwise Regression based on predefined feature combinations. Models are evaluated using BIC.'

    def calc_model(self):
        logging.info('    {} - calc_model()'.format(self.x))

        global T1, T2, T3, T4
        T1, T2, T3, T4 = self.features_seas['T1'], self.features_seas['T2'], self.features_seas['T3'], self.features_seas['T4']

        def reg_xarray_StepwiseSimple(x, y):
            """
            Function implementing the model training
                x -> Feature/Predictor
                y -> Target/Predictand
            """

            # Organize features
            map_feat_ind = {'const': 0, 'X': 1, 'T1': 2, 'T2': 3, 'T3': 4, 'T4': 5}
            np_X = sm.add_constant(np.column_stack((x, T1, T2, T3, T4)))

            def get_sub_np_features(np_X, list_feat):
                ind = [map_feat_ind[x] for x in list_feat]
                return np.take(np_X, ind, axis=1)

            def get_sorted_corrs_np(Y, X, features_names):
                test2 = {y: abs(pearsonr(Y, X[:, x])[0]) for (x, y) in enumerate(features_names)}
                sorted_x = sorted(test2.items(), key=operator.itemgetter(1), reverse=True)
                return [x[0] for x in sorted_x], [x[1] for x in sorted_x]

            # Find best feature based on correlation coeficient
            list_feat = ['T2', 'T3', 'T4']
            best_X, corr_vals = get_sorted_corrs_np(y, get_sub_np_features(np_X, list_feat), list_feat)

            # # Calc VIF
            # from statsmodels.stats.outliers_influence import variance_inflation_factor
            # pd_X = pd.DataFrame(np_X)
            # vif["VIF Factor"] = [variance_inflation_factor(pd_X.values, i) for i in range(pd_X.shape[1])]
            # vif["features"] = pd_X.columns

            # OLS regression
            def myreg_np(Y, np_X):
                return sm.OLS(Y, np_X).fit()

            # All features sets beeing tested in regression
            cfgs = [['const'],
                    ['const', 'X'],
                    ['const', 'X', 'T1'],
                    ['const', 'X'] + best_X[:1],
                    ['const', 'X', 'T1'] + best_X[:1],
                    # ['const', 'X'] + best_X[:2],
                    # ['const', 'X'] + best_X[:3],
                    # ['const', 'X'] + best_X[:4]
                    ]

            # Iterate through all predefined feature combinations and find best set based on BIC
            bic_curr = 100000
            best_mod = None
            best_feat = None
            for cfg in cfgs:
                this_mod = myreg_np(y, get_sub_np_features(np_X, cfg))
                if this_mod.bic < bic_curr:
                    best_mod = this_mod
                    best_feat = cfg
                    bic_curr = this_mod.bic

            # Reorganize coefs of features
            def get_single_coefs(best_mod, best_feat):
                coefs = []
                for feat in ['const', 'X', 'T1', 'T2', 'T3', 'T4']:
                    if feat in best_feat:
                        coefs.append(best_mod.params[best_feat.index(feat)])
                    else:
                        coefs.append(None)
                return coefs
            coefs = get_single_coefs(best_mod, best_feat)

            res2 = {'mae': eval_measures.meanabs(best_mod.fittedvalues, y),
                    'rmse': np.sqrt(best_mod.mse_total),
                    'rsq': best_mod.rsquared,
                    'rsq_adj': best_mod.rsquared_adj,
                    'bic': best_mod.bic,
                    'nnonzero': np.count_nonzero(best_mod.fittedvalues),
                    'nfeat': len(best_feat),
                    'intercept': coefs[0],
                    'coef_X': coefs[1],
                    'coef_T1': coefs[2],
                    'coef_T2': coefs[3],
                    'coef_T3': coefs[4],
                    'coef_T4': coefs[5]}
            return res2

        self.models_raw = xr.apply_ufunc(reg_xarray_StepwiseSimple,
                                         self.da_era5, self.da_echres,
                                         vectorize=True,
                                         input_core_dims=[['time'], ['time']],
                                         output_core_dims=[[]])


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
            # map_feat_ind = {'X': 0, 'T1': 1, 'T2': 2, 'T3': 3, 'T4': 4}
            np_X = sm.add_constant(np.column_stack((x, T1, T2, T3, T4)), has_constant='add')
            # np_X = np.column_stack((x, T1, T2, T3, T4))

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
            # list_feat = ['X'] + best_X[:2]
            mod = myreg_np(y, get_sub_np_features(np_X, list_feat))

            # Reorganize coefs of features
            def get_single_coefs(best_mod, best_feat):
                coefs = []
                # for feat in ['X', 'T1', 'T2', 'T3', 'T4']:
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


HYPERPARA_Cs = {'all': [1, 5, 10, 50, 100, 500, 1000, 2500, 5000, 7500, 10000, 15000],
                'reduced_generic': [2500, 5000, 7500, 10000, 15000]}
HYPERPARA_gammas = {'all': [0.00001, 0.00005, 0.000075, 0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.05, 0.1],
                    'reduced_generic': [0.00001, 0.00005, 0.000075, 0.0001, 0.00025]}


class Model_SVR(Model):
    """
    This class implements the abstract Model()-class using a non-linear Support Vector Regression (SVR) with a rbf kernel

        Please NOTE: As this nonlinear model is not compatible with the selected approach of serializing
                     the model not as object instances but as simple regression equations, this implentation
                     is not fully working in its current form. It was only used for a simple POC.
    """

    def __init__(self, param_name, x, lat_sel, features_seas, dir_in_EChres, dir_in_ERA5, dir_out):
        Model.__init__(self, param_name, x, lat_sel, features_seas, dir_in_EChres, dir_in_ERA5, dir_out)

        self.model_name = 'ModelSVR'
        self.model_desc = 'SVR model using a preselected set of hyperparameters using a rbf kernel'

    def calc_model(self):
        logging.info('    {} - calc_model()'.format(self.x))

        global T1, T2, T3, T4
        T1, T2, T3, T4 = self.features_seas['T1'], self.features_seas['T2'], self.features_seas['T3'], self.features_seas['T4']

        def mod_xarray_svr(x, y):
            """
            Function implementing the model training
                x -> Feature/Predictor
                y -> Target/Predictand
            """
            from sklearn import preprocessing
            from sklearn.svm import SVR
            from sklearn.model_selection import GridSearchCV

            # Organize Features
            # map_feat_ind = {'const': 0, 'X': 1, 'T1': 2, 'T2': 3, 'T3': 4, 'T4': 5}
            np_X = sm.add_constant(np.column_stack((x, T1, T2, T3, T4)), has_constant='add')

            # Scale features
            scaler = preprocessing.StandardScaler().fit(np_X)
            X_train_norm = scaler.transform(np_X)

            # Find optimal hyperparameters
            hyperpara_tuning = 'reduced_generic'
            hyperpara_tuning = False
            if hyperpara_tuning:
                param_grid = {'C': HYPERPARA_Cs.get(hyperpara_tuning),
                              'gamma': HYPERPARA_gammas.get(hyperpara_tuning)}
                grid_search = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=2, n_jobs=-1)
                grid_search.fit(X_train_norm, y)
                C, gamma = grid_search.best_params_['C'], grid_search.best_params_['gamma']
            else:
                C, gamma = 1000, 0.004

            # Train and predict estimator
            estimator = SVR(kernel='rbf', C=C, gamma=gamma)
            estimator.fit(X_train_norm, y)
            R2 = estimator.score(X_train_norm, y)
            y_hat = estimator.predict(X_train_norm)

            res2 = {'mae': eval_measures.meanabs(y_hat, y),
                    'rmse': eval_measures.rmse(y_hat, y),
                    'maxabs': eval_measures.maxabs(y_hat, y),
                    'bias': eval_measures.bias(y_hat, y),
                    'rsq': R2,
                    # 'rsq': None,
                    # 'rsq_adj': None,
                    # 'bic': mod.bic,
                    'bic': None,
                    'nnonzero': np.count_nonzero(x),
                    'nfeat': None,
                    'intercept': None,
                    'coef_X': None,
                    'coef_T1': None,
                    'coef_T2': None,
                    'coef_T3': None,
                    'yhat': y_hat,
                    'coef_T4': None}
            return res2

        self.models_raw = xr.apply_ufunc(mod_xarray_svr,
                                         self.da_era5, self.da_echres,
                                         vectorize=True,
                                         input_core_dims=[['time'], ['time']],
                                         output_core_dims=[[]])
