import os
import warnings
import logging
import numpy as np
import pandas as pd
from timeit import default_timer as timer


def calc_T1(doy):
    return 1 * np.sin(2 * np.pi * ((doy - 21) / 365.))


def calc_T2(doy):
    return 1 * np.sin(2 * np.pi * ((doy - 81) / 365.))


def calc_T3(doy):
    return 1 * np.sin(2 * np.pi * ((doy - 111) / 365.))


def calc_T4(doy):
    return 1 * np.sin(2 * np.pi * ((doy - 141) / 365.))


def calc_seasonal_features(*args):
    """
    Calculate the seasonal features for the handed over day oder period of days

        arguments accepted:
        - String describing day, eg. '20171231'
        - Numpy array of dates
        - Numpy array of Pandas DatetimeIndexes
    """

    if len(args) != 1:
        logging.error('calc_seasonal_features() is not supposed to be called with this parameter set')
        exit(1)
    dayordays = args[0]

    if isinstance(dayordays, str):
        day = dayordays
        doy = int(pd.to_datetime(day).strftime('%j'))
        T1 = calc_T1(doy)
        T2 = calc_T2(doy)
        T3 = calc_T3(doy)
        T4 = calc_T4(doy)
    elif isinstance(dayordays, np.ndarray) or isinstance(dayordays, pd.core.indexes.datetimes.DatetimeIndex):
        days = dayordays
        doys = [int(pd.to_datetime(x).strftime('%j')) for x in days]
        T1 = [calc_T1(x) for x in doys]
        T2 = [calc_T2(x) for x in doys]
        T3 = [calc_T3(x) for x in doys]
        T4 = [calc_T4(x) for x in doys]
    else:
        logging.error('calc_seasonal_features() is not supposed to be called with this parameter set')
        exit(1)

    features_seas = {'T1': np.round(T1, 2), 'T2': np.round(T2, 2),
                     'T3': np.round(T3, 2), 'T4': np.round(T4, 2)}

    return features_seas


def init_logging():
    """
    Configure the logger
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(process)s - %(levelname)s - %(message)s')

    logging.getLogger("botocore.vendored.requests.packages.urllib3").setLevel(logging.WARNING)
    logging.getLogger('boto3').setLevel(logging.WARNING)
    logging.getLogger('pysftp').setLevel(logging.WARNING)
    logging.getLogger('paramiko').setLevel(logging.WARNING)
    warnings.simplefilter(action='ignore', category=FutureWarning)


def timer_start():
    """
    Start a timer
    """
    return (timer())


def timer_end(start, tab=1, log='info'):
    """
    Stop timer and log time
    """
    logging.info('{}... took {} sec'.format('    ' * tab, round((timer() - start), 3)))


def deep_dict_get(key, dictionary):
    """
    Helper function for finding a given key in a deep/nested dictionary
    """
    for k, v in dictionary.items():
        if k == key:
            yield v
        elif isinstance(v, dict):
            for result in deep_dict_get(key, v):
                yield result
        elif isinstance(v, list):
            for d in v:
                if isinstance(d, dict):
                    for result in deep_dict_get(key, d):
                        yield result


def make_dirs_avalable(dirs):
    """
    Create all directories handed over, if not yet available
    """
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)
            logging.info('created following dir: {}'.format(dir))
