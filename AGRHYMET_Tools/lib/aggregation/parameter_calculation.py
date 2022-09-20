""""
@Songoti Henri Juil 2020
@Centre regional AGRHYMET/CILSS
@Adaptation de code issus de : @JRC et @FAO
"""

import xarray as xr
import numpy as np
import logging


class ParameterCalculator():
    def __init__(self, cfg_parameter):
        self.cfg_parameter = cfg_parameter

    def do_it(self, indata):
        """
        Do derived parameter calculation (eg u10/v10 -> ff)
        """
        logging.info('  Calculate derived parameter...')
        self.__check_indata(indata)
        # Apply Parameter calculation function
        derived_parameter_func = self.cfg_parameter['ParamFunc']
        print('Allo')
        print(indata)
        data_der = derived_parameter_func(indata)

        # Set parameter name and unit
        print('testtest')
        print(data_der)
        data_der.name = self.cfg_parameter['ParamResult']['NcParamName']
        data_der.attrs['units'] = self.cfg_parameter['ParamResult']['Unit']
        data_der.attrs['long_name'] = self.cfg_parameter['ParamResult']['LongName']
        data_der.attrs['aggregation'] = self.cfg_parameter['ParamResult']['AggrName']

        # Convert from xr.Dataarry to xr.Dataset
        data_der = data_der.to_dataset()

        return data_der

    def __check_indata(self, indata):
        """
        Check if all needed source parameters are available
        """
        needed_params = self.cfg_parameter['ParamSource']['NcParamName']
        for param in needed_params:
            if param not in indata.data_vars.keys():
                logging.error('Expected parameter not availble!')


def calc_10u10v_to_ff(da):
    """
    Function to calculate the 10m windspeed from the 10m u and v wind components
        ff = sqrt(10u**2 + 10v**2)
    """
    logging.info('    calc 10u/10v -> ff')

    # Calc windspeed
    ff = xr.ufuncs.sqrt(da.u10 ** 2 + da.v10 ** 2)

    # Set attributes
    ff.attrs = da.u10.attrs

    return ff


def calc_sdrsn_to_sdDer(da):
    """
    Function to calculate the snow depth from snow density (rsn) and snow depth of water equivalent (sd)
        sdDer = (sd / rsn) * 1000 * 100
            sd [mm of liquid water] or [kg/m2]
            rsn [kg/m3]
            sdDer [cm of snow]
    """
    logging.info('    calc sd/rsn -> sdDer')

    # Calc snow depth
    sdDer = (da.sd / da.rsn) * 1000 * 100

    # Set attributes
    sdDer.attrs = da.sd.attrs

    return sdDer


# def calc_2t2d_to_vp(da):
#     """
#     Alternative function for vapour pressure
#     2d,2t -> rh
#     2t,rh -> ea
#     gives nealy same results as calc_2d_to_vp()
#     """
#     logging.info('    calc 2t/2d -> vp')
#
#     import numpy as np
#     Td = da.d2m - 273.15
#     T = da.t2m - 273.15
#     rh = 100 * (np.exp((17.27 * Td) / (237.3 + Td)) / np.exp((17.27 * T) / (237.3 + T)))
#     # import pdb; pdb.set_trace()
#
#     ea = rh / 100 * 6.105 * np.exp((17.27 * T) / (237.7 + T))
#
#     # Set attributes
#     ea.attrs = da.d2m.attrs
#
#     return ea


def synth_sum_func_OLD(da):
    """
    Aggregation function for calculating the SUM of a given sequence of timeslots
    """

    da_aggr = da.sum(dim='time')
    # Get parameter name
    par_names = list(da.data_vars.keys())
    if len(par_names) != 1:
        logging.error('something wrong')
        exit(1)
    par_name = par_names[0]

    # Set name and attributes
    da_aggr.attrs = da[par_name].attrs

    return da_aggr


def synth_sum_func(da, lon_range, zone_timesteps, slots):
    """
    Aggregation function for calculating the MEAN of a given sequence of timeslots
    """

    ts_start, ts_end = zone_timesteps
    ts_range = slice(ts_start, ts_end)

    da_sel = da.sel(ts=ts_range, lon=lon_range)
    ts_n = len(da_sel.ts.data)
    da_aggr = da_sel.mean(dim='ts')

    # Write string describing operation
    slots_selected = [slots[x] for x in da_sel.ts.data]
    explanation = ', '.join(slots_selected)
    explanation = 'sum({})/{}'.format(explanation, ts_n)

    return da_aggr, explanation




def synth_mean_func(da):
    """
    Aggregation function for calculating the mean of a given sequence of timeslots
    """

    da_aggr = da.mean(dim='time')
    # Get parameter name
    par_names = list(da.data_vars.keys())
    if len(par_names) != 1:
        logging.error('something wrong')
        exit(1)
    par_name = par_names[0]

    # Set name and attributes
    da_aggr.attrs = da[par_name].attrs

    return da_aggr


def calc_2d_to_vp(da):
    """
    Function to calculate the partial water vapour pressure from dewpoint temperature (Priestley and Taylor, 1972))
        ea = 10 * 0.6108 * .exp((17.27 * Td) / (Td + 237.3))
            Td: Dewpoint [degC]
            ea: Vapour pressure [hPa]
    """
    logging.info('    calc 2d -> vp')

    # Calc partial water vapour pressure in hPa from dewpoint in Kelvin
    Td_C = da.d2m - 273.15
    ea = 10 * 0.6108 * xr.ufuncs.exp((17.27 * Td_C) / (Td_C + 237.3))

    # Set attributes
    ea.attrs = da.d2m.attrs

    return ea


def calc_2t2d_to_rh(da):
    """
    Function to calculate the relative humidity from 2m temperature and dewpoint temperature
        rh = 100 * (exp((17.27 * Td) / (237.3 + Td)) / exp((17.27 * T) / (237.3 + T)))
            T:  2m temperature [degC]
            Td: 2m dewpoint temperature [degC]
            rh: relative humidity [%]
    """
    logging.info('    calc 2t/2d -> rh')

    Td_C = da.d2m - 273.15
    T_C = da.t2m - 273.15
    rh = 100 * (xr.ufuncs.exp((17.27 * Td_C) / (237.3 + Td_C)) / xr.ufuncs.exp((17.27 * T_C) / (237.3 + T_C)))

    # Set attributes
    rh.attrs = da.d2m.attrs

    return rh


def unit_tp_m_to_mm(da):
    """
    Function for unit conversion of precipitation: m -> mm
    """
    logging.info('    unit m -> mm')

    tp = da.tp * 1000

    # Set attributes
    tp.attrs = da.tp.attrs

    return tp


def unit_sd_m_to_cm(da):
    """
    Function for unit conversion of snow: m -> cm
    """
    logging.info('    unit m -> cm')

    sd = da.sd * 100

    # Set name and attributes
    sd.attrs = da.sd.attrs

    return sd


def unit_temp_K_to_degC(da):
    """
    Function for unit conversion of temperature: K -> degC
    """
    logging.info('    unit K -> degC')

    # Get parameter name
    par_names = list(da.data_vars.keys())
    if len(par_names) != 1:
        logging.error('something wrong')
        exit(1)
    par_name = par_names[0]

    temp = da[par_name] - 273.15

    # Set name and attributes
    temp.attrs = da[par_name].attrs

    return temp


def unit_temp_degC_to_K(da):
    """
    Function for unit conversion of temperature:  deg -> CK
    """
    logging.info('    unit K -> degC')

    # Get parameter name
    par_names = list(da.data_vars.keys())
    if len(par_names) != 1:
        logging.error('something wrong')
        exit(1)
    par_name = par_names[0]

    temp = da[par_name] + 273.16

    # Set name and attributes
    temp.attrs = da[par_name].attrs

    return temp


def unit_tcc_01_to_percent(da):
    """
    Function for unit conversion of cloud cover: 0-1 -> percent
    """
    logging.info('    unit 0-1 -> percent')

    tcc = da.tcc * 100

    # Set name and attributes
    tcc.attrs = da.tcc.attrs

    return tcc


"""
Library of functions for estimating reference evapotransporation (ETo) for
a grass reference crop using the FAO-56 Penman-Monteith and Hargreaves
equations. The library includes numerous functions for estimating missing
meteorological data.

:copyright: (c) 2015 by Mark Richards.
:license: BSD 3-Clause, see LICENSE.txt for more details.
"""

import math

from ._check import (
    check_day_hours as _check_day_hours,
    check_doy as _check_doy,
    check_latitude_rad as _check_latitude_rad,
    check_sol_dec_rad as _check_sol_dec_rad,
    check_sunset_hour_angle_rad as _check_sunset_hour_angle_rad,
)

#: Solar constant [ MJ m-2 min-1]
SOLAR_CONSTANT = 0.0820

# Stefan Boltzmann constant [MJ K-4 m-2 day-1]
STEFAN_BOLTZMANN_CONSTANT = 0.000000004903  #
"""Stefan Boltzmann constant [MJ K-4 m-2 day-1]"""


def atm_pressure(da):
    """
    Estimate atmospheric pressure from altitude.

    Calculated using a simplification of the ideal gas law, assuming 20 degrees
    Celsius for a standard atmosphere. Based on equation 7, page 62 in Allen
    et al (1998).

    :param altitude: Elevation/altitude above sea level [m]
    :return: atmospheric pressure [kPa]
    :rtype: float
    """
    altitude = da.ALT
    tmp = (293.0 - (0.0065 * altitude)) / 293.0
    atm_pressure = np.power(tmp, 5.26) * 101.3

    # Set name and attributes
    atm_pressure.attrs = da.ALT.attrs

    return atm_pressure


def avp_from_tmin(da):
    """
    Estimate actual vapour pressure (*ea*) from minimum temperature.

    This method is to be used where humidity data are lacking or are of
    questionable quality. The method assumes that the dewpoint temperature
    is approximately equal to the minimum temperature (*tmin*), i.e. the
    air is saturated with water vapour at *tmin*.

    **Note**: This assumption may not hold in arid/semi-arid areas.
    In these areas it may be better to subtract 2 deg C from the
    minimum temperature (see Annex 6 in FAO paper).

    Based on equation 48 in Allen et al (1998).

    :param tmin: Daily minimum temperature [deg C]
    :return: Actual vapour pressure [kPa]
    :rtype: float
    """
    tmin = da.TN
    avp = 0.611 * math.exp((17.27 * tmin) / (tmin + 237.3))
    # Set name and attributes
    avp.attrs = da.TN.attrs
    return avp


def avp_from_rhmin_rhmax(svp_tmin, svp_tmax, rh_min, rh_max):
    """
    Estimate actual vapour pressure (*ea*) from saturation vapour pressure and
    relative humidity.

    Based on FAO equation 17 in Allen et al (1998).

    :param svp_tmin: Saturation vapour pressure at daily minimum temperature
        [kPa]. Can be estimated using ``svp_from_t()``.
    :param svp_tmax: Saturation vapour pressure at daily maximum temperature
        [kPa]. Can be estimated using ``svp_from_t()``.
    :param rh_min: Minimum relative humidity [%]
    :param rh_max: Maximum relative humidity [%]
    :return: Actual vapour pressure [kPa]
    :rtype: float
    """
    tmp1 = svp_tmin * (rh_max / 100.0)
    tmp2 = svp_tmax * (rh_min / 100.0)
    return (tmp1 + tmp2) / 2.0


def avp_from_rhmax(svp_tmin, rh_max):
    """
    Estimate actual vapour pressure (*e*a) from saturation vapour pressure at
    daily minimum temperature and maximum relative humidity

    Based on FAO equation 18 in Allen et al (1998).

    :param svp_tmin: Saturation vapour pressure at daily minimum temperature
        [kPa]. Can be estimated using ``svp_from_t()``.
    :param rh_max: Maximum relative humidity [%]
    :return: Actual vapour pressure [kPa]
    :rtype: float
    """
    return svp_tmin * (rh_max / 100.0)


def avp_from_rhmean(svp_tmin, svp_tmax, rh_mean):
    """
    Estimate actual vapour pressure (*ea*) from saturation vapour pressure at
    daily minimum and maximum temperature, and mean relative humidity.

    Based on FAO equation 19 in Allen et al (1998).

    :param svp_tmin: Saturation vapour pressure at daily minimum temperature
        [kPa]. Can be estimated using ``svp_from_t()``.
    :param svp_tmax: Saturation vapour pressure at daily maximum temperature
        [kPa]. Can be estimated using ``svp_from_t()``.
    :param rh_mean: Mean relative humidity [%] (average of RH min and RH max).
    :return: Actual vapour pressure [kPa]
    :rtype: float
    """
    return (rh_mean / 100.0) * ((svp_tmax + svp_tmin) / 2.0)


def avp_from_tdew(da):
    """
    Estimate actual vapour pressure (*ea*) from dewpoint temperature.

    Based on equation 14 in Allen et al (1998). As the dewpoint temperature is
    the temperature to which air needs to be cooled to make it saturated, the
    actual vapour pressure is the saturation vapour pressure at the dewpoint
    temperature.

    This method is preferable to calculating vapour pressure from
    minimum temperature.

    :param tdew: Dewpoint temperature [deg C]
    :return: Actual vapour pressure [kPa]
    :rtype: float
    """
    tdew = da.T2D
    tmp1 = ((17.27 * tdew) / (tdew + 237.3))
    avp = 0.6108 * math.exp(tmp1)

    avp.attrs = da.T2D.attrs
    return avp
    # return 0.6108 * math.exp((17.27 * tdew) / (tdew + 237.3))


def avp_from_twet_tdry(twet, tdry, svp_twet, psy_const):
    """
    Estimate actual vapour pressure (*ea*) from wet and dry bulb temperature.

    Based on equation 15 in Allen et al (1998). As the dewpoint temperature
    is the temperature to which air needs to be cooled to make it saturated, the
    actual vapour pressure is the saturation vapour pressure at the dewpoint
    temperature.

    This method is preferable to calculating vapour pressure from
    minimum temperature.

    Values for the psychrometric constant of the psychrometer (*psy_const*)
    can be calculated using ``psyc_const_of_psychrometer()``.

    :param twet: Wet bulb temperature [deg C]
    :param tdry: Dry bulb temperature [deg C]
    :param svp_twet: Saturated vapour pressure at the wet bulb temperature
        [kPa]. Can be estimated using ``svp_from_t()``.
    :param psy_const: Psychrometric constant of the pyschrometer [kPa deg C-1].
        Can be estimated using ``psy_const()`` or
        ``psy_const_of_psychrometer()``.
    :return: Actual vapour pressure [kPa]
    :rtype: float
    """
    return svp_twet - (psy_const * (tdry - twet))


def cs_rad(altitude, et_rad):
    """
    Estimate clear sky radiation from altitude and extraterrestrial radiation.

    Based on equation 37 in Allen et al (1998) which is recommended when
    calibrated Angstrom values are not available.

    :param altitude: Elevation above sea level [m]
    :param et_rad: Extraterrestrial radiation [MJ m-2 day-1]. Can be
        estimated using ``et_rad()``.
    :return: Clear sky radiation [MJ m-2 day-1]
    :rtype: float
    """
    return (0.00002 * altitude + 0.75) * et_rad


def daily_mean_t(da):
    """
    Estimate mean daily temperature from the daily minimum and maximum
    temperatures.

    :param tmin: Minimum daily temperature [deg C]
    :param tmax: Maximum daily temperature [deg C]
    :return: Mean daily temperature [deg C]
    :rtype: float
    """
    tmin = da.TN
    tmax = da.TX
    tmoy = (tmax + tmin) / 2.0
    tmoy.attrs = da.TN.attrs
    return tmoy


def daylight_hours(sha):
    """
    Calculate daylight hours from sunset hour angle.

    Based on FAO equation 34 in Allen et al (1998).

    :param sha: Sunset hour angle [rad]. Can be calculated using
        ``sunset_hour_angle()``.
    :return: Daylight hours.
    :rtype: float
    """
    _check_sunset_hour_angle_rad(sha)
    return (24.0 / math.pi) * sha


def delta_svp(da):
    """
    Estimate the slope of the saturation vapour pressure curve at a given
    temperature.

    Based on equation 13 in Allen et al (1998). If using in the Penman-Monteith
    *t* should be the mean air temperature.

    :param t: Air temperature [deg C]. Use mean air temperature for use in
        Penman-Monteith.
    :return: Saturation vapour pressure [kPa degC-1]
    :rtype: float
    """
    t = da.T2M
    tmp = 4098 * (0.6108 * math.exp((17.27 * t) / (t + 237.3)))
    my_delta_svp = tmp / math.pow((t + 237.3), 2)
    my_delta_svp.attrs = da.T2M.attrs
    return my_delta_svp


def energy2evap(energy):
    """
    Convert energy (e.g. radiation energy) in MJ m-2 day-1 to the equivalent
    evaporation, assuming a grass reference crop.

    Energy is converted to equivalent evaporation using a conversion
    factor equal to the inverse of the latent heat of vapourisation
    (1 / lambda = 0.408).

    Based on FAO equation 20 in Allen et al (1998).

    :param energy: Energy e.g. radiation or heat flux [MJ m-2 day-1].
    :return: Equivalent evaporation [mm day-1].
    :rtype: float
    """
    return 0.408 * energy


def et_rad(latitude, sol_dec, sha, ird):
    """
    Estimate daily extraterrestrial radiation (*Ra*, 'top of the atmosphere
    radiation').

    Based on equation 21 in Allen et al (1998). If monthly mean radiation is
    required make sure *sol_dec*. *sha* and *irl* have been calculated using
    the day of the year that corresponds to the middle of the month.

    **Note**: From Allen et al (1998): "For the winter months in latitudes
    greater than 55 degrees (N or S), the equations have limited validity.
    Reference should be made to the Smithsonian Tables to assess possible
    deviations."

    :param latitude: Latitude [radians]
    :param sol_dec: Solar declination [radians]. Can be calculated using
        ``sol_dec()``.
    :param sha: Sunset hour angle [radians]. Can be calculated using
        ``sunset_hour_angle()``.
    :param ird: Inverse relative distance earth-sun [dimensionless]. Can be
        calculated using ``inv_rel_dist_earth_sun()``.
    :return: Daily extraterrestrial radiation [MJ m-2 day-1]
    :rtype: float
    """
    _check_latitude_rad(latitude)
    _check_sol_dec_rad(sol_dec)
    _check_sunset_hour_angle_rad(sha)

    tmp1 = (24.0 * 60.0) / math.pi
    tmp2 = sha * math.sin(latitude) * math.sin(sol_dec)
    tmp3 = math.cos(latitude) * math.cos(sol_dec) * math.sin(sha)
    return tmp1 * SOLAR_CONSTANT * ird * (tmp2 + tmp3)


def fao56_penman_monteith(da):
    """
    Estimate reference evapotranspiration (ETo) from a hypothetical
    short grass reference surface using the FAO-56 Penman-Monteith equation.

    Based on equation 6 in Allen et al (1998).

    :param net_rad: Net radiation at crop surface [MJ m-2 day-1]. If
        necessary this can be estimated using ``net_rad()``.
    :param t: Air temperature at 2 m height [deg Kelvin].
    :param ws: Wind speed at 2 m height [m s-1]. If not measured at 2m,
        convert using ``wind_speed_at_2m()``.
    :param svp: Saturation vapour pressure [kPa]. Can be estimated using
        ``svp_from_t()''.
    :param avp: Actual vapour pressure [kPa]. Can be estimated using a range
        of functions with names beginning with 'avp_from'.
    :param delta_svp: Slope of saturation vapour pressure curve [kPa degC-1].
        Can be estimated using ``delta_svp()``.
    :param psy: Psychrometric constant [kPa deg C]. Can be estimatred using
        ``psy_const_of_psychrometer()`` or ``psy_const()``.
    :param shf: Soil heat flux (G) [MJ m-2 day-1] (default is 0.0, which is
        reasonable for a daily or 10-day time steps). For monthly time steps
        *shf* can be estimated using ``monthly_soil_heat_flux()`` or
        ``monthly_soil_heat_flux2()``.
    :return: Reference evapotranspiration (ETo) from a hypothetical
        grass reference surface [mm day-1].
    :rtype: float


        logging.info('    calc 10u/10v -> ff')

    # Calc windspeed
    ff = xr.ufuncs.sqrt(da.u10 ** 2 + da.v10 ** 2)

    # Set attributes
    ff.attrs = da.u10.attrs

    return ff
    fao56_penman_monteith(net_rad, t, ws, svp, avp, delta_svp, psy, shf=0.0)
    """
    shf = 0.0
    net_rad = da.SSRD/1000.
    t = da.T2M
    tn = da.TN
    tx = da.TX

    print('qui_suis_je')

    t2d = da.TD

    # ws vitesse à 10m
    ws = da.FFM
    # conversion de la vitesse à 10m d'altitue à une vitesse à 2m d'altitude
    z = 10
    ws2m = ws * (4.87 / np.log((67.8 * z) - 5.42))

    # Saturation vapour pressure [kPa]. Can be estimated using
    # svp=svp_from_t(t)
    svp = 0.6108 * np.exp((17.27 * t) / (t + 237.3))

    # actual vapour pressure [kPa]. Can be estimated using
    # avp=avp_from_tdew(t2d)
    avp = 0.6108 * np.exp(((17.27 * t2d) / (t2d + 237.3)))

    #  my_delta_svp=delta_svp(t)
    my_delta_svp = 4098 * (0.6108 * np.exp((17.27 * t) / (t + 237.3))) / np.power((t + 237.3), 2)
    alt = 280
    atmos_pres = np.power(((293.0 - (0.0065 * alt)) / 293.0), 5.26) * 101.3
    # atmos_pres=atm_pressure(alt)
    # Set attributes
    psy = 0.000665 * atmos_pres
    a1 = (0.408 * (net_rad - shf) * my_delta_svp /
          (my_delta_svp + (psy * (1 + 0.34 * ws2m))))
    a2 = (900 * ws2m / (t + 273.15) * (svp - avp) * psy /
          (my_delta_svp + (psy * (1 + 0.34 * ws2m))))
    eto = a1 + a2

    eto.attrs = da.T2M.attrs

    return eto


def hargreaves(da):
    """
    Estimate reference evapotranspiration over grass (ETo) using the Hargreaves
    equation.

    Generally, when solar radiation data, relative humidity data
    and/or wind speed data are missing, it is better to estimate them using
    the functions available in this module, and then calculate ETo
    the FAO Penman-Monteith equation. However, as an alternative, ETo can be
    estimated using the Hargreaves ETo equation.

    Based on equation 52 in Allen et al (1998).

    :param tmin: Minimum daily temperature [deg C]
    :param tmax: Maximum daily temperature [deg C]
    :param tmean: Mean daily temperature [deg C]. If emasurements not
        available it can be estimated as (*tmin* + *tmax*) / 2.
    :param et_rad: Extraterrestrial radiation (Ra) [MJ m-2 day-1]. Can be
        estimated using ``et_rad()``.
    :return: Reference evapotranspiration over grass (ETo) [mm day-1]
    :rtype: float
    """
    # Note, multiplied by 0.408 to convert extraterrestrial radiation could
    # be given in MJ m-2 day-1 rather than as equivalent evaporation in
    # mm day-1
    tmin = da.TN
    tmax = da.TX
    tmean = da.T2M
    et_rad = da.SSRD
    eto = 0.0023 * (tmean + 17.8) * (tmax - tmin) ** 0.5 * 0.408 * et_rad
    eto.attrs = da.T2M.attrs
    return eto


def inv_rel_dist_earth_sun(day_of_year):
    """
    Calculate the inverse relative distance between earth and sun from
    day of the year.

    Based on FAO equation 23 in Allen et al (1998).

    :param day_of_year: Day of the year [1 to 366]
    :return: Inverse relative distance between earth and the sun
    :rtype: float
    """
    _check_doy(day_of_year)
    return 1 + (0.033 * math.cos((2.0 * math.pi / 365.0) * day_of_year))


def mean_svp(tmin, tmax):
    """
    Estimate mean saturation vapour pressure, *es* [kPa] from minimum and
    maximum temperature.

    Based on equations 11 and 12 in Allen et al (1998).

    Mean saturation vapour pressure is calculated as the mean of the
    saturation vapour pressure at tmax (maximum temperature) and tmin
    (minimum temperature).

    :param tmin: Minimum temperature [deg C]
    :param tmax: Maximum temperature [deg C]
    :return: Mean saturation vapour pressure (*es*) [kPa]
    :rtype: float
    """
    return (svp_from_t(tmin) + svp_from_t(tmax)) / 2.0


def monthly_soil_heat_flux(t_month_prev, t_month_next):
    """
    Estimate monthly soil heat flux (Gmonth) from the mean air temperature of
    the previous and next month, assuming a grass crop.

    Based on equation 43 in Allen et al (1998). If the air temperature of the
    next month is not known use ``monthly_soil_heat_flux2()`` instead. The
    resulting heat flux can be converted to equivalent evaporation [mm day-1]
    using ``energy2evap()``.

    :param t_month_prev: Mean air temperature of the previous month
        [deg Celsius]
    :param t_month2_next: Mean air temperature of the next month [deg Celsius]
    :return: Monthly soil heat flux (Gmonth) [MJ m-2 day-1]
    :rtype: float
    """
    return 0.07 * (t_month_next - t_month_prev)


def monthly_soil_heat_flux2(t_month_prev, t_month_cur):
    """
    Estimate monthly soil heat flux (Gmonth) [MJ m-2 day-1] from the mean
    air temperature of the previous and current month, assuming a grass crop.

    Based on equation 44 in Allen et al (1998). If the air temperature of the
    next month is available, use ``monthly_soil_heat_flux()`` instead. The
    resulting heat flux can be converted to equivalent evaporation [mm day-1]
    using ``energy2evap()``.

    Arguments:
    :param t_month_prev: Mean air temperature of the previous month
        [deg Celsius]
    :param t_month_cur: Mean air temperature of the current month [deg Celsius]
    :return: Monthly soil heat flux (Gmonth) [MJ m-2 day-1]
    :rtype: float
    """
    return 0.14 * (t_month_cur - t_month_prev)


def net_in_sol_rad(sol_rad, albedo=0.23):
    """
    Calculate net incoming solar (or shortwave) radiation from gross
    incoming solar radiation, assuming a grass reference crop.

    Net incoming solar radiation is the net shortwave radiation resulting
    from the balance between incoming and reflected solar radiation. The
    output can be converted to equivalent evaporation [mm day-1] using
    ``energy2evap()``.

    Based on FAO equation 38 in Allen et al (1998).

    :param sol_rad: Gross incoming solar radiation [MJ m-2 day-1]. If
        necessary this can be estimated using functions whose name
        begins with 'sol_rad_from'.
    :param albedo: Albedo of the crop as the proportion of gross incoming solar
        radiation that is reflected by the surface. Default value is 0.23,
        which is the value used by the FAO for a short grass reference crop.
        Albedo can be as high as 0.95 for freshly fallen snow and as low as
        0.05 for wet bare soil. A green vegetation over has an albedo of
        about 0.20-0.25 (Allen et al, 1998).
    :return: Net incoming solar (or shortwave) radiation [MJ m-2 day-1].
    :rtype: float
    """
    return (1 - albedo) * sol_rad


def net_out_lw_rad(tmin, tmax, sol_rad, cs_rad, avp):
    """
    Estimate net outgoing longwave radiation.

    This is the net longwave energy (net energy flux) leaving the
    earth's surface. It is proportional to the absolute temperature of
    the surface raised to the fourth power according to the Stefan-Boltzmann
    law. However, water vapour, clouds, carbon dioxide and dust are absorbers
    and emitters of longwave radiation. This function corrects the Stefan-
    Boltzmann law for humidity (using actual vapor pressure) and cloudiness
    (using solar radiation and clear sky radiation). The concentrations of all
    other absorbers are assumed to be constant.

    The output can be converted to equivalent evaporation [mm day-1] using
    ``energy2evap()``.

    Based on FAO equation 39 in Allen et al (1998).

    :param tmin: Absolute daily minimum temperature [degrees Kelvin]
    :param tmax: Absolute daily maximum temperature [degrees Kelvin]
    :param sol_rad: Solar radiation [MJ m-2 day-1]. If necessary this can be
        estimated using ``sol+rad()``.
    :param cs_rad: Clear sky radiation [MJ m-2 day-1]. Can be estimated using
        ``cs_rad()``.
    :param avp: Actual vapour pressure [kPa]. Can be estimated using functions
        with names beginning with 'avp_from'.
    :return: Net outgoing longwave radiation [MJ m-2 day-1]
    :rtype: float
    """
    tmp1 = (STEFAN_BOLTZMANN_CONSTANT *
            ((math.pow(tmax, 4) + math.pow(tmin, 4)) / 2))
    tmp2 = (0.34 - (0.14 * math.sqrt(avp)))
    tmp3 = 1.35 * (sol_rad / cs_rad) - 0.35
    return tmp1 * tmp2 * tmp3


def net_rad(ni_sw_rad, no_lw_rad):
    """
    Calculate daily net radiation at the crop surface, assuming a grass
    reference crop.

    Net radiation is the difference between the incoming net shortwave (or
    solar) radiation and the outgoing net longwave radiation. Output can be
    converted to equivalent evaporation [mm day-1] using ``energy2evap()``.

    Based on equation 40 in Allen et al (1998).

    :param ni_sw_rad: Net incoming shortwave radiation [MJ m-2 day-1]. Can be
        estimated using ``net_in_sol_rad()``.
    :param no_lw_rad: Net outgoing longwave radiation [MJ m-2 day-1]. Can be
        estimated using ``net_out_lw_rad()``.
    :return: Daily net radiation [MJ m-2 day-1].
    :rtype: float
    """
    return ni_sw_rad - no_lw_rad


def psy_const(da):
    """
    Calculate the psychrometric constant.

    This method assumes that the air is saturated with water vapour at the
    minimum daily temperature. This assumption may not hold in arid areas.

    Based on equation 8, page 95 in Allen et al (1998).

    :param atmos_pres: Atmospheric pressure [kPa]. Can be estimated using
        ``atm_pressure()``.
    :return: Psychrometric constant [kPa degC-1].
    :rtype: float
    """
    atmos_pres = da.ATMOS_PRES
    psy = 0.000665 * atmos_pres
    psy.attrs = da.ATMOS_PRES
    return psy


def psy_const_of_psychrometer(psychrometer, atmos_pres):
    """
    Calculate the psychrometric constant for different types of
    psychrometer at a given atmospheric pressure.

    Based on FAO equation 16 in Allen et al (1998).

    :param psychrometer: Integer between 1 and 3 which denotes type of
        psychrometer:
        1. ventilated (Asmann or aspirated type) psychrometer with
           an air movement of approximately 5 m/s
        2. natural ventilated psychrometer with an air movement
           of approximately 1 m/s
        3. non ventilated psychrometer installed indoors
    :param atmos_pres: Atmospheric pressure [kPa]. Can be estimated using
        ``atm_pressure()``.
    :return: Psychrometric constant [kPa degC-1].
    :rtype: float
    """
    # Select coefficient based on type of ventilation of the wet bulb
    if psychrometer == 1:
        psy_coeff = 0.000662
    elif psychrometer == 2:
        psy_coeff = 0.000800
    elif psychrometer == 3:
        psy_coeff = 0.001200
    else:
        raise ValueError(
            'psychrometer should be in range 1 to 3: {0!r}'.format(psychrometer))

    return psy_coeff * atmos_pres


def rh_from_avp_svp(avp, svp):
    """
    Calculate relative humidity as the ratio of actual vapour pressure
    to saturation vapour pressure at the same temperature.

    See Allen et al (1998), page 67 for details.

    :param avp: Actual vapour pressure [units do not matter so long as they
        are the same as for *svp*]. Can be estimated using functions whose
        name begins with 'avp_from'.
    :param svp: Saturated vapour pressure [units do not matter so long as they
        are the same as for *avp*]. Can be estimated using ``svp_from_t()``.
    :return: Relative humidity [%].
    :rtype: float
    """
    return 100.0 * avp / svp


def sol_dec(day_of_year):
    """
    Calculate solar declination from day of the year.

    Based on FAO equation 24 in Allen et al (1998).

    :param day_of_year: Day of year integer between 1 and 365 or 366).
    :return: solar declination [radians]
    :rtype: float
    """
    _check_doy(day_of_year)
    return 0.409 * math.sin(((2.0 * math.pi / 365.0) * day_of_year - 1.39))


def sol_rad_from_sun_hours(daylight_hours, sunshine_hours, et_rad):
    """
    Calculate incoming solar (or shortwave) radiation, *Rs* (radiation hitting
    a horizontal plane after scattering by the atmosphere) from relative
    sunshine duration.

    If measured radiation data are not available this method is preferable
    to calculating solar radiation from temperature. If a monthly mean is
    required then divide the monthly number of sunshine hours by number of
    days in the month and ensure that *et_rad* and *daylight_hours* was
    calculated using the day of the year that corresponds to the middle of
    the month.

    Based on equations 34 and 35 in Allen et al (1998).

    :param dl_hours: Number of daylight hours [hours]. Can be calculated
        using ``daylight_hours()``.
    :param sunshine_hours: Sunshine duration [hours].
    :param et_rad: Extraterrestrial radiation [MJ m-2 day-1]. Can be
        estimated using ``et_rad()``.
    :return: Incoming solar (or shortwave) radiation [MJ m-2 day-1]
    :rtype: float
    """
    _check_day_hours(sunshine_hours, 'sun_hours')
    _check_day_hours(daylight_hours, 'daylight_hours')

    # 0.5 and 0.25 are default values of regression constants (Angstrom values)
    # recommended by FAO when calibrated values are unavailable.
    return (0.5 * sunshine_hours / daylight_hours + 0.25) * et_rad


def sol_rad_from_t(et_rad, cs_rad, tmin, tmax, coastal):
    """
    Estimate incoming solar (or shortwave) radiation, *Rs*, (radiation hitting
    a horizontal plane after scattering by the atmosphere) from min and max
    temperature together with an empirical adjustment coefficient for
    'interior' and 'coastal' regions.

    The formula is based on equation 50 in Allen et al (1998) which is the
    Hargreaves radiation formula (Hargreaves and Samani, 1982, 1985). This
    method should be used only when solar radiation or sunshine hours data are
    not available. It is only recommended for locations where it is not
    possible to use radiation data from a regional station (either because
    climate conditions are heterogeneous or data are lacking).

    **NOTE**: this method is not suitable for island locations due to the
    moderating effects of the surrounding water.

    :param et_rad: Extraterrestrial radiation [MJ m-2 day-1]. Can be
        estimated using ``et_rad()``.
    :param cs_rad: Clear sky radiation [MJ m-2 day-1]. Can be estimated
        using ``cs_rad()``.
    :param tmin: Daily minimum temperature [deg C].
    :param tmax: Daily maximum temperature [deg C].
    :param coastal: ``True`` if site is a coastal location, situated on or
        adjacent to coast of a large land mass and where air masses are
        influenced by a nearby water body, ``False`` if interior location
        where land mass dominates and air masses are not strongly influenced
        by a large water body.
    :return: Incoming solar (or shortwave) radiation (Rs) [MJ m-2 day-1].
    :rtype: float
    """
    # Determine value of adjustment coefficient [deg C-0.5] for
    # coastal/interior locations
    if coastal:
        adj = 0.19
    else:
        adj = 0.16

    sol_rad = adj * math.sqrt(tmax - tmin) * et_rad

    # The solar radiation value is constrained by the clear sky radiation
    return min(sol_rad, cs_rad)


def sol_rad_island(et_rad):
    """
    Estimate incoming solar (or shortwave) radiation, *Rs* (radiation hitting
    a horizontal plane after scattering by the atmosphere) for an island
    location.

    An island is defined as a land mass with width perpendicular to the
    coastline <= 20 km. Use this method only if radiation data from
    elsewhere on the island is not available.

    **NOTE**: This method is only applicable for low altitudes (0-100 m)
    and monthly calculations.

    Based on FAO equation 51 in Allen et al (1998).

    :param et_rad: Extraterrestrial radiation [MJ m-2 day-1]. Can be
        estimated using ``et_rad()``.
    :return: Incoming solar (or shortwave) radiation [MJ m-2 day-1].
    :rtype: float
    """
    return (0.7 * et_rad) - 4.0


def sunset_hour_angle(latitude, sol_dec):
    """
    Calculate sunset hour angle (*Ws*) from latitude and solar
    declination.

    Based on FAO equation 25 in Allen et al (1998).

    :param latitude: Latitude [radians]. Note: *latitude* should be negative
        if it in the southern hemisphere, positive if in the northern
        hemisphere.
    :param sol_dec: Solar declination [radians]. Can be calculated using
        ``sol_dec()``.
    :return: Sunset hour angle [radians].
    :rtype: float
    """
    _check_latitude_rad(latitude)
    _check_sol_dec_rad(sol_dec)

    cos_sha = -math.tan(latitude) * math.tan(sol_dec)
    # If tmp is >= 1 there is no sunset, i.e. 24 hours of daylight
    # If tmp is <= 1 there is no sunrise, i.e. 24 hours of darkness
    # See http://www.itacanet.org/the-sun-as-a-source-of-energy/
    # part-3-calculating-solar-angles/
    # Domain of acos is -1 <= x <= 1 radians (this is not mentioned in FAO-56!)
    return math.acos(min(max(cos_sha, -1.0), 1.0))


def svp_from_t(t):
    """
    Estimate saturation vapour pressure (*es*) from air temperature.

    Based on equations 11 and 12 in Allen et al (1998).

    :param t: Temperature [deg C]
    :return: Saturation vapour pressure [kPa]
    :rtype: float
    """
    return 0.6108 * math.exp((17.27 * t) / (t + 237.3))


def wind_speed_2m(ws, z):
    """
    Convert wind speed measured at different heights above the soil
    surface to wind speed at 2 m above the surface, assuming a short grass
    surface.

    Based on FAO equation 47 in Allen et al (1998).

    :param ws: Measured wind speed [m s-1]
    :param z: Height of wind measurement above ground surface [m]
    :return: Wind speed at 2 m above the surface [m s-1]
    :rtype: float
    """
    return ws * (4.87 / math.log((67.8 * z) - 5.42))
