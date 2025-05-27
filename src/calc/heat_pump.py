"""
Module to calculate heat pump COPs and maximum available heat power.

Authors:
- Lennart Trentmann (lennart.trentmann@tum.de)
- Amedeo Ceruti (amedeo.ceruti@tum.de)
"""

import warnings
import pandas as pd
import numpy as np
import oemof.thermal.compression_heatpumps_and_chillers as hp
from atlite.convert import convert_coefficient_of_performance


def cop_decentral(delta_temp, source):
    """
    Calculate COP of small-scale heat pumps based on empirical regression models.

    Source:
    "Time series of heat demand and heat pump efficiency for energy system modeling"
    https://doi.org/10.1038/s41597-019-0199-y

    Args:
        delta_temp (pd.DataFrame or pd.Series): Temperature lift (sink - source) in Kelvin.
        source (str): Heat source type: 'air', 'ground', 'soil', or 'water'.

    Returns:
        np.ndarray: COP time series.
    """
    if source == 'air':
        cop = 6.08 - 0.09 * delta_temp + 0.0005 * delta_temp**2  # ASHP
    elif source in ('ground', 'soil'):
        cop = 10.29 - 0.21 * delta_temp + 0.0012 * delta_temp**2  # GSHP
    elif source == 'water':
        cop = 9.97 - 0.20 * delta_temp + 0.0012 * delta_temp**2   # WSHP
    else:
        raise ValueError(f"Unsupported heat source type: {source}")
    
    return cop


def max_heat_power(Q_source, cop):
    """
    Compute the maximum thermal output power of a heat pump.

    Based on the equation:
        COP = Q_out / (Q_out - Q_in)
        → Q_out = Q_in * COP / (COP - 1)

    Args:
        Q_source (float or np.ndarray): Available heat input in kW.
        cop (float or np.ndarray): Coefficient of performance.

    Returns:
        float or np.ndarray: Maximum thermal output power in kW.
    """
    return Q_source * cop / (cop - 1)

def max_elec_power(Q_source, cop):
    """
    Compute the maximum elec input power of a heat pump.

    Based on the equation:
        COP = Q_out / (Q_out - Q_in)
        → Q_out = Q_in * COP / (COP - 1)

    Args:
        Q_source (float or np.ndarray): Available heat input in kW.
        cop (float or np.ndarray): Coefficient of performance.

    Returns:
        float or np.ndarray: Maximum thermal output power in kW.
    """
    return cop / (cop - 1) / cop * Q_source


def classify_hp(T_sink_out, parameters):
    """
    Classify heat pump type based on sink outlet temperature.

    Args:
        T_sink_out (float): Sink outlet temperature in °C.
        parameters (pd.DataFrame): DataFrame of HP types and their operational temperature ranges.

    Returns:
        str or None: Identified heat pump type or None if no match is found.
    """
    for hp_type, values in parameters.iterrows():
        if values['T_sink_out_low'] <= T_sink_out <= values['T_sink_out_high']:
            return hp_type
    return None


def conventional(Delta_T, T_sink_out, parameters):
    """
    COP model for conventional heat pumps based on Jesper et al.

    Args:
        Delta_T (float or np.ndarray): Temperature lift in °C.
        T_sink_out (float or np.ndarray): Sink outlet temperature in °C.
        parameters (dict): Model parameters.

    Returns:
        float or np.ndarray: COP values.
    """
    term1 = parameters["a"] * (Delta_T + 2 * parameters["b"]) ** parameters["c"]
    term2 = ((T_sink_out + 273.15) + parameters["b"]) ** parameters["d"]
    return term1 * term2


def very_high(Delta_T, T_sink_out, parameters):
    """
    COP model for very high-temperature heat pumps.

    Args:
        Delta_T (float or np.ndarray): Temperature lift in °C.
        T_sink_out (float or np.ndarray): Sink outlet temperature in °C.
        parameters (dict): Model parameters.

    Returns:
        float or np.ndarray: COP values.
    """
    term1 = parameters["a"] * (Delta_T + 2 * parameters["d"]) ** parameters["b"]
    term2 = ((T_sink_out + 273.15) + parameters["d"]) ** parameters["c"]
    return term1 * term2


def cop_jesper(df_sink, temp_source, df_model):
    """
    Calculate COP using Jesper et al.'s model based on DHN supply temp and source temp.

    Args:
        df_sink (pd.DataFrame): Contains the column 'T_supply' for sink temperatures (in °C).
        temp_source (float or pd.Series): Source temperature in °C (can be time series).
        df_model (pd.DataFrame): Parameter set indexed by HP type.

    Returns:
        pd.DataFrame: Updated df_sink with columns 'COP', 'Delta T', and 'T_source'.
    """
    T_sink_out = df_sink["T_supply"]

    # Support time series source temperature
    if isinstance(temp_source, (pd.Series, np.ndarray)):
        temp_source_series = temp_source
    else:
        # Force conversion to float
        temp_source_val = float(temp_source)
        temp_source_series = pd.Series(temp_source_val, index=T_sink_out.index)

    Delta_T_lift = abs(temp_source_series - T_sink_out)

    # Classify the HP type based on the max sink temp (still static)
    hp_type = classify_hp(T_sink_out.max(), df_model)
    if hp_type is None:
        raise ValueError("Source and sink temperatures do not match any defined HP type.")

    parameters = df_model.loc[hp_type]

    if hp_type == "conventional":
        cop = conventional(Delta_T_lift, T_sink_out + 273.15, parameters)
        outside = (Delta_T_lift < 10) | (Delta_T_lift > 78)
        if outside.any():
            warnings.warn("Temperature lift out of range for 'conventional' model.")
    elif hp_type == "very high temperature":
        cop = very_high(Delta_T_lift, T_sink_out + 273.15, parameters)
        outside = (Delta_T_lift < 25) | (Delta_T_lift > 95)
        if outside.any():
            warnings.warn("Temperature lift out of range for 'very high temperature' model.")
    else:
        raise ValueError(f"Unknown heat pump type: {hp_type}")

    df_sink["COP"] = cop
    df_sink["Delta T"] = Delta_T_lift
    df_sink["T_source"] = temp_source_series
    df_sink.rename(columns={"T_supply": "T_sink"}, inplace=True)

    return df_sink
    