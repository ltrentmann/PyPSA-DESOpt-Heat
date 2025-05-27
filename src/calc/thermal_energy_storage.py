"""
Thermal energy storage calculations for the energy system model pre-processing.

Authors:
- Lennart Trentmann (lennart.trentmann@tum.de)
- Amedeo Ceruti (amedeo.ceruti@tum.de)
"""

import pandas as pd
import CoolProp.CoolProp as CP
from oemof.thermal.stratified_thermal_storage import (
    calculate_capacities,
    calculate_storage_dimensions,
    calculate_storage_u_value,
    calculate_losses
)
import math

def standing_loss_TTES(storage_row, temp):
    """
    Calculate the standing heat losses of a single thermal energy storage configuration.

    Parameters
    ----------
    storage_row : pd.Series
        A row from the storage DataFrame containing storage parameters (height, diameter, 
        temp_h, temp_c, insulation thickness, conductivity, heat transfer coefficients).
    
    temp : pd.Series
        Ambient temperature time series (in °C) used for loss calculations.

    Returns
    -------
    pd.DataFrame
        DataFrame containing relative fixed heat losses over time.
    """
    fluid = "Water"
    pressure = 101325  # Pa (atmospheric pressure)
    temperature = (storage_row["temp_h"] + storage_row["temp_c"]) / 2 + 273.15  # Convert to Kelvin

    # Thermophysical properties
    density = CP.PropsSI("D", "P", pressure, "T", temperature, fluid)           # kg/m³
    heat_capacity = CP.PropsSI("C", "P", pressure, "T", temperature, fluid)     # J/(kg·K)

    # Storage geometry
    volume, surface = calculate_storage_dimensions(storage_row["height"], storage_row["diameter"])

    # Energy capacity
    nominal_storage_capacity = calculate_capacities(
        volume,
        storage_row["temp_h"],
        storage_row["temp_c"],
        heat_capacity,
        density
    )

    # Heat transfer coefficient (U-value)
    u_value = calculate_storage_u_value(
        storage_row["s_iso"],
        storage_row["lamb_iso"],
        storage_row["alpha_inside"],
        storage_row["alpha_outside"]
    )

    # Heat losses
    loss_rate, fixed_losses_relative, fixed_losses_absolute = calculate_losses(
        u_value,
        storage_row["diameter"],
        storage_row["temp_h"],
        storage_row["temp_c"],
        temp_env=temp.values,
        time_increment=1,
        heat_capacity=heat_capacity,
        density=density
    )

    return pd.Series(fixed_losses_relative, index=temp.index, name="relative_loss")

def standing_loss_PTES(storage_row, temp):
    """
    Calculate the standing heat losses of a single thermal energy storage configuration.

    Parameters
    ----------
    storage_row : pd.Series
        A row from the storage DataFrame containing storage parameters (height, diameter, 
        temp_h, temp_c, insulation thickness, conductivity, heat transfer coefficients).
    
    temp : pd.Series
        Ambient temperature time series (in °C) used for loss calculations.

    Returns
    -------
    pd.DataFrame
        DataFrame containing relative fixed heat losses over time.
    """
    fluid = "Water"
    pressure = 101325  # Pa (atmospheric pressure)
    temperature = (storage_row["temp_h"] + storage_row["temp_c"]) / 2 + 273.15  # Convert to Kelvin

    # Thermophysical properties
    density = CP.PropsSI("D", "P", pressure, "T", temperature, fluid)           # kg/m³
    heat_capacity = CP.PropsSI("C", "P", pressure, "T", temperature, fluid)     # J/(kg·K)

    # Calculate height from aspect ratio (height = 1 / aspect_ratio)
    h = 1 / storage_row["aspect_ratio"]

    # Calculate diameter from volume formula for a cylinder: V = π * (d/2)^2 * h
    d = math.sqrt((4 * storage_row["volume"]) / (math.pi * h))

    # Surface areas
    A_top = math.pi * (d / 2)**2
    A_side = math.pi * d * h
    A_bottom = A_top
    A_total = A_top + A_side + A_bottom

    # Calculate overall U-value
    u_value = (A_top * storage_row["U_top"] + A_side * storage_row["U_side"] + A_bottom * storage_row["U_bottom"]) / A_total

    # Heat losses
    loss_rate, fixed_losses_relative, fixed_losses_absolute = calculate_losses(
        u_value,
        d,
        storage_row["temp_h"],
        storage_row["temp_c"],
        temp_env=temp.values,
        time_increment=1,
        heat_capacity=heat_capacity,
        density=density
    )

    return pd.Series(fixed_losses_relative, index=temp.index, name="relative_loss")