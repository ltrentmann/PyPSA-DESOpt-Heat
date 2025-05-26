"""Solar calculations for the energy system model pre-processing.

Authors:
- Lennart Trentmann (lennart.trentmann@tum.de)
- Amedeo Ceruti (amedeo.ceruti@tum.de)
"""
import warnings
import pandas as pd
import oemof.thermal.solar_thermal_collector as stc
import oemof.thermal.concentrating_solar_power as concsp
import atlite.convert
import pvlib


def pv(cutout, layout, orientations=None, panel="CSi"):
    """
    Calculates the PV output for a given cutout and layout, based on panel orientation.

    Key assumption: uniform layout, divided equally for each orientation.
    If orientations are None, uses optimal orientation based on latitude.

    Args:
        cutout (atlite.Cutout): Atlite cutout containing weather data.
        layout (atlite.Layout): Atlite layout of spatial points.
        orientations (pd.DataFrame, optional): Orientation specs with 'id', 'slope', and 'azimuth'.
        panel (str): Type of PV panel ('CdTe', 'CSi', 'KANENA').

    Returns:
        pd.DataFrame: PV time series with one column per orientation.
    """
    res = {}  # Dictionary to store results

    # Iterate over provided orientations and compute PV generation
    if orientations is not None:
        for _, v in orientations.iterrows():
            res[v.id] = cutout.pv(
                panel=panel,
                orientation={'slope': v.slope, 'azimuth': v.azimuth},
                layout=layout
            ).squeeze().to_series()
    else:
        # Default to optimal orientation if none provided
        res['latitude_optimal'] = cutout.pv(
            panel=panel,
            orientation='latitude_optimal',
            layout=layout
        ).squeeze().to_series()

    return pd.DataFrame.from_dict(res)


def thermal_collector(df_irradiation, lat, lon, df_collectors):
    """
    Calculates solar thermal output using flat-plate collector model.

    Args:
        df_irradiation (pd.DataFrame): Irradiation time series (GHI, DHI, DNI, temp, etc.).
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.
        df_collectors (pd.DataFrame): Collector parameters: 'id', 'slope', 'azimuth', 
                                      'eta_0', 'a_1', 'a_2', 'T_coll_inlet', 'Delta_T'.

    Returns:
        pd.DataFrame: Solar thermal results (efficiency, heat, irradiance).
    """
    d = {}  # Store results

    for _, v in df_collectors.iterrows():
        res = stc.flat_plate_precalc(
            lat=lat, long=lon,
            collector_tilt=v.slope, collector_azimuth=v.azimuth,
            eta_0=v.eta_0, a_1=v.a_1, a_2=v.a_2,
            temp_collector_inlet=v.T_coll_inlet,
            delta_temp_n=v.Delta_T,
            irradiance_global=df_irradiation['ghi'],
            irradiance_diffuse=df_irradiation['influx_diffuse'],
            temp_amb=df_irradiation['temperature'] - 273.15  # Convert from K to °C
        )
        d[str(v.id) + '-eta_c'] = res['eta_c']
        d[str(v.id) + '-heat'] = res['collectors_heat']
        d[str(v.id) + '-ira'] = res['col_ira']

    return pd.DataFrame.from_dict(d)


def irradiation(cutout, orientation, lat, lon):
    """
    Calculates various irradiation components for a given orientation and location.

    Includes both atlite and pvlib estimates for total, direct, and diffuse irradiation.

    Args:
        cutout (atlite.Cutout): Atlite cutout.
        orientation (dict or str): Panel orientation as dict or 'latitude_optimal'.
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.

    Returns:
        pd.DataFrame: Time series of irradiation components and temperatures.
    """
    res = {}

    # Get data slice for the nearest grid point
    ds = cutout.data.sel({'x': lon, 'y': lat}, method='nearest')

    # Set orientation
    if orientation == 'latitude_optimal':
        _orient = {'slope': 180, 'azimuth': 45}
        warnings.warn(
            "Using optimal orientation for latitude. "
            "This is not equal to the given module orientation."
        )
    else:
        _orient = orientation

    tag = f'slope{_orient["slope"]}-az{_orient["azimuth"]}'

    # Calculate atlite irradiation components
    orient = atlite.convert.get_orientation(_orient)
    res[f"total-atlite_{tag}"] = atlite.convert.convert_irradiation(
        ds, orientation=orient, irradiation="total"
    ).squeeze().to_series()
    res[f"direct-atlite_{tag}"] = atlite.convert.convert_irradiation(
        ds, orientation=orient, irradiation="direct"
    ).squeeze().to_series()
    res[f"diffuse-atlite_{tag}"] = atlite.convert.convert_irradiation(
        ds, orientation=orient, irradiation="diffuse"
    ).squeeze().to_series()

    # Raw radiation components from cutout
    res['influx_direct'] = ds["influx_direct"].squeeze().to_series()
    res['influx_diffuse'] = ds["influx_diffuse"].squeeze().to_series()
    res['influx_toa'] = ds["influx_toa"].squeeze().to_series()
    res['ghi'] = res['influx_direct'] + res['influx_diffuse']  # Global Horizontal Irradiance
    res['temperature'] = ds.temperature.squeeze().to_series()

    # Calculate DNI using atlite
    sp = atlite.convert.SolarPosition(ds)
    res['dni-atlite'] = atlite.convert.cspm.calculate_dni(ds, sp).squeeze().to_series()

    return pd.DataFrame.from_dict(res)