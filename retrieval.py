import time
import os
from pathlib import Path
import multiprocessing as mp

from astropy.io import fits
from netCDF4 import Dataset
import numpy as np
import psycopg
from scipy.constants import Boltzmann
from scipy.integrate import quadrature as quad

import disort
import pyrt

##############
# IUVS data
##############

# Load in all the info from the given file
orbit: int = 3751
file: int = 0

with psycopg.connect(host='localhost', dbname='iuvs', user='kyle',
                     password='iuvs') as connection:
    # I still need Mars year, Sol
    # Open a cursor for db operations
    with connection.cursor() as cursor:
        cursor.execute(f"select sol from apoapse where orbit = {orbit}")
        sol = cursor.fetchall()[0][0]

# Get the l1b file
files = sorted(Path(f'/media/kyle/Samsung_T5/IUVS_data/orbit0{3700}').glob(f'*apoapse*{orbit}*muv*.gz'))
hdul = fits.open(files[file])

# Load in the reflectance and the wavelengths
# TODO: reflectance
# TODO: wavelengths properly
wavelengths = hdul['observation'].data['wavelength'][0, 0] / 1000  # convert to microns

pixelgeometry = hdul['pixelgeometry'].data
latitude = pixelgeometry['pixel_corner_lat']
longitude = pixelgeometry['pixel_corner_lon']
local_time = pixelgeometry['pixel_local_time']
solar_zenith_angle = pixelgeometry['pixel_solar_zenith_angle']
emission_angle = pixelgeometry['pixel_emission_angle']
phase_angle = pixelgeometry['pixel_phase_angle']
data_ls = hdul['observation'].data['solar_longitude']
solar_zenith_angle = np.where(solar_zenith_angle <= 90, solar_zenith_angle, 0)

# Get the corresponding l1c file
ref = np.load(f'/home/kyle/ql_testing/reflectance{orbit}-{file}.npy')
print(ref.shape)

azimuth = pyrt.azimuth(solar_zenith_angle, emission_angle, phase_angle)
print(azimuth.shape)

##############
# Ames GCM
##############
# Read in the Ames GCM
grid = Dataset('/home/kyle/ames/sim1/10000.fixed.nc')
gcm = Dataset('/home/kyle/ames/sim1/c48_big.atmos_diurn_plev-002.nc')
yearly_gcm = Dataset('/home/kyle/ames/sim1/c48_big.atmos_average_plev-001.nc')
#print(gcm.variables)

gcm_lat = grid.variables['lat'][:]
gcm_lon = grid.variables['lon'][:]

gcm_dust_vprof = yearly_gcm.variables['dustref']   # shape: (year, pressure level, lat, lon)
gcm_ice_vprof = yearly_gcm.variables['cldref']    # shape: (year, pressure level, lat, lon)

gcm_pressure_levels = gcm.variables['pstd']  # shape: (pressure level,)
gcm_surface_pressure = gcm.variables['ps']   # shape: (season, time of day, lat, lon)
gcm_surface_temperature = gcm.variables['ts']   # shape: (season, time of day, lat, lon)
gcm_temperature = gcm.variables['temp']  # shape: (season, time of day, pressure level, lat, lon)
gcm_season = gcm.variables['time'][:]   # idk why the times start from 10,000
gcm_local_time = gcm.variables['time_of_day_24'][:]

season_idx = np.argmin(np.abs(gcm_season - 10000 - sol))
yearly_idx = np.argmin(np.abs(yearly_gcm.variables['time'][:] - 10000 - sol))

##############
# Aerosol radprop
##############
# Dust
'''dust_cext = np.load('/home/kyle/repos/paper3/radprop/mars_dust/extinction_cross_section.npy')  # (24, 317)
dust_csca = np.load('/home/kyle/repos/paper3/radprop/mars_dust/scattering_cross_section.npy')  # (24, 317)
dust_pmom = np.load('/home/kyle/repos/paper3/radprop/mars_dust/legendre_coefficients.npy')  # (129, 24, 317)
dust_wavelengths = np.load('/home/kyle/repos/paper3/radprop/mars_dust/wavelengths.npy')
dust_particle_sizes = np.load('/home/kyle/repos/paper3/radprop/mars_dust/particle_sizes.npy')

# Ice. TODO: use better properties
ice_cext = np.load('/home/kyle/repos/paper3/radprop/mars_ice/extinction_cross_section.npy')  # (24, 317)
ice_csca = np.load('/home/kyle/repos/paper3/radprop/mars_ice/scattering_cross_section.npy')  # (24, 317)
#ice_pmom = np.load('/home/kyle/repos/paper3/radprop/mars_dust/legendre_coefficients.npy')  # (129, 24, 317)
ice_g = np.load('/home/kyle/repos/paper3/radprop/mars_ice/asymmetry_parameter.npy')
ice_wavelengths = np.load('/home/kyle/repos/paper3/radprop/mars_ice/wavelengths.npy')
ice_particle_sizes = np.load('/home/kyle/repos/paper3/radprop/mars_ice/particle_sizes.npy')
ice_pmom = pyrt.decompose_hg(ice_g, 129)'''


def column_density(pressure_profile, temperature_profile, altitude, profargs, tempargs):
    """Make the column density.

    Parameters
    ----------
    pressure_profile
    temperature_profile
    altitude
    profargs
    tempargs

    Returns
    -------

    """
    def hydrostatic_profile(alts):
        return pressure_profile(alts, *profargs) / temperature_profile(alts, *tempargs) / Boltzmann

    n = [quad(hydrostatic_profile, altitude[i + 1], altitude[i])[0] for i in range(len(altitude) - 1)]
    return np.array(n) * 1000


def linear_grid_profile(altitude, altitude_grid, profile_grid) -> np.ndarray:
    """Make a profile with linear interpolation between grid points.

    Parameters
    ----------
    altitude: ArrayLike
        The altitudes at which to make the vertical profile. Must be
        1-dimensional and monotonically decreasing.
    altitude_grid: ArrayLike
        The altitudes where ``profile_grid`` is defined. Must be 1-dimensional
        and monotonically decreasing.
    profile_grid: ArrayLike
        The profile at each point in ``altitude_grid``.

    Returns
    -------
    np.ndarray
        1-dimensional array of the profile with shape ``altitude.shape``.

    Raises
    ------
    TypeError
        Raised if any of the inputs cannot be cast to ndarrays.
    ValueError
        Raised if any of the inputs do not have the aforementioned desired
        properties.
    """
    return np.interp(altitude, np.flip(altitude_grid), profile_grid)


##############
# Lat/lon grid
##############
# Use NN to get a 1D temperature profile


def retrieval(integration: int, spatial_bin: int):
    ##############
    # Equation of state
    ##############
    # Get the nearest neighbor values of the lat/lon/lt values (for speed I'm not using linear interpolation)
    pixel_lat = latitude[integration, spatial_bin, 4]
    pixel_lon = longitude[integration, spatial_bin, 4]
    lat_idx = np.argmin(np.abs(gcm_lat - pixel_lat))
    lon_idx = np.argmin(np.abs(gcm_lon - pixel_lon))
    lt_idx = np.argmin(np.abs(gcm_local_time - (local_time[integration, spatial_bin] - (longitude[integration, spatial_bin, 4] / 360 * 24)) % 24))

    # Get 1D profiles of temperature and pressure for this pixel (well, pressures are just fixed by the model to be the same everywhere)
    pixel_temperature = gcm_temperature[season_idx, lt_idx, :, lat_idx, lon_idx]

    # Get where the "bad" values are
    bads = np.sum(pixel_temperature.mask)   # The number of missing points
    first_bad = np.argmin(~pixel_temperature.mask)

    # Insert the surface values into the temperature and pressure arrays so they're shape (41,) instead of (40,)
    sfc_pressure = gcm_surface_pressure[season_idx, lt_idx, lat_idx, lon_idx]
    pixel_temperature = np.insert(pixel_temperature, first_bad, gcm_surface_temperature[season_idx, lt_idx, lat_idx, lon_idx])
    pixel_pressure = np.insert(gcm_pressure_levels, first_bad, sfc_pressure)

    # Make the altitudes assuming exponentially decreasing pressure and constant scale height
    z = -np.log(pixel_pressure / sfc_pressure) * 10

    # Finally, use these to compute the column density in each "good" layer
    colden = column_density(linear_grid_profile, linear_grid_profile, z, (z, pixel_pressure), (z, pixel_temperature))[:-bads]

    ##############
    # Rayleigh scattering
    ##############
    rayleigh_scattering_optical_depth = pyrt.rayleigh_co2_optical_depth(colden, wavelengths)
    rayleigh_ssa = np.ones((rayleigh_scattering_optical_depth.shape))
    rayleigh_pmom = pyrt.rayleigh_legendre(rayleigh_scattering_optical_depth.shape[0], wavelengths.shape[0])

    ##############
    # Aerosol vertical profiles
    ##############
    dust_vprof = gcm_dust_vprof[yearly_idx, :, lat_idx, lon_idx]
    ice_vprof = gcm_ice_vprof[yearly_idx, :, lat_idx, lon_idx]



retrieval(100, 50)

