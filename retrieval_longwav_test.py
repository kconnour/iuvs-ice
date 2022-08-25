import time
import os
from pathlib import Path
import multiprocessing as mp
import math
from tempfile import mkdtemp

from astropy.io import fits
from netCDF4 import Dataset
import numpy as np
import psycopg
from scipy.constants import Boltzmann
from scipy.integrate import quadrature as quad
from scipy.optimize import minimize
from scipy.io import readsav
from scipy.interpolate import interpn

import disort
import pyrt

lamber = True

#print(disort.disort.__doc__)
#print(disort.disobrdf.__doc__)
#raise SystemExit(9)

##############
# IUVS data
##############

# Load in all the info from the given file
orbit: int = 3467
file: int = 8
block = math.floor(orbit / 100) * 100

# Connect to the DB to get the time of year of the orbit
with psycopg.connect(host='localhost', dbname='iuvs', user='kyle', password='iuvs') as connection:
    with connection.cursor() as cursor:
        cursor.execute(f"select sol from apoapse where orbit = {orbit}")
        sol = cursor.fetchall()[0][0]

# Get the l1b file
l1b_files = sorted(Path(f'/media/kyle/McDataFace/iuvsdata/production/orbit0{block}').glob(f'*apoapse*{orbit}*muv*.gz'))
hdul = fits.open(l1b_files[file])

# Load in the reflectance
ff = np.load('/home/kyle/repos/PyUVS/pyuvs/anc/flatfields/mid-hi-res-flatfield-update.npy')
reflectance_files = sorted(Path(f'/home/kyle/iuvs/reflectance/orbit0{block}').glob(f'reflectance{orbit}*nonlinear-solstice*'))
reflectance = np.load(reflectance_files[file]) / ff * 1/1.18

# Load in the corrected wavelengths
wavelengths_files = sorted(Path(f'/home/kyle/iuvs/wavelengths/orbit0{block}/mvn_iuv_wlnonlin_apoapse-orbit0{orbit}-muv').glob(f'*orbit0{orbit}-muv*.sav'))
wavelengths = readsav(wavelengths_files[file])['wavelength_muv'] / 1000  # convert to microns

# Get the data from the l1b file
pixelgeometry = hdul['pixelgeometry'].data
latitude = pixelgeometry['pixel_corner_lat']
longitude = pixelgeometry['pixel_corner_lon']
local_time = pixelgeometry['pixel_local_time']
solar_zenith_angle = pixelgeometry['pixel_solar_zenith_angle']
emission_angle = pixelgeometry['pixel_emission_angle']
phase_angle = pixelgeometry['pixel_phase_angle']
data_ls = hdul['observation'].data['solar_longitude']

# Make the azimuth angles
solar_zenith_angle_good = np.where(solar_zenith_angle <= 90, True, False)
solar_zenith_angle = np.where(solar_zenith_angle <= 90, solar_zenith_angle, 0)
azimuth = pyrt.azimuth(solar_zenith_angle, emission_angle, phase_angle)

mu0 = np.cos(np.radians(solar_zenith_angle))
mu = np.cos(np.radians(emission_angle))

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

gcm_pressure_levels = gcm.variables['pstd']  # shape: (pressure level,). Index 0 is TOA
gcm_surface_pressure = gcm.variables['ps']   # shape: (season, time of day, lat, lon)
gcm_surface_temperature = gcm.variables['ts']   # shape: (season, time of day, lat, lon)
gcm_temperature = gcm.variables['temp']  # shape: (season, time of day, pressure level, lat, lon)
gcm_season = gcm.variables['time'][:]   # idk why the times start from 10,000
gcm_local_time = gcm.variables['time_of_day_24'][:]

season_idx = np.argmin(np.abs(gcm_season - 10000 - sol))
yearly_idx = np.argmin(np.abs(yearly_gcm.variables['time'][:] - 10000 - sol))

##############
# Surface albedo map
##############
band6 = fits.open('/home/kyle/repos/iuvs-ice/map/band6_w.fits')['primary'].data
band7 = fits.open('/home/kyle/repos/iuvs-ice/map/band7_w.fits')['primary'].data
hapke_w = np.dstack((band6, band7))

##############
# Aerosol radprop
##############
# Dust
dust_cext = np.load('/home/kyle/repos/iuvs-ice/radprop/mars_dust/extinction_cross_section.npy')  # (24, 317)
dust_csca = np.load('/home/kyle/repos/iuvs-ice/radprop/mars_dust/scattering_cross_section.npy')  # (24, 317)
dust_pmom = np.load('/home/kyle/repos/iuvs-ice/radprop/mars_dust/legendre_coefficients.npy')  # (129, 24, 317)
dust_wavelengths = np.load('/home/kyle/repos/iuvs-ice/radprop/mars_dust/wavelengths.npy')
dust_particle_sizes = np.load('/home/kyle/repos/iuvs-ice/radprop/mars_dust/particle_sizes.npy')

ice_cext = np.load('/home/kyle/repos/iuvs-ice/radprop/mars_ice/extinction_cross_section.npy')  # (24, 317)
ice_csca = np.load('/home/kyle/repos/iuvs-ice/radprop/mars_ice/scattering_cross_section.npy')  # (24, 317)
ice_pmom = np.load('/home/kyle/repos/iuvs-ice/radprop/mars_ice/legendre_coefficients.npy')  # (96,)   # For 3 micron particles at 321 nm
ice_g = np.load('/home/kyle/repos/iuvs-ice/radprop/mars_ice/asymmetry_parameter.npy')
ice_wavelengths = np.load('/home/kyle/repos/iuvs-ice/radprop/mars_ice/wavelengths.npy')
ice_particle_sizes = np.load('/home/kyle/repos/iuvs-ice/radprop/mars_ice/particle_sizes.npy')
#ice_pmom = pyrt.decompose_hg(ice_g, 129)

##############
# Surface arrays
##############
n_streams = 16
n_polar = 1    # defined by IUVS' viewing geometry
n_azimuth = 1    # defined by IUVS' viewing geometry

bemst = np.zeros(int(0.5*n_streams))
emust = np.zeros(n_polar)
rho_accurate = np.zeros((n_polar, n_azimuth))
rhoq = np.zeros((int(0.5 * n_streams), int(0.5 * n_streams + 1), n_streams))
rhou = np.zeros((n_polar, int(0.5 * n_streams + 1), n_streams))


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

    n = [quad(hydrostatic_profile, altitude[i+1], altitude[i])[0] for i in range(len(altitude) - 1)]
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
    # NOTE: numpy needs linearly increasing arrays, which is opposite of the altitude convention! They don't warn about this
    return np.interp(np.flip(altitude), np.flip(altitude_grid), np.flip(profile_grid))


def retrieval(integration: int, spatial_bin: int):
    # Exit if the angles are not retrievable
    if (not solar_zenith_angle_good[integration, spatial_bin]) or \
            solar_zenith_angle[integration, spatial_bin] >= 72 or \
            emission_angle[integration, spatial_bin] >= 72:
        answer = np.zeros((2,)) * np.nan
        return integration, spatial_bin, answer, np.nan

    pixel_wavs = wavelengths[spatial_bin, :]

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
    first_bad = np.argmin(~pixel_temperature.mask)

    # Insert the surface values into the temperature and pressure arrays so they're shape (41,) instead of (40,)
    sfc_pressure = gcm_surface_pressure[season_idx, lt_idx, lat_idx, lon_idx]
    sfc_temperature = gcm_surface_temperature[season_idx, lt_idx, lat_idx, lon_idx]
    pixel_pressure = np.insert(gcm_pressure_levels, first_bad, sfc_pressure)
    pixel_temperature = np.insert(pixel_temperature, first_bad, sfc_temperature)

    # Make the altitudes assuming exponentially decreasing pressure and constant scale height
    z = -np.log(pixel_pressure / sfc_pressure) * 10

    # Finally, use these to compute the column density in each "good" layer
    colden = column_density(linear_grid_profile, linear_grid_profile, z[:first_bad+1],
                            (z[:first_bad+1], pixel_pressure[:first_bad+1]),
                            (z[:first_bad+1], pixel_temperature[:first_bad+1]))

    ##############
    # Rayleigh scattering
    ##############
    rayleigh_scattering_optical_depth = pyrt.rayleigh_co2_optical_depth(colden, pixel_wavs)
    rayleigh_ssa = np.ones(rayleigh_scattering_optical_depth.shape)
    rayleigh_pmom = pyrt.rayleigh_legendre(rayleigh_scattering_optical_depth.shape[0], pixel_wavs.shape[0])

    #print(np.sum(rayleigh_scattering_optical_depth, axis=0))
    #print(sfc_pressure)
    #raise SystemExit(9)

    rayleigh_column = pyrt.Column(rayleigh_scattering_optical_depth, rayleigh_ssa, rayleigh_pmom)

    ##############
    # Aerosol vertical profiles
    ##############
    # Get the GCM profiles and normalize them
    dust_vprof = gcm_dust_vprof[yearly_idx, :first_bad, lat_idx, lon_idx]
    dust_vprof = dust_vprof / np.sum(dust_vprof)
    ice_vprof = gcm_ice_vprof[yearly_idx, :first_bad, lat_idx, lon_idx]
    ice_vprof = ice_vprof / np.sum(ice_vprof)

    ##############
    # Surface
    ##############
    # Make empty arrays. These are fine to be empty with Lambert surfaces and they'll be filled with more complicated surfaces later on
    bemst_empty = np.zeros(int(0.5 * n_streams))
    emust_empty = np.zeros(n_polar)
    rho_accurate_empty = np.zeros((n_polar, n_azimuth))
    rhoq_empty = np.zeros(
                 (int(0.5 * n_streams), int(0.5 * n_streams + 1), n_streams))
    rhou_empty = np.zeros((n_polar, int(0.5 * n_streams + 1), n_streams))

    # For Lambert
    clancy = np.linspace(0.014, 0.017, num=100)
    clancy_wavs = np.linspace(0.258, 0.32, num=100)
    sfc = np.interp(wavelengths, clancy_wavs, clancy)

    # For Hapke HG2. The idea is the make arrays that are (n_wavelengths, array_shape). This way I can compute them once for each wavelength
    # and each pixel, and then just call them once.
    #wolff_hapke = np.interp(pixel_wavs, np.linspace(0.258, 0.32, num=100), np.linspace(0.06, 0.08, num=100))
    wolff_hapke = np.array([interpn((np.linspace(-90, 90, num=180),
                           np.linspace(0, 360, num=360), np.array([0.26, 0.32])),
                           hapke_w, np.array([pixel_lat, pixel_lon, f]), bounds_error=False,
                           fill_value=None)[0] for f in pixel_wavs])
    bemst = np.zeros((19,) + bemst_empty.shape)
    emust = np.zeros((19,) + emust_empty.shape)
    rho_accurate = np.zeros((19,) + rho_accurate_empty.shape)
    rhoq = np.zeros((19,) + rhoq_empty.shape)
    rhou = np.zeros((19,) + rhou_empty.shape)

    # Make the surface arrays just once for Hapke HG2
    if not lamber:
        for counter, foobar in enumerate(wolff_hapke):
            #brdf_arg = np.array([0.8, 0.06, foobar, 0.3, 0.45, 20])  # From Wolff2010
            brdf_arg = np.array([1, 0.06, foobar, 0.3, 0.7, 20])  # From Wolff2019
            _rhoq, _rhou, _emust, _bemst, _rho_accurate = \
                disort.disobrdf(True, mu[integration, spatial_bin], np.pi, mu0[integration, spatial_bin], False, 0.01, False,
                                rhoq_empty, rhou_empty, emust_empty, bemst_empty, False,
                                azimuth[integration, spatial_bin], 0, rho_accurate_empty,
                                6, brdf_arg, 200, nstr=n_streams, numu=1, nphi=1)
            bemst[counter] = _bemst
            emust[counter] = _emust
            rho_accurate[counter] = _rho_accurate
            rhoq[counter] = _rhoq
            rhou[counter] = _rhou

    wavelength_indices = [1, 2, 3, -4, -3, -2]
    #wavelength_indices = [-5, -4, -3, -2]

    def simulate_tau(guess: np.ndarray) -> np.ndarray:
        #print(f'guess = {guess}')
        dust_guess = guess[0]
        ice_guess = guess[1]

        simulated_toa_reflectance = np.zeros(len(wavelength_indices))

        # This is a hack to add bounds to the solver
        if np.any(guess < 0):
            simulated_toa_reflectance[:] = 999999
            return simulated_toa_reflectance

        for counter, wav_index in enumerate(wavelength_indices):
            ##############
            # Dust FSP
            ##############
            # Before: pixel_wavs[wav_index]. Now I'm scaling it to 240 nm
            dust_extinction = pyrt.extinction_ratio_grid(dust_cext, dust_particle_sizes, dust_wavelengths, pixel_wavs[wav_index])
            dust_reff_grid = np.linspace(1.5, 1.5, num=100)
            dust_z_reff = np.linspace(100, 0, num=100)
            dust_reff = np.interp(np.flip(z[:first_bad]), np.flip(dust_z_reff), np.flip(dust_reff_grid))   # The "bad" altitudes don't get assigned a particle size

            dust_extinction_grid = pyrt.regrid(dust_extinction, dust_particle_sizes, dust_wavelengths, dust_reff, pixel_wavs)

            dust_od = pyrt.optical_depth(dust_vprof, colden, dust_extinction_grid, dust_guess)
            dust_ssa = pyrt.regrid(dust_csca / dust_cext, dust_particle_sizes, dust_wavelengths, dust_reff, pixel_wavs)
            dust_legendre = pyrt.regrid(dust_pmom, dust_particle_sizes, dust_wavelengths, dust_reff, pixel_wavs)
            dust_column = pyrt.Column(dust_od, dust_ssa, dust_legendre)

            ##############
            # Ice FSP
            ##############
            # Before: pixel_wavs[wav_index]. Now I'm scaling it to 240 nm
            ice_extinction = pyrt.extinction_ratio_grid(ice_cext, ice_particle_sizes, ice_wavelengths, pixel_wavs[wav_index])
            ice_reff_grid = np.linspace(3, 3, num=100)
            ice_z_reff = np.linspace(100, 0, num=100)
            ice_reff = np.interp(np.flip(z[:first_bad]), np.flip(ice_z_reff), np.flip(ice_reff_grid))   # The "bad" altitudes don't get assigned a particle size

            ice_extinction_grid = pyrt.regrid(ice_extinction, ice_particle_sizes, ice_wavelengths, ice_reff, pixel_wavs)

            ice_od = pyrt.optical_depth(ice_vprof, colden, ice_extinction_grid, ice_guess)
            ice_ssa = pyrt.regrid(ice_csca / ice_cext, ice_particle_sizes, ice_wavelengths, ice_reff, pixel_wavs)
            #ice_asym = pyrt.regrid(ice_g, ice_particle_sizes, ice_wavelengths, ice_reff, pixel_wavs)
            #ice_legendre = pyrt.decompose_hg(ice_asym, 129)
            # For now I only have a 1D array of phase functions so I need to broadcast
            ice_legendre = np.broadcast_to(ice_pmom, (ice_ssa.shape + ice_pmom.shape))
            ice_legendre = np.moveaxis(ice_legendre, -1, 0)
            ice_column = pyrt.Column(ice_od, ice_ssa, ice_legendre)

            ##############
            # Total atmosphere
            ##############
            atm = rayleigh_column + dust_column + ice_column
            weight = np.arange(129) * 2 + 1
            atm.legendre_coefficients = (atm.legendre_coefficients.T / weight).T

            ##############
            # Output arrays
            ##############
            n_user_levels = atm.optical_depth.shape[0] + 1
            albedo_medium = np.zeros(n_polar)
            diffuse_up_flux = np.zeros(n_user_levels)
            diffuse_down_flux = np.zeros(n_user_levels)
            direct_beam_flux = np.zeros(n_user_levels)
            flux_divergence = np.zeros(n_user_levels)
            intensity = np.zeros((n_polar, n_user_levels, n_azimuth))
            mean_intensity = np.zeros(n_user_levels)
            transmissivity_medium = np.zeros(n_polar)

            # Misc
            user_od_output = np.zeros(n_user_levels)
            temper = np.zeros(n_user_levels)
            h_lyr = np.zeros(n_user_levels)
            ##############
            # Call DISORT
            ##############

            # The 2nd option of the 2nd line is LAMBER
            rfldir, rfldn, flup, dfdt, uavg, uu, albmed, trnmed = \
                disort.disort(True, False, False, False, [False, False, False, False, False],
                              False, lamber, True, False,
                              atm.optical_depth[:, wav_index],
                              atm.single_scattering_albedo[:, wav_index],
                              atm.legendre_coefficients[:, :, wav_index],
                              temper, 1, 1, user_od_output,
                              mu0[integration, spatial_bin], 0,
                              mu[integration, spatial_bin], azimuth[integration, spatial_bin],
                              np.pi, 0, sfc[wav_index], 0, 0, 1, 3400000, h_lyr,
                              rhoq[wav_index], rhou[wav_index], rho_accurate[wav_index], bemst[wav_index], emust[wav_index],
                              0, '', direct_beam_flux,
                              diffuse_down_flux, diffuse_up_flux, flux_divergence,
                              mean_intensity, intensity, albedo_medium,
                              transmissivity_medium, maxcmu=n_streams, maxulv=n_user_levels, maxmom=128)
            simulated_toa_reflectance[counter] = uu[0, 0, 0]
        #print([210, 215, 220, 290, 295, 300])
        #print(reflectance[integration, spatial_bin, wavelength_indices])
        #print(simulated_toa_reflectance)
        '''fig, ax = plt.subplots()
        ax.scatter(pixel_wavs, reflectance[integration, spatial_bin, :], label='data')
        ax.scatter(pixel_wavs[wavelength_indices], simulated_toa_reflectance, label='simulation')
        ax.legend()
        plt.savefig('/home/kyle/iuvs/test.png')
        raise SystemExit(9)'''
        return simulated_toa_reflectance

    def find_best_fit(guess: np.ndarray):
        simulated_toa_reflectance = simulate_tau(guess)
        print(f'{reflectance[integration, spatial_bin, wavelength_indices]} \n'
              f'{simulated_toa_reflectance} \n'
              f'{guess}')
        return np.sum((simulated_toa_reflectance - reflectance[integration, spatial_bin, wavelength_indices])**2)

    fitted_optical_depth = minimize(find_best_fit, np.array([0.7, 0.2]), method='Powell', bounds=((0, 2), (0, 1))).x
    solution = np.array(fitted_optical_depth)
    sim = simulate_tau(solution)
    chi_squared = np.sum((reflectance[integration, spatial_bin, wavelength_indices] - sim)**2 / sim)
    print(f'chisq = {chi_squared}')
    print(f'answer={fitted_optical_depth}')
    return integration, spatial_bin, solution, chi_squared


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Make a shared array
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
memmap_filename_dust = os.path.join(mkdtemp(), 'myNewFileDust.dat')
memmap_filename_ice = os.path.join(mkdtemp(), 'myNewFileIce.dat')
memmap_filename_chisq = os.path.join(mkdtemp(), 'myNewFileChiSquared.dat')
retrieved_dust = np.memmap(memmap_filename_dust, dtype=float,
                           shape=reflectance.shape[:-1], mode='w+')
retrieved_ice = np.memmap(memmap_filename_ice, dtype=float,
                           shape=reflectance.shape[:-1], mode='w+')
retrieved_chi_squared = np.memmap(memmap_filename_chisq, dtype=float,
                           shape=reflectance.shape[:-1], mode='w+')


def make_answer(inp):
    integration = inp[0]
    position = inp[1]
    answer = inp[2]
    chisq = inp[3]
    retrieved_dust[integration, position] = answer[0]
    retrieved_ice[integration, position] = answer[1]
    retrieved_chi_squared[integration, position] = chisq


t0 = time.time()
n_cpus = mp.cpu_count()    # = 8 for my desktop, 12 for my laptop
pool = mp.Pool(n_cpus - 1)   # save one/two just to be safe. Some say it's faster

# NOTE: if there are any issues in the argument of apply_async (here,
# retrieve_ssa), it'll break out of that and move on to the next iteration.
for integ in range(reflectance.shape[0]):
    for posit in range(reflectance.shape[1]):
        pool.apply_async(retrieval, args=(integ, posit), callback=make_answer)
        #retrieval(integ, posit)
# https://www.machinelearningplus.com/python/parallel-processing-python/
pool.close()
pool.join()  # I guess this postpones further code execution until the queue is finished
np.save(f'/home/kyle/iuvs/retrievals/orbit0{block}/orbit{orbit}-{file}-dust.npy', retrieved_dust)
np.save(f'/home/kyle/iuvs/retrievals/orbit0{block}/orbit{orbit}-{file}-ice.npy', retrieved_ice)
np.save(f'/home/kyle/iuvs/retrievals/orbit0{block}/orbit{orbit}-{file}-chi_squared.npy', retrieved_chi_squared)
t1 = time.time()
print(t1-t0)
