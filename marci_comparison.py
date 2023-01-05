from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from pyuvs.spice import *
from pyuvs import swath_number
import math

# Set the orbit info
orbit = 7300
orbit_code = f'orbit' + f'{orbit}'.zfill(5)
block = math.floor(orbit / 100) * 100
orbit_block = 'orbit' + f'{block}'.zfill(5)

# Compute SPICE info
spice_path = Path('/media/kyle/McDataFace/spice')
s = Spice(spice_path)
s.load_spice()
orbits, all_et = s.find_all_maven_apsis_et('apoapse', endtime=datetime(2022, 9, 4))


print('found apsis info')
et = all_et[orbits == orbit][0]
pf = PositionFinder(et)
dt = pf.get_datetime()
print(dt)

raise SystemExit(9)

# Get the relevant MARCI file
doy = f'{dt.timetuple().tm_yday}'.zfill(3)
marci = fits.open(f'/home/kyle/Downloads/cld_{dt.year}_{doy}_{doy}.fits.gz')
marci_ice = marci['tauice'].data
marci_ice = np.roll(marci_ice, 1440, axis=1)   # shape: (1440, 2880)
marci_lat = np.broadcast_to(np.linspace(-90, 90, num=1441), (2881, 1441))
marci_lon = np.broadcast_to(np.linspace(0, 360, num=2881), (1441, 2881)).T

# Get the IUVS data
dust_files = sorted(Path(f'/home/kyle/iuvs/retrievals/{orbit_block}/data/').glob(f'{orbit_code}-*-dust-radiance-nm.npy'))
ice_files = sorted(Path(f'/home/kyle/iuvs/retrievals/{orbit_block}/data/').glob(f'{orbit_code}-*-ice-radiance-nm.npy'))
chi_squared_files = sorted(Path(f'/home/kyle/iuvs/retrievals/{orbit_block}/data/').glob(f'{orbit_code}-*-chi_squared-radiance-nm.npy'))
dust = np.vstack([np.load(f) for f in dust_files])
ice = np.vstack([np.load(f) for f in ice_files])
chi_squared = np.vstack([np.load(f) for f in chi_squared_files])

files = sorted(Path(f'/media/kyle/McDataFace/iuvsdata/production/{orbit_block}').glob(f'*apoapse*{orbit_code}*muv*.gz'))
files = [fits.open(f) for f in files]

lat = np.vstack([f['pixelgeometry'].data['pixel_corner_lat'] for f in files])
lon = np.vstack([f['pixelgeometry'].data['pixel_corner_lon'] for f in files])
lon = np.where(lon < 0, lon + 360, lon)
alt = np.vstack([f['pixelgeometry'].data['pixel_corner_mrh_alt'][..., 4] for f in files])
fov = np.concatenate([f['integration'].data['fov_deg'] for f in files])
swath_number = swath_number(fov)


def latlon_meshgrid(latitude, longitude, altitude):
    # make meshgrids to hold latitude and longitude grids for pcolormesh display
    X = np.zeros((latitude.shape[0] + 1, latitude.shape[1] + 1))
    Y = np.zeros((longitude.shape[0] + 1, longitude.shape[1] + 1))
    mask = np.ones((latitude.shape[0], latitude.shape[1]))

    # loop through pixel geometry arrays
    for i in range(int(latitude.shape[0])):
        for j in range(int(latitude.shape[1])):

            # there are some pixels where some of the pixel corner longitudes are undefined
            # if we encounter one of those, set the data value to missing so it isn't displayed
            # with pcolormesh
            if np.size(np.where(np.isfinite(longitude[i, j]))) != 5:
                mask[i, j] = np.nan

            # also mask out non-disk pixels
            if altitude[i, j] != 0:
                mask[i, j] = np.nan

            # place the longitude and latitude values in the meshgrids
            X[i, j] = longitude[i, j, 1]
            X[i + 1, j] = longitude[i, j, 0]
            X[i, j + 1] = longitude[i, j, 3]
            X[i + 1, j + 1] = longitude[i, j, 2]
            Y[i, j] = latitude[i, j, 1]
            Y[i + 1, j] = latitude[i, j, 0]
            Y[i, j + 1] = latitude[i, j, 3]
            Y[i + 1, j + 1] = latitude[i, j, 2]

    # set any of the NaN values to zero (otherwise pcolormesh will break even if it isn't displaying the pixel).
    X[np.where(~np.isfinite(X))] = 0
    Y[np.where(~np.isfinite(Y))] = 0

    # set to domain [-180,180)
    #X[np.where(X > 180)] -= 360

    # return the coordinate arrays and the mask
    return X, Y



# Make the plot and fill it with MARCI data, then IUVS
fig, ax = plt.subplots(1, 2)
vmax = 0.5
ax[1].pcolormesh(marci_lon, marci_lat, marci_ice.T, vmin=0, vmax=vmax)
ax[1].set_xlim(210, 300)
ax[1].set_ylim(-45, 45)

for swath in np.unique(swath_number):
    x, y = latlon_meshgrid(lat[swath==swath_number], lon[swath==swath_number], alt[swath==swath_number])
    ax[0].pcolormesh(x, y, ice[swath==swath_number], vmin=0, vmax=vmax)

ax[0].set_xlim(210, 300)
ax[0].set_ylim(-45, 45)

plt.savefig(f'/home/kyle/iuvs/retrievals/{orbit}-marci-comparison.png')
