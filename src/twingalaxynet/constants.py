"""Physical constants for TwinGalaxyNET.

The gravitational constant is converted with Astropy into the internal unit
system kpc^3 / (solar mass Myr^2). These units are convenient for galaxy
interaction experiments and avoid repeated unit conversions inside GPU kernels.
"""

from astropy import constants as const
from astropy import units as u


G_KPC3_PER_MSUN_MYR2 = const.G.to(
    u.kpc**3 / (u.M_sun * u.Myr**2)
).value

KM_S_TO_KPC_MYR = (1.0 * u.km / u.s).to(u.kpc / u.Myr).value
