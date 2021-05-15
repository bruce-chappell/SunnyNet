import numpy as np
from numpy import newaxis as nax
from tqdm import tqdm
from astropy import units as u
from astropy import units
from astropy import constants as const
from astropy.modeling.models import BlackBody 
from scipy.integrate import cumtrapz
from scipy.special import wofz   # for Voigt function


i_units = "kW m-2 sr-1 nm-1"  # More practical SI units
# For line extinction:
alpha_const = const.e.si ** 2 / (4 * const.eps0 * const.m_e * const.c ** 2) 

##
# THIS HAS BEEN DROPPED IN FAVOR OF 3_INTENSITY_CALC.JL
##

def voigt(gamma, x):
    """
    Computes the Voigt function.
    """
    z = (x + 1j * gamma)
    return wofz(z).real


def compute_doppler_width(wave0, temperature, mass):
    """
    Computes the Doppler width.
    
    Parameters
    ----------
    wave0: astropy.units.quantity (scalar)
        Rest wavelength of the bound-bound transition, in units of length.
    temperature: astropy.units.quantity (scalar or array)
        Temperature(s) where to calculate the Doppler width.
    mass: astropy.units.quantity (scalar)
        Mass of the atomic species, in units of mass.
        
    Returns
    -------
    doppler_width: astropy.units.quantity (scalar or array)
        Doppler width in units of length. Same shape as temperature.
    """
    return wave0 / const.c * np.sqrt(2 * const.k_B * temperature / mass)


def compute_gamma_natural(wave0, g_ratio, f_value):
    """
    Computes the natural line damping parameter for a bound-bound transition.
    Here using the expression for Aul in SI:
    
    $$
    A_{ul} = \frac{2\pi e^2}{\varepsilon_0 m_e c} \frac{g_l}{g_u} \frac{f_{lu}}{\lambda^2}
    $$
        
    Parameters
    ----------
    wave0: astropy.units.quantity (scalar or array)
        Transition wavelength.
    g_ratio: float
        Ratio between statistical weights of lower and upper levels (gl / gu)
    f_value: float
        Transition f value.

    Returns
    -------
    gamma - astropy.units.quantity (scalar or array)
        Damping parameter in units of s^-1.
    """
    a_const = 2 * np.pi * const.e.si**2 / (const.eps0 * const.m_e * const.c)
    return (a_const * g_ratio * f_value / wave0**2).to('s-1')


def compute_gamma_vdW(temperature, gas_pressure, l_l, l_u, stage, e_ion_l, e_ion_u):
    """
    Computes the van der Waals damping parameter for a bound-bound transition.
        
    Parameters
    ----------
    wave: astropy.units.quantity (scalar or array)
        Wavelength to calculate.
    temperature: astropy.units.quantity (scalar or array)
        Gas temperature in units of K or equivalent.
    gas_pressure: astropy.units.quantity (scalar)
        Gas pressure in units of Pa or equivalent. Same shape as temperature.
    l_l: integer
        Angular quantum number of lower level
    l_u: integer
        Angular quantum number of upper level
    stage: integer
        Ionisation stage, where 1 is neutral, 2 first ionised, and so on.
    e_ion_l: astropy.units.quantity (scalar)
        Ionisation energy from the lower level of the transition.
    e_ion_u: astropy.units.quantity (scalar)
        Ionisation energy from the upper level of the transition.
    """
    # effective principal quantum number
    def _compute_r_square(l_number, e_ion):
        ryd = const.Ryd.to("eV", equivalencies=units.spectral())
        n_star = ryd * stage ** 2 / e_ion
        r_square = (n_star / (2 * stage ** 2) * 
                    (5 * n_star + 1 - 3 * l_number * (l_number + 1)))
        return r_square
    
    # From Unsold recipe
    rdiff = _compute_r_square(l_u, e_ion_u) - _compute_r_square(l_l, e_ion_l) 
    log_gamma = (6.33 + 0.4 * np.log10(rdiff.value) + 
                 np.log10(gas_pressure.cgs.value) - 0.7 * np.log10(temperature.value))
    return 10 ** log_gamma / units.s


def compute_line_profile(wave0, element_mass, wavelength, temperature, v_los, gamma):
    """
    Computes line profile for Na I D divided by Doppler width, given a wavelength, 
    temperature, pressure, and line of sight velocity.
    
    Parameters
    ----------
    wavelength: astropy.units.quantity (scalar)
        Wavelength to calculate, in units of length.
    temperature: astropy.units.quantity (n-D array)
        Gas temperature in units of K or equivalent.
    pressure: astropy.units.quantity (n-D array)
        Gas pressure in units of Pa or equivalent.
    v_los: astropy.units.quantity (n-D array)
        Line of sight velocity in units of m/s or equivalent.
    gamma : astropy.units.quantity (n-D array)
        Radiative broadening in units of Hz or equivalent.
        
    Returns
    -------
    profile: n-D array
        Line profile divided by Doppler width for a given wavelength, same shape as temperature.
    """
    doppler_width = compute_doppler_width(wave0, temperature, element_mass)
    damping = wavelength ** 2 / (4 * np.pi * const.c * doppler_width) * (gamma)
    v = ((wavelength - wave0 + wave0 * v_los / const.c) / doppler_width).si
    return voigt(damping, v) / doppler_width / np.sqrt(np.pi)


def compute_line_extinction(wavelength, g_l, g_u, f_value, n_l, n_u, profile):
    """
    Computes total extinction coefficient for a spectal line, including also
    continuum extinction (H- and Thomson scattering), and stimulated emission,
    all for a single wavelength and a 3D model.
    
    Parameters
    ----------
    wave0: astropy.units.quantity (scalar or array)
        Transition wavelength.
    f_value: float
        Transition f value.
    wavelength: astropy.units.quantity (scalar)
        Wavelength to calculate, in units of length.
    n_l: astropy.units.quantity (3-D array)
        Number density of atoms in the lower state (in m^-3)
    n_u: astropy.units.quantity (3-D array)
        Number density of atoms in the upper state (in m^-3)
    g_l: scalar
        Statistical weight of lower level
    g_u: scalar
        Statistical weight of upper level
    profile: astropy.units.quantity (4-D array)
        Line profile in units of nm^-1.
        
    Returns
    -------
    total_ext: 3-D array
        Total extinction in units of per metre.
    """
    stim = (1 - (g_l/g_u) * n_u/n_l)
    return (alpha_const * (wavelength**2) * n_l * f_value * profile * stim).si


def compute_hminus_extinction(wavelength, temperature, electron_density):
    """
    Computes the H minus extinction cross section, both free-free and
    bound-free as per Gray (1992).
    
    Parameters
    ----------
    wavelength : astropy.units.quantity (array)
        Wavelength(s) to calculate in units of length.
    temperature: astropy.units.quantity (scalar or array)
        Gas temperature in units of K or equivalent.
    electron_density: astropy.units.quantity (scalar or array)
        Electron density in units of per cubic length.
        
    Returns
    -------
    extinction : astropy.units.quantity (scalar or array)
        Total H- extinction in si units. 
        Shape: shape of temperature + (nwave,)
    """
    # Broadcast to allow function of temperature and wavelength
    temp = temperature[..., nax]
    wave = wavelength[nax]
    theta = 5040 * units.K / temp
    electron_pressure = electron_density[..., nax] * const.k_B * temp
    # Compute bound-free opacity for H-, following Gray 8.11-8.12
    sigma_coeff = np.array([2.78701e-23, -1.39568e-18,  3.23992e-14, -4.40524e-10,
                               2.64243e-06, -1.18267e-05,  1.99654e+00])
    sigma_bf = np.polyval(sigma_coeff, wave.to_value('AA'))
    sigma_bf = sigma_bf * 1.e-22 * units.m ** 2
    # Set to zero above the H- ionisation limit at 1644.4 nm
    sigma_bf[wave > 1644.2 * units.nm] = 0.
    # convert into bound-free per neutral H atom assuming Saha,  Gray p156
    k_const = 4.158E-10 * units.cm ** 2 / units.dyn
    gray_saha = k_const * electron_pressure.cgs * theta ** 2.5 * 10. ** (0.754 * theta)
    kappa_bf = sigma_bf * gray_saha                    # per neutral H atom
    # correct for stimulated emission
    kappa_bf *= (1 - np.exp(-const.h * const.c / (wave * const.k_B * temp))) 

    # Now compute free-free opacity, following Gray 8.13
    # coefficients for 4th degree polynomials in the log of wavelength (in AA)
    coeffs = np.array([[-0.0533464, 0.76661, -1.685, -2.2763],
                          [-0.142631, 1.99381, -9.2846, 15.2827],
                          [-0.625151, 10.6913, -67.9775, 190.266, -197.789]], dtype='object')
    log_wave = np.log10(wave.to_value('AA'))
    log_theta = np.log10(theta.value)
    tmp = 0
    for i in range(3):
        tmp += np.polyval(coeffs[i], log_wave) * (log_theta ** i)
    kappa_ff = electron_pressure * (10 ** tmp) 
    kappa_ff = kappa_ff * 1e-26 * (units.cm ** 4) / units.dyn
    return (kappa_bf + kappa_ff).si


def compute_continuum_extinction(wavelength, temperature, electron_density, hpops):
    """
    Combines all sources of continuum extinction.
    """
    thomson_ext = 6.648e-29 * units.m ** 2 * electron_density
    hminus_ext = np.squeeze(compute_hminus_extinction(wavelength, temperature, electron_density)) * hpops
    return hminus_ext + thomson_ext 


def compute_intensity(wavelength, distance, source_function, extinction):
    """
    Solves the radiative transfer equation assuming LTE for a single ray
    and a single wavelength.
    
    Parameters
    ----------
    wavelength: astropy.units.quantity (scalar)
        Wavelength to calculate, in units of length.
    distance : astropy.units.quantity (1-D array)
        Distances along path of ray, in units of length. Can be different
        length than wavelength array.
    extinction: astropy.units.quantity (n-D array)
        Monochromatic extinction coefficient in units of inverse length, 
        for all points along the ray. The shape of this array
        should be `(npath,)`, where npath is thenumber of points the ray crosses.
        npath can also be multidimensional, so the shape could be `(nz, ny, nx)`
        or `(nz,)`.
    source_function: astropy.units.quantity (n-D array)
        Monochromatic source function in units of intensity (e.g. W m-2 sr-1 nm-1).
        Same shape as extinction.
    """
    tau = cumtrapz(extinction, x=distance, initial=0, axis=-1)
    return np.trapz(source_function * np.exp(-tau), tau, axis=-1)


def calculate_halpha(hpops, temperature, electron_density, pressure, vlos, height, wavelength):
    """
    Calculates Halpha line profile for a given set of H populations,
    temperature, electron density, line-of-sight velocities, height,
    and wavelength.
    """
    # Atomic data for Halpha
    wave0 = 656.47726 * u.nm  # vacuum
    g_u = 18
    g_l = 8
    f_value = 6.411e-01
    l_u = 2
    l_l = 1
    e_ion_l = 0.54465284 * u.aJ
    e_ion_u = 0.24205465 * u.aJ
    element_mass = const.u + const.m_e
    n_l = hpops[1]
    n_u = hpops[2]

    gamma_vdw = compute_gamma_vdW(temperature, pressure, l_l, l_u, 1, e_ion_l, e_ion_u)
    Aul = compute_gamma_natural(wave0, g_l / g_u, f_value)
    gamma = (gamma_vdw + Aul)
    
    blambda = BlackBody(temperature, scale=1.0*units.W / (units.m ** 2 * units.nm * units.sr))(wave0)
    ext_cont = compute_continuum_extinction(wave0, temperature, electron_density, 
                                            hpops[:-1].sum(0))
    j_cont = blambda * ext_cont
    
    nx, ny = temperature.shape[:2]
    nwave = len(wavelength)
    intensity = u.Quantity(np.empty((nx, ny, nwave), dtype='f'), unit=i_units)
    
    for i, wave in tqdm(enumerate(wavelength), total=nwave):
        profile = compute_line_profile(wave0, element_mass, wave, temperature, vlos, gamma)
        ext_line = compute_line_extinction(wave, g_l, g_u, f_value, n_l, n_u, profile)
        j_line = const.h * const.c / (4 * np.pi * wave0) * n_u * Aul * profile / u.sr
        source_function = (j_line + j_cont) / (ext_line + ext_cont)
        intensity[..., i] = compute_intensity(wave, -height, source_function, 
                                              ext_cont + ext_line).astype('f')
    return intensity