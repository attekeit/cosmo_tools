import numpy as np
import pygad

#Some functions that hopefully are helpful with analysis and plotting

__all__ = ['time_to_z', 
           'z_to_time', 
           'get_eddington_rate',
           'add_z_as_second_xaxis']


def z_to_time(z, h = 0.674, Omega_L = 0.685, Omega_m = 0.315):

    """
    Calculate the time at redshift(s) z in Gyr. Default cosmological parameters
    are set to be the one used in both Mannerkoski and Keitaanranta zooms

    Args:
        z (double or an array of doubles): redshift(s)

        h (double): dimensionless Hubble parameter

        Omega_L: 

        Omega_m:

    Returns:
        t: either a double or an array correspinding to z in unit of Gyr
    """

    H = 100 * h
    Mpc_per_km = 3.086e19

    scale = 1.0/(z+1)
    time = 2./(3*np.sqrt(Omega_L))* np.arcsinh(np.sqrt(Omega_L/Omega_m)*  scale**(3./2)) /H*Mpc_per_km
    return time/(1e9*365.25*24*3600)

def time_to_z(t, h = 0.674, Omega_L = 0.685, Omega_m = 0.315):

    """
    Calculate the time redshift(t)s mathcing the time(s) given in Gyr. Default 
    cosmological parameters are set to be the one used in both Mannerkoski and 
    Keitaanranta zooms

    Args:
        t (double or an array of doubles): time(s) in units of Gyr

        h (double): dimensionless Hubble parameter

        Omega_L: 

        Omega_m:

    Returns:
        z: either a double or an array correspinding to t
    """

    H = 100 * h
    Mpc_per_km = 3.086e19

    #assuming t is given in Gyr
    t_in_s = t*1e9*365.25*24*3600
    a = (Omega_m/Omega_L)**(1./3)*(np.sinh(1.5*np.sqrt(Omega_L)*t_in_s*H/Mpc_per_km))**(2./3)
    z = 1/a-1
    return z

def get_eddington_rate(m, rad_eff = 0.1):
    """

    Calculate the time redshift(t)s mathcing the time(s) given in Gyr. Default 
    cosmological parameters are set to be the one used in both Mannerkoski and 
    Keitaanranta zooms

    Args:
        t (double or an array of doubles): time(s) in units of Gyr

        rad_eff (double): radiative efficiency of the simulation, default 0.1.

    Returns:
        fac: Eddington fraction(s) correspinding to mass(es) m
    """
    fac  = 2.2e-9/rad_eff * m
    return fac

def add_z_as_second_xaxis(ax, z_ticks = None, ticklabels= True, time_unit = 'Gyr',
                          h = 0.674, Omega_L = 0.685, Omega_m = 0.315):

    xlim = ax.get_xlim()
    
    if time_unit == 'Gyr':
        t0 = xlim[0]
        tend = xlim[-1]
    elif time_unit == 'Myr':
        t0 = xlim[0]/1000
        tend = xlim[-1]/1000
    else:
        print('Units', time_unit, 'not supported, use Myr or Gyr!')
        quit()

    zstart = time_to_z(t0)
    zend = time_to_z(tend)

    if z_ticks == None:
        z_ticks = np.linspace(9,2,8, dtype=int)

    
    def time_to_z_wrapper(t):
        if time_unit == 'Gyr':
            return time_to_z(t, h=h, Omega_L=Omega_L, Omega_m=Omega_m)
        else: 
            return time_to_z(t/1e3, h=h, Omega_L=Omega_L, Omega_m=Omega_m)
    def z_to_time_wrapper(z):
        if time_unit == 'Gyr':
            return z_to_time(z, h=h, Omega_L=Omega_L, Omega_m=Omega_m)
        else:
            return z_to_time(z, h=h, Omega_L=Omega_L, Omega_m=Omega_m)*1000

    #let's see how the ticks look... 

    secax = ax.secondary_xaxis(
        'top',
        functions=(time_to_z_wrapper, z_to_time_wrapper)
    )

    secax.set_xticks(z_ticks)
    if ticklabels:
        secax.set_xlabel('Redshift $z$')
    else:
        secax.set_xticklabels([])

    #Not sure what this should return, do we need to return these two?
    #Maybe good to return, in case of user wanting to make further edits
    return ax, secax

def get_luminosity_distance(z, H = 0.674, Omega_L = 0.686, Omega_m = 0.315,
                                Omega_b = 0.0491, sigma_8 = 0.81, n_s=0.965):

    
    cosmology = pygad.physics.FLRWCosmo(H/100, Omega_L, Omega_m, Omega_b, sigma_8, n_s)
    d_L = np.zeros(len(z))
    for i in range(len(d_L)):
        d_L[i] = cosmology.luminosity_distance(z[i]) #in Gpc
    return d_L

#Copied from Matias's second zoom paper analysis
def stellar_sigma(stars):
    
    vels = stars['vel'].in_units_of('km/s',subs=stars)
    if len(vels) < 3:
        return pygad.UnitQty(0., 'km/s')
    vmean = np.mean(vels, axis=0) 
    sigma2_ij = np.einsum("ki,kj", vels, vels)/len(vels) - vmean[:,np.newaxis] * vmean[np.newaxis]
    #maximum eigenvalue
    return pygad.UnitQty(np.sqrt(np.linalg.eigh(sigma2_ij)[0][-1]), 'km/s')
