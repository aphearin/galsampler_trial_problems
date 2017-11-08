"""
"""
import numpy as np
from astropy.table import Table
from scipy.stats import powerlaw, norm
from scipy.linalg import eigh
from astropy.utils.misc import NumpyRNGContext
from halotools.empirical_models import Cacciato09Cens


rband_dict1 = dict(gamma_1=3.273, log_L_0=9.935, log_M_1=11.07, gamma_2=0.255)
gband_dict1 = dict(gamma_1=2.5, log_L_0=9., log_M_1=10.7, gamma_2=0.55)
default_sqrt_cov = np.array(((0.05, 0.05), (0.05, 0.15)))


__all__ = ('generate_mock', )


def generate_mock(num_halos=int(1e6), seed=None,
            rband_dict=rband_dict1, gband_dict=gband_dict1,
            sqrt_cov=default_sqrt_cov):
    """
    """
    #  Initialize the table storing the galaxies
    galaxies = Table()
    galaxies['galid'] = np.arange(num_halos).astype(int)

    #  First generate a power law variable distributed across the unit interval
    with NumpyRNGContext(seed):
        uran = np.random.rand(num_halos)
    index = 2
    y = powerlaw.isf(1-uran, index)

    #  Now rescale the variable to span the typical range of host halo masses
    log_mhalo_min, log_mhalo_max = 10, 15.5
    dlog_mhalo = log_mhalo_max - log_mhalo_min
    galaxies['host_halo_mass'] = 10**(log_mhalo_max - dlog_mhalo*y)

    #  Calculate the eigenvectors of Cov
    cov = sqrt_cov*sqrt_cov
    evals, evecs = eigh(cov)

    #  Define a transformation matrix U so that U*U^T = cov
    U = np.dot(evecs, np.diag(np.sqrt(evals)))
    a = norm.rvs(size=num_halos)
    b = norm.rvs(size=num_halos)
    X = np.array((a, b)).reshape((2, num_halos))

    #  Use the transformation matrix U to calculate correlated variables (c, d)
    Y = np.dot(U, X)
    c, d = Y[0, :], Y[1, :]

    #  Use the Cacciato09 model in Halotools to model the mass-to-light ratio
    clf_model = Cacciato09Cens()

    #  Update the model dictionary and compute the median
    clf_model.param_dict.update(gband_dict)
    median_lg = clf_model.median_prim_galprop(prim_haloprop=galaxies['host_halo_mass'])

    #  Update the model dictionary and compute the median
    clf_model.param_dict.update(rband_dict)
    median_lr = clf_model.median_prim_galprop(prim_haloprop=galaxies['host_halo_mass'])

    #  Logarithmically add Gaussian noise
    log_median_r_and_g = np.log10(np.array((median_lr, median_lg)).reshape((2, num_halos)))

    log_mc_r_and_g = log_median_r_and_g + Y

    galaxies['luminosity_rband'] = 10**log_mc_r_and_g[0, :]
    galaxies['luminosity_gband'] = 10**log_mc_r_and_g[1, :]

    return galaxies
