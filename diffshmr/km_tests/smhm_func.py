import jax.numpy as jnp
from jax import grad
from jax import jit as jjit
from jax.scipy.special import erf as jsp_erf
from functools import partial
from collections import namedtuple
from scipy.stats import qmc
from .. import diffndhist


# Sigmoid function
@jjit
def Sigmoid(x, xc, y_lo, y_hi):
    # k = 1.0
    return y_lo + (y_hi - y_lo) / (1.0 + jnp.exp(-1.0 * (x - xc)))


# The rolling index
@jjit
def gamma(theta, mh):
    """
    Rolling index of three-roll SMHM relation.
    Introduced in Hearin et al. (2022).
    arXiv:2112.08423
    """
    xc_gL_par = theta.xc_gL
    y_lo_gL_par = theta.y_lo_gL
    y_hi_gL_par = theta.y_hi_gL
    xc_gH_par = theta.xc_gH
    y_lo_gH_par = theta.y_lo_gH
    y_hi_gH_par = theta.y_hi_gH
    mcrit_par = theta.mcrit

    gamma_lo = Sigmoid(mh, xc_gL_par, y_lo_gL_par, y_hi_gL_par)
    gamma_hi = Sigmoid(mh, xc_gH_par, y_lo_gH_par, y_hi_gH_par)
    return Sigmoid(mh, mcrit_par, gamma_lo, gamma_hi)


# The three-roll stellar mass-halo mass relation
@jjit
def three_roll_SMHM(theta, mh):
    """
    Three-roll stellar mass-halo mass relation.
    Introduced in Hearin et al. (2022).
    arXiv:2112.08423
    """
    mcrit_par = theta.mcrit
    m0_par = theta.m0

    ms = m0_par + gamma(theta, mh) * (mh - mcrit_par)
    return ms


# A broken power-law SMHM relation
@jjit
def broken_power_law_SMHM(theta, mh):
    lgM1_par = theta.lgM1
    lgMs0_par = theta.lgMs0
    gamma1_par = theta.gamma1
    gamma2_par = theta.gamma2

    return (
        lgMs0_par
        + gamma1_par * (mh - lgM1_par)
        - (gamma1_par - gamma2_par) * jnp.log10(1.0 + 10.0 ** (mh - lgM1_par))
    )


# The Moster et al. (2013) SMHM relation
@jjit
def ms_Moster13_fix_z(theta_moster, mh, z):
    """
    Moster et al. (2013) SMHM relation at fixed redshift z.
    """
    M_10_par = theta_moster.M_10
    M_11_par = theta_moster.M_11
    N_10_par = theta_moster.N_10
    N_11_par = theta_moster.N_11
    beta_10_par = theta_moster.beta_10
    beta_11_par = theta_moster.beta_11
    gamma_10_par = theta_moster.gamma_10
    gamma_11_par = theta_moster.gamma_11

    M_1_z = 10.0 ** (M_10_par + M_11_par * z / (1.0 + z))
    N_z = N_10_par + N_11_par * z / (1.0 + z)
    beta_z = beta_10_par + beta_11_par * z / (1.0 + z)
    gamma_z = gamma_10_par + gamma_11_par * z / (1.0 + z)

    M_h = 10.0**mh
    return (
        mh
        + jnp.log10(2.0 * N_z)
        - jnp.log10((M_h / M_1_z) ** (-beta_z) + (M_h / M_1_z) ** gamma_z)
    )


def default_Moster13_params():
    """Returns the default Moster et al. (2013) parameters."""
    theta_Moster = namedtuple(
        "theta_Moster",
        ["M_10", "M_11", "N_10", "N_11", "beta_10", "beta_11", "gamma_10", "gamma_11"],
    )
    theta_Moster_values = theta_Moster(
        M_10=11.590,
        M_11=1.195,
        N_10=0.0351,
        N_11=-0.0247,
        beta_10=1.376,
        beta_11=-0.826,
        gamma_10=0.608,
        gamma_11=0.329,
    )
    return theta_Moster_values


@jjit
def ms_Moster13_z0(theta_moster, mh):
    """
    Moster et al. (2013) SMHM relation at z=0."""
    return ms_Moster13_fix_z(theta_moster, mh, 0.0)


# The MSE loss function
@partial(jjit, static_argnums=(3,))
def mse_loss(theta, x, y_target, model):
    y_pred = model(theta, x)
    return jnp.mean((y_pred - y_target) * (y_pred - y_target))


# The gradient of the loss function
grad_mse_loss = jjit(grad(mse_loss), static_argnums=(3,))


# A simple gradient descent optimizer
def gradient_descent(theta, x, y_target, model, N_steps, stepsize):
    loss_history = []
    for step in range(N_steps):
        loss_history.append(mse_loss(theta, x, y_target, model))
        grad = grad_mse_loss(theta, x, y_target, model)

        # Update parameters
        theta = theta._make([t - stepsize * g for t, g in zip(theta, grad)])
    return theta, loss_history


@jjit
def dn_dlnM200m_Tinker08_z0(M):
    """
    Tinker et al. (2008) mass function fitting function
    at z=0 for overdensity 200m.
    The functional form here is obtained by fitting the
    profile generated from the Colossus package using a
    modified Schechter function.
    """
    # Parameters
    phi_s = 3.383319252e-5
    M_s = 2.60433e14
    alpha = -1.87238395
    beta = 0.72596204

    dn_dlnM = phi_s * (M / M_s) ** (alpha + 1) * jnp.exp(-((M / M_s) ** beta))

    return dn_dlnM


@jjit
def GSMF(theta, SMHM_model, HMF_model, sigma_ms, ms_array, mh_min, mh_max):
    """
    Galaxy stellar mass function for a given SMHM relation,
    and a given halo mass function.
    """
    # Define a few things
    dms = ms_array[1] - ms_array[0]
    gsmf = jnp.zeros_like(ms_array)

    # Create an array of halo masses
    mh = jnp.linspace(mh_min, mh_max, 1000)
    M = 10.0**mh
    dmh = mh[1] - mh[0]

    # Compute the halo mass function using Tinker et al. (2008)
    dn_dlnM = HMF_model(M)

    # Compute the GSMF:
    for i, ms in enumerate(ms_array):
        mu = SMHM_model(theta, mh)
        bin_edge_1 = ms - 0.5 * dms
        bin_edge_2 = ms + 0.5 * dms
        n_gal = jnp.sum(
            dn_dlnM
            * 0.5
            * (
                jsp_erf((bin_edge_2 - mu) / (sigma_ms * 2**0.5))
                - jsp_erf((bin_edge_1 - mu) / (sigma_ms * 2**0.5))
            )
        )
        gsmf = gsmf.at[i].set(n_gal / dms * jnp.log(10.0) * dmh)

    return gsmf


@jjit
def GSMF_3rSMHM(theta, ms_array):
    """
    Galaxy stellar mass function for the three-roll SMHM relation,
    and the Tinker et al. (2008) halo mass function at z=0.
    """
    return GSMF(
        theta=theta,
        SMHM_model=three_roll_SMHM,
        HMF_model=dn_dlnM200m_Tinker08_z0,
        sigma_ms=theta.sigma_ms,
        ms_array=ms_array,
        mh_min=10.0,
        mh_max=15.5,
    )


@jjit
def GSMF_bplSMHM(theta, ms_array):
    """
    Galaxy stellar mass function for the broken power-law SMHM relation,
    and the Tinker et al. (2008) halo mass function at z=0.
    """
    return GSMF(
        theta=theta,
        SMHM_model=broken_power_law_SMHM,
        HMF_model=dn_dlnM200m_Tinker08_z0,
        sigma_ms=theta.sigma_ms,
        ms_array=ms_array,
        mh_min=10.0,
        mh_max=15.5,
    )


"""
#----------------------------------------------
Some auxiliary functions for testing:
#----------------------------------------------
"""


@jjit
def func_Gaussian(theta, x):
    mu = theta.mu
    sig = theta.sig
    norm = 1.0 / (sig * jnp.sqrt(2.0 * jnp.pi))
    return norm * jnp.exp(-0.5 * ((x - mu) / sig) ** 2)


@jjit
def func_double_Gaussian(theta, x):
    mu1 = theta.mu1
    sig1 = theta.sig1
    mu2 = theta.mu2
    sig2 = theta.sig2
    f = theta.f

    gauss_par = namedtuple("gauss_par", ["mu", "sig"])
    par1 = gauss_par(mu1, sig1)
    par2 = gauss_par(mu2, sig2)

    gauss1 = func_Gaussian(par1, x)
    gauss2 = func_Gaussian(par2, x)

    return f * gauss1 + (1.0 - f) * gauss2


@partial(jjit, static_argnums=(1,))
def random_draw_inverse_CDF(theta, model, xmin, xmax, N_draws):
    """
    Random draws from a distribution defined by its inverse CDF.
    """
    # Create an array of x values
    x = jnp.linspace(xmin, xmax, 1000)
    dx = x[1] - x[0]

    # Compute the PDF
    pdf = model(theta, x)
    pdf = pdf / jnp.sum(pdf * dx)

    # Compute the CDF
    cdf = jnp.cumsum(pdf * dx)

    # Create an array of uniform random numbers
    sampler = qmc.Sobol(d=1, scramble=True)
    sobol_draws = sampler.random(n=N_draws).flatten()
    u = jnp.array(sobol_draws)

    # Interpolate to get the random draws
    random_draws = jnp.interp(u, cdf, x)

    return random_draws


@jjit
def model_double_Gaussian_ndhist(theta, bins, N_draws):
    mu1, sig1 = theta.mu1, theta.sig1
    mu2, sig2 = theta.mu2, theta.sig2
    f = theta.f

    xmin = jnp.min(bins)
    xmax = jnp.max(bins)

    gauss_par = namedtuple("gauss_par", ["mu", "sig"])
    par1 = gauss_par(mu1, sig1)
    par2 = gauss_par(mu2, sig2)

    g1_draws = random_draw_inverse_CDF(par1, func_Gaussian, xmin, xmax, N_draws)

    g2_draws = random_draw_inverse_CDF(par2, func_Gaussian, xmin, xmax, N_draws)

    hist1 = tw_ndhist_1D(g1_draws, bins, 1.0)
    hist2 = tw_ndhist_1D(g2_draws, bins, 1.0)

    return f * hist1 + (1.0 - f) * hist2


@jjit
def tw_ndhist_1D(data, bins, ndsig_by_dbin):
    """
    1D histogram with Gaussian kernel smoothing.
    """
    x = data.reshape((-1, 1))
    bin_lo = jnp.array(bins[:-1]).reshape((-1, 1))
    bin_hi = jnp.array(bins[1:]).reshape((-1, 1))
    dbin = bins[1] - bins[0]
    ndsig = jnp.ones_like(x) * (ndsig_by_dbin * dbin)

    hist = diffndhist.tw_ndhist(x, ndsig, bin_lo, bin_hi)
    return hist
