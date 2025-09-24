import jax.numpy as jnp
import matplotlib.pyplot as plt
from . import smhm_func
from collections import namedtuple

theta_Moster_values = smhm_func.default_Moster13_params()
mh_data = jnp.linspace(10.0, 15.5, 30)
ms_data = smhm_func.ms_Moster13_z0(theta_Moster_values, mh_data)

ms_gal = jnp.linspace(8.5, 12.0, 30)
n_gal = smhm_func.GSMF(
    theta=theta_Moster_values,
    SMHM_model=smhm_func.three_roll_SMHM,
    HMF_model=smhm_func.dn_dlnM200m_Tinker08_z0,
    sigma_ms=0.2,
    ms_array=ms_gal,
    mh_min=10.0,
    mh_max=15.5,
)

# Fitting the data with the three-roll SMHM relation
threeroll_par_with_sigma = namedtuple(
    "threeroll_par_with_sigma",
    [
        "mcrit",
        "m0",
        "xc_gL",
        "y_lo_gL",
        "y_hi_gL",
        "xc_gH",
        "y_lo_gH",
        "y_hi_gH",
        "sigma_ms",
    ],
)
threeroll_par_with_sig_init = threeroll_par_with_sigma(
    mcrit=12.0,
    m0=10.5,
    xc_gL=11.5,
    y_lo_gL=0.3,
    y_hi_gL=0.7,
    xc_gH=13.0,
    y_lo_gH=0.7,
    y_hi_gH=0.3,
    sigma_ms=0.3,
)

best_fit_params, loss_hist = smhm_func.gradient_descent(
    theta=threeroll_par_with_sig_init,
    x=ms_gal,
    y_target=n_gal,
    model=smhm_func.GSMF_3rSMHM,
    N_steps=10_000,
    stepsize=1e-4,
)

print("Best-fit parameters:")
print(best_fit_params)
print("Final loss:", loss_hist[-1])

# Plotting the results
fig, axs = plt.subplots(1, 3, figsize=(13, 5))
mh_plotting = jnp.linspace(10.0, 15.0, 100)

ax = axs[0]
ax.plot(jnp.linspace(1, len(loss_hist) + 1, len(loss_hist)), loss_hist, "-")
ax.set_xlabel("number of steps")
ax.set_ylabel("MSE loss")
ax.set_xscale("log")
ax.set_yscale("log")

ax = axs[1]
ax.set_xlabel(r"$\log ( \, M_{\rm h}  \,\, / \,\, [M_\odot] \, )$")
ax.set_ylabel(r"$\log ( \, M_{\rm *,cen} \,\, / \,\, [M_\odot] \, )$")
ax.plot(mh_data, ms_data, ".k", label="Moster+13")
ax.plot(
    mh_plotting,
    smhm_func.three_roll_SMHM(threeroll_par_with_sig_init[:-1], mh_plotting),
    "--",
    color="orange",
    label="initial guess",
)
ax.plot(
    mh_plotting,
    smhm_func.three_roll_SMHM(best_fit_params[:-1], mh_plotting),
    "-",
    color="red",
    label="final fit",
)
ax.legend()

ax = axs[2]
ax.plot(ms_gal, n_gal, ".k")
ax.plot(
    ms_gal,
    10.0 ** smhm_func.GSMF_3rSMHM(threeroll_par_with_sig_init, ms_gal),
    "--",
    color="orange",
)
ax.plot(ms_gal, 10.0 ** smhm_func.GSMF_3rSMHM(best_fit_params, ms_gal), "-r")
ax.set_yscale("log")
ax.set_ylabel(r"${\rm d}n_{\rm gal} / {\rm d} \log M_{\rm *,cen} \,\, \, [Mpc^{-3}]$")
ax.set_xlabel(r"$\log ( \, M_{\rm *,cen} \,\, / \,\, [M_\odot] \, )$")

plt.tight_layout()
plt.savefig("threeroll_shmr_GSMF_fit_plot.pdf")
