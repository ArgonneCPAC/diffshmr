import jax.numpy as jnp
import matplotlib.pyplot as plt
from . import smhm_func
from collections import namedtuple

theta_Moster_values = smhm_func.default_Moster13_params()
mh_data = jnp.linspace(10.0, 15.5, 30)
ms_data = smhm_func.ms_Moster13_z0(theta_Moster_values, mh_data)

# Fitting the data with the three-roll SMHM relation
threeroll_par = namedtuple(
    "threeroll_par",
    ["mcrit", "m0", "xc_gL", "y_lo_gL", "y_hi_gL", "xc_gH", "y_lo_gH", "y_hi_gH"],
)
threeroll_par_init = threeroll_par(
    mcrit=12.0,
    m0=10.5,
    xc_gL=11.5,
    y_lo_gL=0.3,
    y_hi_gL=0.7,
    xc_gH=13.0,
    y_lo_gH=0.7,
    y_hi_gH=0.3,
)

best_fit_params, loss_hist = smhm_func.gradient_descent(
    theta=threeroll_par_init,
    x=mh_data,
    y_target=ms_data,
    model=smhm_func.three_roll_SMHM,
    N_steps=10_000,
    stepsize=1e-4,
)

print("Best-fit parameters:")
print(best_fit_params)
print("Final loss:", loss_hist[-1])

# Plotting the results
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
mh_plotting = jnp.linspace(10.0, 15.0, 100)

ax[1].set_xlabel(r"$\log ( \, M_{\rm h}  \,\, / \,\, [M_\odot] \, )$")
ax[1].set_ylabel(r"$\log ( \, m_{\rm *,cen} \,\, / \,\, [M_\odot] \, )$")
ax[1].plot(mh_data, ms_data, ".k", label="Moster+13")
ax[1].plot(
    mh_plotting,
    smhm_func.three_roll_SMHM(threeroll_par_init, mh_plotting),
    "--",
    color="orange",
    label="initial guess",
)
ax[1].plot(
    mh_plotting,
    smhm_func.three_roll_SMHM(best_fit_params, mh_plotting),
    "--",
    color="red",
    label="final fit",
)

ax[1].legend()

ax[0].plot(jnp.linspace(1, len(loss_hist) + 1, len(loss_hist)), loss_hist, "-")
ax[0].set_xlabel("number of steps")
ax[0].set_ylabel("MSE loss")
ax[0].set_xscale("log")
ax[0].set_yscale("log")
ax[0].plot([1, 5_000], [0, 0], "--", color="grey")
plt.savefig("threeroll_shmr_direct_fit_plot.pdf")
