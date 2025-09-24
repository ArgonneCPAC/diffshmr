import jax.numpy as jnp
import matplotlib.pyplot as plt
import smhm_func
from collections import namedtuple

# Define the parameters for the double Gaussian data
parameter = namedtuple("parameter", ["mu1", "sig1", "mu2", "sig2", "f"])
theta_true = parameter(mu1=0.0, sig1=0.7, mu2=3.0, sig2=1.2, f=0.3)

# Generate synthetic data
N_draws = 10000
x_draws = smhm_func.random_draw_inverse_CDF(
    theta_true, smhm_func.func_double_Gaussian, -5.0, 8.0, N_draws
)
# Create a histogram of the synthetic data
bins = jnp.linspace(-5.0, 8.0, 100)
hist, bin_edges = jnp.histogram(x_draws, bins=bins, density=False)
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

# tw_ndhist to smooth the data
data = smhm_func.tw_ndhist_1D(x_draws, bins, 1.0)

# Fit the double Gaussian model to the smooth histogram data
theta_init = parameter(mu1=-1.0, sig1=1.0, mu2=2.0, sig2=1.0, f=0.5)
bestfit_theta, loss_hist = smhm_func.gradient_descent(
    theta=theta_init,
    x=bins,
    y_target=data,
    model=smhm_func.model_double_Gaussian_ndhist,
    N_steps=10_000,
    stepsize=1e-5,
)
print("True parameters:", theta_true)
print("Best-fit parameters:", bestfit_theta)
print("Final loss:", loss_hist[-1])

# Plot the results
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(jnp.linspace(1, len(loss_hist) + 1, len(loss_hist)), loss_hist, "-")
ax[0].set_xlabel("number of steps")
ax[0].set_ylabel("MSE loss")
ax[0].set_xscale("log")
ax[0].set_yscale("log")

ax[1].hist(x_draws, bins=bins, alpha=0.5, label="raw data")
ax[1]lt.plot(bin_centers, data, label="smooth data")

ax[1].plot(
    bin_centers,
    smhm_func.model_dbl_Gauss_hist(theta_init, bins),
    ":k",
    label="initial guess",
)
ax[1].plot(
    bin_centers,
    smhm_func.model_dbl_Gauss_hist(bestfit_theta, bins),
    "--g",
    label="final model",
)
ax[1].legend()
plt.tight_layout()

plt.savefig("double_Gauss_fit_plot.pdf")