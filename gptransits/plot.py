import sys
import numpy as np
import matplotlib.pyplot as plt
import corner
import celerite
from astropy.timeseries import LombScargle
from astropy.convolution import convolve, Box1DKernel

from .convergence import gelman_rubin

days_to_microsec = (24*3600) / 1e6

# ----------------------------- PLOTTING ----------------------------------
def gelman_rubin_plot(chain, steps=100, pnames=None):
	R_hat = np.zeros([steps, chain.shape[2]])
	V = np.zeros([steps, chain.shape[2]])
	W = np.zeros([steps, chain.shape[2]])
	
	for s in range(steps):
		R_hat[s], V[s], W[s] = gelman_rubin(chain[:,0:int((s+1)/steps*chain.shape[1]),:])
	
	nrows = int((chain.shape[2]+1)/2)
	fig, ax = plt.subplots(nrows=nrows, ncols=2, figsize=(14, nrows*3))
	plt.subplots_adjust(bottom=0.1, top=0.95, left=0.08, right=0.9, wspace=0.4, hspace=0.4)
	ax = ax.flatten()

	x = np.array(range(1,steps+1))/steps * chain.shape[1]
	for p in range(chain.shape[2]):
		if pnames is not None:
			ax[p].set_title(f"Parameter - {pnames[p]}")
		else:
			ax[p].set_title(f"Parameter - {p+1}")

		ax[p].plot(x, R_hat[:,p], color='tab:red', lw=2, alpha=0.7, label=r"$\hat{R}$")
		ax[p].set_xlabel("Iterations")
		ax[p].set_ylabel(r"$\hat{R}$", color='tab:red')
		ax[p].legend(loc='upper left')

		ax2 = ax[p].twinx()
		ax2.plot(x, V[:,p], label=r"$\hat{V}$", color='tab:blue', lw=2, ls='--', alpha=0.6)
		ax2.plot(x, W[:,p], label="W", color='tab:blue', lw=2, ls=':', alpha=0.6)
		ax2.set_ylabel("Variance Estimates", color='tab:blue')
		ax2.legend()

	return fig

def parameter_hist(chain, params, bins=50, pnames=None):
	
	font=16
	w = chain.reshape([chain.shape[0]*chain.shape[1], chain.shape[2]])

	nrows = int(np.ceil(chain.shape[2] / 2))
	fig, ax = plt.subplots(ncols=2, nrows=nrows, figsize=(14,nrows*4))
	fig.subplots_adjust(hspace=0.3, top=0.94, bottom=0.05)
	ax = ax.ravel()
	fig.suptitle("Parameter Distributions", fontsize=24)

	for i in range(chain.shape[2]):
		if pnames is None:
			ax[i].set_title(f"Parameter - {i+1}", fontsize=font)
		else:
			ax[i].set_title(f"Parameter - {pnames[i]}", fontsize=font)
		ax[i].hist(w[:,i], bins=bins, density=True, fill=False, hatch='\\', alpha=0.7)
		
		if "median" in params:
			ax[i].axvline(params["median"][i], color="r", alpha=0.8, label="Median")
			if "hpd_down" in params and "hpd_up" in params:
				ax[i].axvline(params["hpd_up"][i], color="r", ls="--", alpha=0.8, label="HPD")
				ax[i].axvline(params["hpd_down"][i], color="r", ls="--", alpha=0.8)
		if "mapv" in params:
			ax[i].axvline(params["mapv"][i], color="b", alpha=0.8, label="MAP")
		if "modes" in params:
			ax[i].axvline(params["modes"][i], color="g", alpha=0.8, label="Mode")
		ax[i].legend(fontsize=font)
		ax[i].tick_params(labelsize=font)

	
	return fig

def trace_plot(chain, posterior, pnames=None, downsample=10):
	
	assert posterior.shape[1] == chain.shape[0]

	nrows = chain.shape[2]+1
	fig, ax = plt.subplots(ncols=1, nrows=nrows, figsize=(14,nrows*3))
	fig.subplots_adjust(hspace=0.3, top=0.95, bottom=0.05)

	# Plot the ln-probabilty for each of the walkers
	# assert(chain.shape[1] % downsample == 0)
	x = np.linspace(1, chain.shape[1], num=int(np.ceil(chain.shape[1]/downsample)))
	ax[0].set_title("Ln-Posterior")
	for w in range(posterior.shape[1]):
		ax[0].plot(x, posterior[::downsample,w], alpha=0.5)

	for i in range(1, len(ax)):
		if pnames is None:
			ax[i].set_title(f"Parameter - {i}")
		else:
			ax[i].set_title(f"Parameter - {pnames[i-1]}")
		for w in range(chain.shape[0]):
			ax[i].plot(x, chain[w,::downsample,i-1], alpha=0.5)
	
	return fig

def log_prob_hist(posterior):
	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14,10))

	ax.set_title("Ln-Posterior Histogram")
	ax.hist(posterior.reshape(posterior.shape[0]*posterior.shape[1]), bins=200, density=True) 
	ax.set_xlabel("Ln-Posterior")
	ax.set_ylabel("Probability")

	return fig

def corner_plot(chain, truths=None, pnames=None, downsample=2):
	sampled_chain = chain[:,::downsample,:]
	samples = sampled_chain.reshape(sampled_chain.shape[0]*sampled_chain.shape[1], sampled_chain.shape[2])
	fig = corner.corner(samples, quantiles=[0.16,0.5,0.84], show_titles=True, title_fmt='.3f', truths=truths, labels=pnames, bins=50, plot_contours=True)

	return fig

def psd_plot(gp_model, mean_model, params, time, flux, include_data=True, parseval_norm=True, nyquist_mult=1):

	font=20
	fig, ax = plt.subplots(figsize=(14, 7))
	fig.subplots_adjust(left=.10, bottom=.10, right=.98, top=.97)
	#fig.suptitle("Power Spectrum", fontsize=24)

	names = gp_model.get_component_names()
	gp_ndim = gp_model.get_parameters_names().size
	freq, psd_vector = gp_model.get_psd(params["median"][:gp_ndim], time, nyquist_mult=nyquist_mult)
	assert names.size == psd_vector.shape[0]

	# nobump_power = np.zeros(freq.size)
	full_psd = np.zeros(freq.size)
	for i in range(len(names)):
		full_psd += psd_vector[i]
		ax.loglog(freq, psd_vector[i], ls="--", alpha=0.8, lw=2, label=names[i])

	# Plot some realizations of the GP distributions in the PSD
	# medians = params["median"][:gp_ndim]
	# sigmas = ((medians - params["hpd_down"][:gp_ndim]) + (params["hpd_up"][:gp_ndim] - medians)) / 2
	# samples = np.transpose([np.random.normal(med, sig, 50) for med, sig in zip(medians, sigmas)])
	# for sample in samples:
	# 	freq, psd_vector = gp_model.get_psd(sample, time, nyquist_mult=nyquist_mult)
	# 	sample_psd = np.sum(psd_vector, axis=0) 
	# 	ax.loglog(freq, sample_psd, ls="-", color='red', lw=1, alpha=0.2)

	# ax.loglog(freq, nobump_power, ls='--', color='r', label='Model without gaussian', lw=2, alpha=0.8)
	ax.loglog(freq, full_psd, ls="-", color='red', label='Power Spectrum', lw=3)
	
	# Get nyquist frequency and plot its line
	cadence = (time[1] - time[0]) * days_to_microsec
	nyquist = 1 / (2 * cadence)
	ax.axvline(nyquist, ls="--", c="k", lw=2)

	if include_data:
		# Psd from data
		if mean_model is not None:
			mean_model.init_model(time, time[1]-time[0], 1)
			mean = mean_model.compute(params["median"][gp_ndim:], time)
			res_flux = flux - mean
		else:
			res_flux = flux

		freq2, power = LombScargle(time*days_to_microsec, res_flux).autopower(maximum_frequency=nyquist*nyquist_mult, normalization='psd', samples_per_peak=5)

		# Parseval Normalization (Enrico)
		if parseval_norm:
			resolution = freq2[1]-freq2[0]
			variance = np.var(res_flux)
			power = (power * variance / sum(power)) / resolution
		# Celerite Normalization
		else:
			power = power / time.size

		ax.loglog(freq2, power, color='k', alpha=0.3, lw=1, label='Data')
		ax.loglog(freq2, convolve(power, Box1DKernel(50)), color='k', alpha=0.5, lw=2, label='Smooth Data')
	
	# SETTINGS
	# ax.set_title("Power Spectrum", fontsize=font)
	ax.set_xlabel(r'Frequency ($\mu$Hz)', fontsize=font)
	ax.set_ylabel(r'PSD (ppm$^2$/$\mu$Hz)', fontsize=font)
	ax.tick_params(labelsize=font)

	ax.set_xlim([1,nyquist*nyquist_mult])
	ax.set_ylim([1, ax.get_ylim()[1]]) # Define lower bound and use auto upper bound
	ax.legend(loc="lower left", fontsize=font)

	return fig

def lc_double_plot(gp_model, mean_model, params, time, flux, flux_err=None, zoom=0.1, offset=0.0, oversample=5):

	# Global font size
	font=22

	# Oversampled array for plotting
	cadence = time[1]-time[0]
	x = np.linspace(time[0], time[-1], num=int((time[-1]-time[0])/cadence*oversample)) # Timespan / cadence * 5

	period_mask = params["names"] == "Period"
	t0_mask = params["names"] == "Epoch"
	period, t0 = params["median"][period_mask][0], params["median"][t0_mask][0]

	if (zoom + offset) > 1.0:
		print("Zoom + Offset percentages are larger than 1.0")
		sys.exit()
	# Zoom in limits
	low = np.where(time >= t0 - (0.15 * period) )[0][0]
	up = np.where(time >= t0 + (0.15 * period) )[0][0]
	olow = np.where(x >= t0 - (0.15 * period) )[0][0]
	oup = np.where(x >= t0 + (0.15 * period) )[0][0]

	# Zoom in data arrays
	time_zoom = time[low:up]
	flux_zoom = flux[low:up]
	if np.any(np.isnan(flux_err)):
		flux_err = None
		flux_err_zoom = None
	else:
		flux_err_zoom = flux_err[low:up]

	# Setup global figure
	fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(14, 10))
	fig.subplots_adjust(left=.11, bottom=.08, right=.98, top=.98)
	#fig.suptitle("Lightcurve", fontsize=24)

	ax = axs[0]
	ax_zoom = axs[1]

	# Plot initial data with errors
	ax.errorbar(time, flux, yerr=flux_err, fmt=".k", capsize=0, label="Data", markersize="5", elinewidth=1.2, alpha=0.5)

	# Zoomed in figure
	# fig_zoom, ax_zoom = plt.subplots(figsize=(14, 10))
	# fig_zoom.subplots_adjust(left=.12, bottom=.10, right=.95, top=.92)
	# Plot initial data with errors in both subplots
	ax_zoom.errorbar(time_zoom, flux_zoom, yerr=flux_err_zoom, fmt=".k", capsize=0, label="Data", markersize="8", elinewidth=1, alpha=0.5)
	
	# If we have both a GP noise model and a mean model, plot them separated and together
	if gp_model is not None and mean_model is not None:
		gp_ndim = gp_model.get_parameters_names().size

		#Setup mean_model
		mean_model.init_model(time, time[1]-time[0], 1)
		mean = mean_model.compute(params["median"][gp_ndim:], time)
		mean_model.init_model(x, time[1]-time[0], 1)
		overmean = mean_model.compute(params["median"][gp_ndim:], x)

		# Setup the gp from the gp_model
		kernel = gp_model.get_kernel(params["median"][:gp_ndim])
		gp = celerite.GP(kernel)
		if flux_err is None:
			gp.compute(time*days_to_microsec)
		else:
			gp.compute(time*days_to_microsec, yerr=flux_err) 

		mu, var = gp.predict(flux-mean, x*days_to_microsec, return_var=True)
		model = mu + overmean
		std = np.sqrt(var)
		std = np.nan_to_num(std)

		wnoise = params["median"][:gp_ndim][-1]
		noise = np.sqrt(wnoise**2 + var)

		# gp_samples = np.transpose([np.random.normal(med, sig, 20) for med, sig in zip(mu, noise)])
		# tr_medians = params["median"][gp_ndim:]
		# tr_sigmas = ((tr_medians - params["hpd_down"][gp_ndim:]) + (params["hpd_up"][gp_ndim:] - tr_medians)) / 2
		# tr_samples = np.transpose([np.random.normal(med, sig, 20) for med, sig in zip(tr_medians, tr_sigmas)])
		# for gp_sample, tr_params in zip(gp_samples, tr_samples):
		# 	tr_sample = mean_model.compute(tr_params, x)
		# 	ax.plot(x, gp_sample + tr_sample, color="tab:gray", lw=1, alpha=0.1, zorder=0.1)

		# Plot the model with GP predictive distribution
		ax.plot(x, model, color="#ff7f0e", linewidth=1, alpha=0.8, label='Model')
		ax.fill_between(x, model-noise, model+noise, color="#ff7f0e", alpha=0.5, edgecolor="none", label= r"1$\sigma$")

		# Correlated + White noise 
		#ax.fill_between(x, model+std, model-std, color="#ff7f0e", alpha=0.5, edgecolor="black", hatch="x", label= r"1$\sigma$ - Correlated", linewidth=1)
		#ax.fill_between(x, model+std, model+noise, color="#ff7f0e", alpha=0.5, edgecolor="black", label= r"1$\sigma$ - White noise")
		#ax.fill_between(x, model-noise, model-std, color="#ff7f0e", alpha=0.5, edgecolor="black")

		# Plot the GP mean
		# ax.plot(x, mu, color="red", linewidth=1, alpha=0.5, label="GP")
		# Plot the mean model
		ax.plot(x, overmean, color="blue", linewidth=1, alpha=0.8, label='Transit')


		# Repeat for the zoomed in plot
		x_zoom = x[olow:oup]
		mu_zoom = mu[olow:oup]
		std_zoom = std[olow:oup]
		model_zoom = model[olow:oup]
		overmean_zoom = overmean[olow:oup]
		noise_zoom = noise[olow:oup]

		# for gp_sample, tr_params in zip(gp_samples, tr_samples):
		# 	tr_sample = mean_model.compute(tr_params, x)
		# 	ax_zoom.plot(x_zoom, gp_sample[olow:oup] + tr_sample[olow:oup], color="tab:gray", lw=2, alpha=0.2, zorder=0.1)

		# Plot the GP predictive distribution
		ax_zoom.plot(x_zoom, model_zoom, color="#ff7f0e", label= 'Model', linewidth=2, alpha=0.7)
		ax_zoom.fill_between(x_zoom, model_zoom-noise_zoom, model_zoom+noise_zoom, color="#ff7f0e", 
			alpha=0.5, edgecolor="none", label= r"1$\sigma$")

		# Correlated + White noise zoom
		# ax_zoom.fill_between(x_zoom, model_zoom+std_zoom, model_zoom-std_zoom, color="#ff7f0e", 
		# 	alpha=0.5, edgecolor="none", hatch="x", label= r"1$\sigma$ - Correlated")
		# ax_zoom.fill_between(x_zoom, model_zoom+std_zoom, model_zoom+noise_zoom, color="#ff7f0e", 
		# 	alpha=0.5, edgecolor="none", label= r"1$\sigma$ - White noise")
		# ax_zoom.fill_between(x_zoom, model_zoom-noise_zoom, model_zoom-std_zoom, color="#ff7f0e", 
		# 	alpha=0.5, edgecolor="none")

		# Plot the GP mean
		#ax_zoom.plot(x_zoom, mu_zoom, color="red", linewidth=2, alpha=0.7, label="GP")
		# Plot the mean model
		ax_zoom.plot(x_zoom, overmean_zoom, color="blue", linewidth=2, alpha=0.7, label='Transit')
	
	# We only have the GP noise model and no mean
	elif gp_model is not None:
		# Setup the gp from the gp_model
		kernel = gp_model.get_kernel(params["median"])
		gp = celerite.GP(kernel)
		if flux_err is None:
			gp.compute(time*days_to_microsec)
		else:
			gp.compute(time*days_to_microsec, yerr=flux_err) 

		mu, var = gp.predict(flux, x*days_to_microsec, return_var=True)
		std = np.sqrt(var)
		std = np.nan_to_num(std)

		# Plot the model with GP predictive distribution
		ax.plot(x, mu, color="#ff7f0e", linewidth=1, alpha=0.5, label= 'Model')
		ax.fill_between(x, mu+std, mu-std, color="#ff7f0e", alpha=0.4, edgecolor="none", label= r"1$\sigma$", linewidth=1.2)

		# Repeat for the zoomed in plot
		x_zoom = x[olow:oup]
		mu_zoom = mu[olow:oup]
		std_zoom = std[olow:oup]

		# Plot the GP predictive distribution
		ax_zoom.plot(x_zoom, mu_zoom, color="#ff7f0e", label= 'Model', linewidth=2, alpha=0.5)
		ax_zoom.fill_between(x_zoom, mu_zoom+std_zoom, mu_zoom-std_zoom, color="#ff7f0e", alpha=0.4, edgecolor="none", label= r"1$\sigma$", linewidth=1.5)

	# Otherwise we only have a mean model
	elif mean_model is not None:
		#Setup mean_model
		mean_model.init_model(x, time[1]-time[0], 1)
		overmean = mean_model.compute(params["median"], x)

		# Plot the mean model
		ax.plot(x, overmean, color="blue", linewidth=1, alpha=0.5, label='Model')

		x_zoom = x[olow:oup]
		overmean_zoom = overmean[olow:oup]

		# Plot the mean model
		ax_zoom.plot(x_zoom, overmean_zoom, color="blue", linewidth=2, alpha=0.5, label='Model')
	
	# ax.set_title("GP", fontsize=font)
	ax.set_xlabel('Time [BTJD]', fontsize=font)
	ax.set_ylabel('Flux [ppm]', fontsize=font)
	ax.legend(loc='lower right', fontsize=font-4)
	ax.tick_params(labelsize=font-1)

	# ax_zoom.set_title("GP Zoom-in", fontsize=font)
	ax_zoom.set_xlabel('Time [BTJD]', fontsize=font)
	ax_zoom.set_ylabel('Flux [ppm]', fontsize=font)
	ax_zoom.legend(loc='lower right', fontsize=font-4)
	ax_zoom.tick_params(labelsize=font-1)

	return fig