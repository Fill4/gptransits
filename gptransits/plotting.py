import numpy as np
import matplotlib.pyplot as plt
import celerite
import corner
from astropy.stats import LombScargle

#Internal imports
from backend import setup_gp, scale_params

def plot_corner(params, samples, priors):
	labels = [priors[i][0] for i in priors]
	fig2 = corner.corner(samples, labels=labels, quantiles=[0.5], show_titles=True, title_fmt='.3f', truths=params, figsize=(14, 14), dpi=250, num=2)

def plot_gp(params, data):

	time, flux, error = data
	# Plotting the results from the MCMC method
	fig1, ax1 = plt.subplots(num=1, figsize=(14, 7), dpi=250)
	
	# Plot initial data with errors in both subplots
	ax1.errorbar(time/(24*3600), flux, yerr=error, fmt=".k", capsize=0, label= 'Flux', markersize='3', elinewidth=1)
	x = np.linspace(min(time), max(time), num=1000)

	# Setup GP
	gp = setup_gp(params)
	#gp.compute(time/1e6, error)
	gp.compute(time/1e6)
	
	# Plot conditional predictive distribution of the model in upper plot
	mu, cov = gp.predict(flux, x/1e6)
	std = np.sqrt(np.diag(cov))
	std = np.nan_to_num(std)
	
	ax1.plot(x/(24*3600), mu, color="#ff7f0e", label= 'Mean distribution GP', linewidth=0.5)
	ax1.fill_between(x/(24*3600), mu+std, mu-std, color="#ff7f0e", alpha=0.6, edgecolor="none", label= '1 sigma', linewidth=0.6)
	#ax1.fill_between(x, mu+2*std, mu-2*std, color="#ff7f0e", alpha=0.3, edgecolor="none", label= '2 sigma', linewidth=0.5)
	#ax1.fill_between(x, mu+3*std, mu-3*std, color="#ff7f0e", alpha=0.15, edgecolor="none", label= '3 sigma', linewidth=0.5)
	
	ax1.set_xlabel('Time [days]',fontsize=15)
	ax1.set_ylabel('Flux[ppm]',fontsize=15)
	ax1.tick_params(axis='both', which='major', labelsize=16)
	ax1.legend(loc='upper left', fontsize=15)
	#ax1.set_xlim((time[0], time[200]/(24*3600)))

def plot_psd(model, data, include_data=True):
	
	time, flux, error = data

	# tess_timespan = 27.4
	# nyquist = (1 / (2*(time[1]-time[0])))*1e6
	# f_sampling = 1 / ( tess_timespan * 24 * 3600 / 1e6)
	# freq = np.linspace(0.0, nyquist, (nyquist/f_sampling)+1 )

	# scaled_params = scale_params(params)
	# nparams = len(scaled_params)

	# if nparams == 5:
	# 	S0_bump, Q_bump, w_bump, S0_gran_1, w_gran_1 = np.log(scaled_params)
	# elif nparams == 6:
	# 	S0_bump, Q_bump, w_bump, S0_gran_1, w_gran_1, jitter = np.log(scaled_params)
	# elif nparams == 7:
	# 	S0_bump, Q_bump, w_bump, S0_gran_1, w_gran_1, S0_gran_2, w_gran_2 = np.log(scaled_params)
	# elif nparams == 8:
	# 	S0_bump, Q_bump, w_bump, S0_gran_1, w_gran_1, S0_gran_2, w_gran_2, jitter = np.log(scaled_params)

	# kernel_bump = celerite.terms.SHOTerm(log_S0=S0_bump, log_Q=Q_bump, log_omega0=w_bump)
	# kernel = kernel_bump

	# Q = 1.0 / np.sqrt(2.0)
	# kernel_1 = celerite.terms.SHOTerm(log_S0=S0_gran_1, log_Q=np.log(Q), log_omega0=w_gran_1)
	# kernel_1.freeze_parameter("log_Q")
	# kernel += kernel_1
	# kernel_gran = kernel_1

	# if nparams == 7 or nparams == 8:
	# 	kernel_2 = celerite.terms.SHOTerm(log_S0=S0_gran_2, log_Q=np.log(Q), log_omega0=w_gran_2)
	# 	kernel_2.freeze_parameter("log_Q")
	# 	kernel += kernel_2 
	# 	kernel_gran += kernel_2

	# if nparams == 6 or nparams == 8:
	# 	kernel_jitter = celerite.terms.JitterTerm(log_sigma=jitter)
	# 	kernel += kernel_jitter

	# power_bump = kernel_bump.get_psd(2*np.pi*freq)
	# power_1 = kernel_1.get_psd(2*np.pi*freq)
	# if nparams == 7 or nparams == 8:
	# 	power_2 = kernel_2.get_psd(2*np.pi*freq)
	# if nparams == 6 or nparams == 8:
	# 	power_jitter = kernel_jitter.get_psd(2*np.pi*freq)
	# 	power_jitter += 6.4

	# nobump_power = kernel_gran.get_psd(2*np.pi*freq)
	# full_power = kernel.get_psd(2*np.pi*freq)
	# if nparams == 6 or nparams == 8:
	# 	nobump_power += power_jitter
	# 	full_power += power_jitter
	font = 6
	ls_dict = {"Granulation": "--", "OscillationBump": "-.", "WhiteNoise": ":"}
	alpha_dict = {"Granulation": 0.8, "OscillationBump": 0.8, "WhiteNoise": 0.6}
	label_dict = {"Granulation": "Granulation", "OscillationBump": "Gaussian envelope", "WhiteNoise": "White noise"}

	freq, power_dict = model.get_psd(time)
	nobump_power = np.zeros(freq.size)
	full_power = np.zeros(freq.size)
	
	fig3, ax3 = plt.subplots(num=3, figsize=(14, 7), dpi=150)
	for name, power in power_dict.items():
		if name == "WhiteNoise":
			power += 8
		if name != "OscillationBump":
			nobump_power += power
		full_power += power

		ax3.loglog(freq, power, ls=ls_dict[name], color='b', alpha=alpha_dict[name], label=label_dict[name])
	
	ax3.loglog(freq, nobump_power, ls='-', color='r', label='Model without gaussian')
	ax3.loglog(freq, full_power, ls='--', color='#7CFC00', label='Full Model')

	# ax3.set_title('KIC012008916')
	# ax3.loglog(freq, full_power, ls='--', color='#7CFC00', label='Full Model')
	# ax3.loglog(freq, nobump_power, ls='-', color='r', label='Model without gaussian')
	# ax3.loglog(freq, power_bump, ls='-.', color='b', alpha=0.8, label='Gaussian envelope')
	# ax3.loglog(freq, power_1, ls='--', color='b', alpha=0.8, label='Granulation')
	# if nparams == 7 or nparams == 8:
	# 	ax3.loglog(freq, power_2, ls='--', color='b', alpha=0.8)
	# if nparams == 6 or nparams == 8:
	# 	ax3.loglog(freq, power_jitter, ls=':', color='b', alpha=0.6, label='White noise')

	ax3.set_xlim([5, 300])
	ax3.set_ylim([0.1, 4000])
	
	# ax3.set_title('KIC012008916')
	ax3.set_xlabel(r'Frequency [$\mu$Hz]',fontsize="large")
	ax3.set_ylabel(r'PSD [ppm$^2$/$\mu$Hz]',fontsize="large")
	ax3.tick_params(labelsize="large")
	
	if include_data:
		# Psd from data
		freq2, power = LombScargle(time/1e6, flux).autopower(nyquist_factor=1, normalization='psd', samples_per_peak=1)
		ax3.loglog(freq2, power/time.size, color='k', alpha=0.4)
		ax3.legend(fontsize="large")