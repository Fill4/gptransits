import numpy as np
import matplotlib.pyplot as plt
import corner
from astropy.stats import LombScargle

def plot_corner(model, samples):
	labels = model.gp.gp_model.get_parameters_latex()
	params = model.gp.gp_model.get_parameters()
	fig2 = corner.corner(samples, labels=labels, quantiles=[0.5], show_titles=True, title_fmt='.3f', truths=params, num=2)

def plot_gp(model, data):

	fig1, ax1 = plt.subplots(num=1, figsize=(14, 7))
	
	# Plot initial data with errors in both subplots
	ax1.errorbar(model.time/(24*3600), model.flux, yerr=model.error, fmt=".k", capsize=0, label= 'Flux', markersize='3', elinewidth=1)
	x = np.linspace(model.time[0], model.time[-1], num=2*model.time.size)

	# Plot conditional predictive distribution of the model in upper plot
	mu, cov = model.gp.predict(model.flux, x/1e6)
	std = np.sqrt(np.diag(cov))
	std = np.nan_to_num(std)
	
	ax1.plot(x/(24*3600), mu, color="#ff7f0e", label= 'Mean distribution GP', linewidth=0.5)
	ax1.fill_between(x/(24*3600), mu+std, mu-std, color="#ff7f0e", alpha=0.6, edgecolor="none", label= '1 sigma', linewidth=0.6)
	#ax1.fill_between(x, mu+2*std, mu-2*std, color="#ff7f0e", alpha=0.3, edgecolor="none", label= '2 sigma', linewidth=0.5)
	#ax1.fill_between(x, mu+3*std, mu-3*std, color="#ff7f0e", alpha=0.15, edgecolor="none", label= '3 sigma', linewidth=0.5)
	
	ax1.set_xlabel('Time [days]',fontsize="medium")
	ax1.set_ylabel('Flux[ppm]',fontsize="medium")
	ax1.tick_params(axis='both', which='major', labelsize="medium")
	ax1.legend(loc='upper left', fontsize="medium")

def plot_psd(model, data, include_data=True):

	ls_dict = {"Granulation": "--", "OscillationBump": "-.", "WhiteNoise": ":"}
	alpha_dict = {"Granulation": 0.8, "OscillationBump": 0.8, "WhiteNoise": 0.6}
	label_dict = {"Granulation": "Granulation", "OscillationBump": "Gaussian envelope", "WhiteNoise": "White noise"}

	freq, power_dict = model.gp.gp_model.get_psd(model.time)
	nobump_power = np.zeros(freq.size)
	full_power = np.zeros(freq.size)
	
	fig3, ax3 = plt.subplots(num=3, figsize=(14, 7))
	for name, power in power_dict:
		# TODO: Need alternative to fix white noise psd
		if name == "WhiteNoise":
			power += 0.0
		if name != "OscillationBump":
			nobump_power += power
		full_power += power

		ax3.loglog(freq, power, ls=ls_dict[name], color='b', alpha=alpha_dict[name], label=label_dict[name])
	
	ax3.loglog(freq, nobump_power, ls='-', color='r', label='Model without gaussian')
	ax3.loglog(freq, full_power, ls='--', color='#7CFC00', label='Full Model')

	ax3.set_xlim([5, 300])
	ax3.set_ylim([0.1, 4000])
	
	# ax3.set_title('KIC012008916')
	ax3.set_xlabel(r'Frequency [$\mu$Hz]',fontsize="large")
	ax3.set_ylabel(r'PSD [ppm$^2$/$\mu$Hz]',fontsize="large")
	ax3.tick_params(labelsize="large")
	
	if include_data:
		# Psd from data
		freq2, power = LombScargle(model.time/1e6, model.flux).autopower(nyquist_factor=1, normalization='psd', samples_per_peak=1)
		ax3.loglog(freq2, power/model.time.size, color='k', alpha=0.4)
	
	ax3.legend(fontsize="large", loc="upper left")