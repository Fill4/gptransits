#!/usr/bin/env python3

import numpy as np
import sys
import matplotlib.pyplot as plt
from astropy.stats import LombScargle
from model import *
from component import *

time, flux, error = np.loadtxt('presentation/KIC012008916_Q16.1.dat', unpack=True)

ls_dict = {"Granulation": "--", "OscillationBump": "-.", "WhiteNoise": ":"}
alpha_dict = {"Granulation": 0.8, "OscillationBump": 0.8, "WhiteNoise": 0.6}
label_dict = {"Granulation": "Granulation", "OscillationBump": "Gaussian envelope", "WhiteNoise": "White noise"}

bump = OscillationBump(78.2919, 7.6602, 167.0217)
gran_1 = Granulation(82.8464, 48.8677)
gran_2 = Granulation(55.8730, 120.2850)
wnoise = WhiteNoise(44.8242)

model = GPModel(bump, gran_1, gran_2, wnoise)
freq, power_dict = model.get_psd(time)

nobump_power = np.zeros(freq.size)
full_power = np.zeros(freq.size)
fig, ax = plt.subplots(figsize=(14, 7))

for name, power in power_dict:
	# TODO: Need alternative to fix white noise psd
	if name == "WhiteNoise":
		power += 0.0
		# power += 44.8242
	if name != "OscillationBump":
		nobump_power += power
	full_power += power

	ax.loglog(freq, power, ls=ls_dict[name], color='b', alpha=alpha_dict[name], label=label_dict[name])

ax.loglog(freq, nobump_power, ls='--', color='r', label='Model without gaussian')
ax.loglog(freq, full_power, ls='-', color='k', label='Full Model')

ax.set_xlim([5, 300])
ax.set_ylim([0.1, 4000])
ax.tick_params(labelsize="large")

ax.set_title('KIC012008916')
ax.set_xlabel(r'Frequency [$\mu$Hz]',fontsize="large")
ax.set_ylabel(r'PSD [ppm$^2$/$\mu$Hz]',fontsize="large")

include_data = True
if include_data:
	# Psd from data
	freq2, power = LombScargle(time/1e6, flux).autopower(nyquist_factor=1, normalization='psd', samples_per_peak=1)
	ax.loglog(freq2, power/time.size, color='k', alpha=0.3)
	
ax.legend(fontsize="large", loc="upper left")
plt.savefig('presentation/psd_no_white.png', dpi = 300)
plt.show()


def psd_data():
	freq2, power = LombScargle(time/1e6, flux).autopower(nyquist_factor=1, normalization='psd', samples_per_peak=1)
	ax.loglog(freq2, power/time.size, color='k', alpha=0.4)