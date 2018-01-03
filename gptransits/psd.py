#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import LombScargle
import sys, os

filepath = 'tda-stars/Star976.clean'
if len(sys.argv) == 2:
	filepath = sys.argv[1]

time, flux, _ = np.loadtxt(filepath, unpack=True)

idx = np.logical_and(~np.isnan(time), ~np.isnan(flux))
time = time[idx]
flux = flux[idx]

time = time - time[0]
time = time*(24*3600)
flux = flux/np.median(flux)
flux = (flux-1)*1e6

nyquist_freq = (1 / (2*(time[1]-time[0])))*1e6

frequency, power = LombScargle(time/1e6, flux).autopower(normalization='psd', minimum_frequency=0.1, nyquist_factor=1, samples_per_peak=1)
power = power/time.size

plt.figure(figsize=(14,8))
plt.plot(time/(24*3600), flux, 'ko', markersize=2)
plt.xlabel('Time [days]')
plt.ylabel('Flux [ppm]')

plt.figure(figsize=(14,8))
plt.loglog(frequency, power, 'k', alpha=0.6)
plt.xlabel(r'Frequency [$\mu$Hz]')
plt.ylabel(r'Power [ppm$^2$]')
plt.show()

mask = np.logical_and(frequency > 0.1, frequency < nyquist_freq)
data_freq = np.column_stack((frequency[mask], power[mask]))
data_time = np.column_stack((time, flux))

filename = os.path.splitext(filepath)[0]
np.savetxt(filename + ".txt", data_freq, fmt="%10.5f %11.5f")
np.savetxt(filename + ".dat", data_time, fmt="%12.3f  %11.5f  -inf")