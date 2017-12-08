#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
import emcee
import celerite
import corner
from astropy.stats import LombScargle

filename = 'tda-stars/Star598.dat'
time, flux, error = np.loadtxt(filename, unpack=True)
error = None

nyquist = (1 / (2*(time[1]-time[0])))*1e6
f_sampling = 1 / (27.4*24*3600 / 1e6)
freq = np.linspace(0.0, nyquist, (nyquist/f_sampling)+1 )


li = np.sqrt(2/np.pi)
xfreq = 2*np.pi
xamp = 1

pars = [78.76, 36.72, 50, 145, 97.05, 102.66, 3.51, 5.7]
pars[0] = (pars[0]**2 /pars[1])*np.sqrt(2)
pars[1] = pars[1]*xfreq
#pars[2] = (pars[2]**2 /pars[3])*np.sqrt(2)
#pars[3] = pars[3]*xfreq
pars[4] = pars[4] / ((pars[6]**2) * li)
pars[5] = pars[5]*xfreq

S1, w1, S2, w2, S_bump, w_bump, Q_bump, sig = np.log(pars)
Q = 1.0 / np.sqrt(2.0)

kernel_1 = celerite.terms.SHOTerm(log_S0=S1, log_Q=np.log(Q), log_omega0=w1)
kernel_1.freeze_parameter("log_Q")
#kernel_2 = celerite.terms.SHOTerm(log_S0=S2, log_Q=np.log(Q), log_omega0=w2)
#kernel_2.freeze_parameter("log_Q")

kernel_bump = celerite.terms.SHOTerm(log_S0=S_bump, log_Q=Q_bump, log_omega0=w_bump)
#kernel_jitter = celerite.terms.JitterTerm(log_sigma=sig)

kernel_gran = kernel_1 #+ kernel_2
kernel = kernel_gran + kernel_bump


power_1 = kernel_1.get_psd(2*np.pi*freq)
#power_2 = kernel_2.get_psd(2*np.pi*freq)
power_bump = kernel_bump.get_psd(2*np.pi*freq)
#power_jitter = kernel_jitter.get_psd(2*np.pi*freq)
#power_jitter += 5.7

nobump_power = kernel_gran.get_psd(2*np.pi*freq)
#nobump_power += power_jitter

full_power = kernel.get_psd(2*np.pi*freq)
#full_power += power_jitter


plt.figure(1, figsize=(12,8))
plt.loglog(freq, full_power, ls='-', color='k')
plt.loglog(freq, nobump_power, ls='--', color='k')
plt.loglog(freq, power_1, ls='--', color='b', alpha=0.8)
#plt.loglog(freq, power_2, ls='--', color='b', alpha=0.8)
plt.loglog(freq, power_bump, ls='--', color='r', alpha=0.8)
#plt.loglog(freq, power_jitter, ls=':', color='b', alpha=0.6)
plt.xlim([-10, 300])
plt.ylim([0.1, 4000])

# Psd from file
#dfreq, dpower = np.loadtxt('sample19_psd/KIC012008916.txt', unpack=True)
#plt.loglog(dfreq, dpower, color='r', alpha=0.5)

# Psd from data
freq, power = LombScargle(time/1e6, flux).autopower(nyquist_factor=1, 
	normalization='psd', samples_per_peak=1)
plt.loglog(freq, power/time.size, color='b', alpha=0.5)


gp = celerite.GP(kernel)
#gp.compute(time/1e6, error)
gp.compute(time/1e6)
x = np.linspace(time[0], time[-1], num=5000)


plt.figure(2, figsize=(12,8))
plt.errorbar(time/(24*3600), flux, yerr=error, fmt=".k", capsize=0, markersize='3', elinewidth=1)

mu, var = gp.predict(flux, x/1e6, return_var=True)
std = np.sqrt(abs(var))
std = np.nan_to_num(std)

plt.plot(x/(24*3600), mu, color='#ff7f0e', linewidth=0.5)
plt.fill_between(x/(24*3600), mu+std, mu-std, color="#ffcc33", alpha=0.6, edgecolor="none", linewidth=0.6)


"""
plt.figure(3, figsize=(12,8))
plt.errorbar(time/(24*3600), flux, yerr=error, fmt=".k", capsize=0, markersize='3', elinewidth=1)
sample = gp.sample_conditional(flux, x/1e6)
plt.plot(x/(24*3600), sample, color='b', linewidth=0.5)
"""

"""
plt.figure(4, figsize=(12,8))
plt.errorbar(time/(24*3600), flux, yerr=error, fmt=".k", capsize=0, markersize='3', elinewidth=1)
nsample = gp.sample()
plt.plot(time/(24*3600), nsample, color='#ff7f0e', linewidth=0.5)
"""

plt.show()