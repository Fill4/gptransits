#!/usr/bin/env python3

import kplr
import numpy as np
from astropy.stats import LombScargle
import matplotlib.pyplot as plt
import pyfits

def lightcurve(kic):
    client = kplr.API()
    star = client.star(kic)
    print(star.get_light_curves(short_cadence=True))
    star.get_light_curves(short_cadence=False, fetch=True, clobber=True)


hdulist = pyfits.open('RGBensemble/kplr010257278_mast.fits')

#buffer = np.loadtxt('RGBensemble/kplr007060732_d21_v1.dat')
#ntime = buffer[:,0]
#nflux = buffer[:,1]
#nerror = buffer[:,2]

ntime = getattr(hdulist[1].data, 'TIME')
nflux = getattr(hdulist[1].data, 'PDCSAP_FLUX')
nerror = getattr(hdulist[1].data, 'PDCSAP_FLUX_ERR')

ind = np.logical_and(~np.isnan(ntime), ~np.isnan(nflux), ~np.isnan(nerror))
time = ntime[ind]
time = (time - time[0])#*(24*3600)
flux = nflux[ind]/np.median(nflux[ind])
error = nerror[ind]/np.median(nflux[ind])
flux = (flux - 1)*1e6
error = error*1e6

#plt.errorbar(time, flux, yerr=error, fmt=".k", capsize=0)
#plt.show()
#sys.exit()

frequency, power = LombScargle(time, flux).autopower()
fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(111)
ax.semilogx(frequency, power, 'k-', lw=1)
ax.set_title(r'Spectrum of KIC010257278',fontsize=16)
ax.set_xlabel(r'Frequency [days$^{-1}$]',fontsize=14)
ax.set_ylabel(r'Power Spectral Density [ppm]',fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.set_xscale('log')
ax.set_yscale('log')
ax.axvline(49, color='r', lw=2)
ax.axvline(57, color='b', lw=2)
ax.axvline(16, color='g', lw=2)
ax.axvline(71, color='g', lw=2)
plt.savefig('spec_kic010257278.png')