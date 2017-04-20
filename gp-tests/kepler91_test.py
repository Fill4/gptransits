import redgiant
import pyfits
import numpy as np
import matplotlib.pyplot as plt
import sys

hdulist = pyfits.open("Kepler91.fits")

Nmax = 500 #len(hdulist[1].data.TIME)/2
plot = False

time = hdulist[1].data.TIME[:Nmax]
flux = hdulist[1].data.PDCSAP_FLUX[:Nmax]
error = hdulist[1].data.SAP_FLUX_ERR[:Nmax]

ind = np.logical_and(~np.isnan(time), ~np.isnan(flux))
ntime = time[ind]
nflux = flux[ind]
nerror = error[ind]

nflux = nflux - np.mean(nflux)

if plot:
	#plt.errorbar(ntime, nflux, yerr=nerror, fmt=".k", capsize=0.1, markersize=0.5, label= 'Flux')
	plt.scatter(ntime, nflux, s=0.3)
	plt.show()
	sys.exit()

dataTuple = (ntime, nflux, nerror)
redgiant.main(dataTuple, plot=True)