import pyfits
import numpy as np
import sys, os

def read_data(filename, Nmax=0, offset=0, fits_options=None):
	_, ext = os.path.splitext(filename)
	
	if ext == '.fits':
		# Read data from fits and remove nans
		hdulist = pyfits.open(filename)

		ntime = getattr(hdulist[1].data, fits_options['time'])
		nflux = getattr(hdulist[1].data, fits_options['flux'])
		nerror = getattr(hdulist[1].data, fits_options['error'])

		ind = np.logical_and(~np.isnan(ntime), ~np.isnan(nflux), ~np.isnan(nerror))
		time = ntime[ind]
		time = time - time[0]
		flux = nflux[ind]/np.median(nflux[ind])
		error = nerror[ind]/np.median(nflux[ind])
		flux = (flux - 1)*1e6 #to ppm
		error = error*1e6 #to ppm

		return (time, flux, error)

	elif (ext == '.lc') or (ext == '.dat'):
		time, flux, error = np.loadtxt(filename, unpack=True)
		return (time, flux, error)

	else:
		sys.exit('File extension not valid')