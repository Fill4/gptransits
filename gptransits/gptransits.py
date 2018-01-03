#!/usr/bin/env python3

''' Filipe Pereira - 2017
General procedure for fitting light curve data to a model and using
gaussian processes to fit the remaining stochastic noise
'''

from config_file import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import timeit
import sys, os, os.path
import backend
from backend import vprint

star_list = []
try:
	star_list.append(filename)
except NameError:
	try:
		f = open(filelist)
	except NameError:
		sys.exit('Neither filename nor filelist defined. Stopping!')
	else:	
		for file in f.readlines():
			if not (file.startswith('#')):
				star_list.append(file.rstrip())

# If there is no file , create it
if os.path.exists('Results/' + results_file + '.dat'):
    pass
else:
	z = open('Results/' + results_file + '.dat', 'w')
	z.write('#{:17}  {:^10}  {:^10}  {:^10}  {:^10}  {:^10}  {:^10}  {:^10}\n'.format('Filename', 'Amp_1', 'Tscale_1', 'Amp_2', 'Tscale_2', 'Amp_osc', 'nu_max', 'osc_bump'))
	z.close()

for filename in star_list:
	
	name = os.path.splitext(os.path.basename(filename))[0]
	# Add filename to buffer and define variable with full path to file
	write_buffer = '{:20}'.format(name)

	# Start timer
	vprint('Starting GP fitting procedure ...')
	itime_script = timeit.default_timer()

	# Bundle data in tuple for organisation
	data = backend.read_data(filename, Nmax, offset, fits_options)

	# Initiate prior distributions according to options set by user
	priors = backend.setup_priors(prior_settings)

	# Run run_minimizationn
	#backend.run_minimization(data, priors, plot=plot)

	# Run MCMC
	final_params = backend.run_mcmc(data, priors, plot_corner=plot_corner, plot_sample=plot_sample, nwalkers=nwalkers, iterations=iterations)

	# Write final params from mcmc to buffer
	for i in range(len(final_params)):
		write_buffer += '  {:^10.4f}'.format(final_params[i])
	write_buffer += '\n'

	# Write buffer to results file
	z = open('Results/' + results_file + '.dat', 'a')
	z.write(write_buffer)
	z.close()

	if plot_sample:
		plt.figure(1)
		plt.savefig('Results/' + name + '_sample.png')
	if plot_corner:
		plt.figure(2)
		plt.savefig('Results/' + name + '_corner.png')

	# Print execution time
	time_script = timeit.default_timer() - itime_script
	vprint("Complete execution time: {:.4f} usec".format(time_script))
	vprint('END')
