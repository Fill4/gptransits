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
	z.write('{:20}  {:^10}  {:^10}  {:^10}  {:^10}  {:^10}  {:^10}  {:^10}\n'.format('Filename', 'Amp_1', 'Tscale_1', 'Amp_2', 'Tscale_2', 'Amp_osc', 'nu_max', 'osc_bump'))
	z.close()

for filename in star_list:
	
	name = os.path.basename(filename)
	# Add filename to buffer and define variable with full path to file
	write_buffer = '{:20}'.format(os.path.basename(name))

	# Start timer
	backend.verboseprint('\n{:_^60}'.format('Starting GP fitting procedure'))
	startTimeScript = timeit.default_timer()

	# Bundle data in tuple for organisation
	if filename.endswith('.fits'):
		data = backend.readfits(filename, Nmax, offset, fits_options)
	else:
		data = backend.readtxt(filename, Nmax, offset)
	#data = (time, flux, error)

	# Initiate prior distributions according to options set by user
	priors = backend.setup_priors(prior_settings)

	# Run run_minimizationn
	#backend.run_minimization(data, priors, plot=plot, module=module)

	# Run MCMC
	final_pars = backend.run_mcmc(data, priors, plot_corner=plot_corner, nwalkers=nwalkers, burnin=burnin, iterations=iterations, module=module)

	for i in range(len(final_pars)):
		write_buffer += '  {:^10.4f}'.format(final_pars[i])
	write_buffer += '\n'

	z = open('Results/' + results_file + '.dat', 'a')
	z.write(write_buffer)
	z.close()

	if plot_corner:
		plt.savefig(name + '_corner.png')

	# Print execution time
	fullTimeScript = timeit.default_timer() - startTimeScript
	backend.verboseprint("\nComplete execution time: {:10.5} usec".format(fullTimeScript))
	backend.verboseprint('\n{:_^60}\n'.format('END'))
