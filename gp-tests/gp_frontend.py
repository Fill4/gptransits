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
import gp_backend

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
if os.path.exists(results_file + '.results'):
    pass
else:
	z = open(results_file + '.results', 'w')
	z.write('{:30}{:^16}{:^16}{:^16}\n'.format('Filename', 'Amplitude', 'Timescale', 'Jitter'))
	z.close()

for file in star_list:
	
	# Add filename to buffer and define variable with full path to file
	write_buffer = '{:30}'.format(file)
	filename = 'RGBensemble/' + file

	# Start timer
	gp_backend.verboseprint('\n{:_^60}'.format('Starting GP fitting procedure'))
	startTimeScript = timeit.default_timer()

	# Bundle data in tuple for organisation
	if filename.endswith('.fits'):
		data = gp_backend.readfits(filename, Nmax)
	else:
		data = gp_backend.readtxt(filename, Nmax)
	#data = (time, flux, error)

	# Initiate prior distributions according to options set by user
	priors = gp_backend.setup_priors(prior_settings)

	# Run minimization
	gp_backend.run_minimization(data, priors, plot=plot, module=module)

	# Run MCMC
	final_pars = gp_backend.run_mcmc(data, priors, plot=plot, nwalkers=nwalkers, burnin=burnin, iterations=iterations, module=module)

	write_buffer += '{:^16.6f}{:^16.6f}{:^16.6f}\n'.format(final_pars[0],final_pars[1],final_pars[2])
	#write_buffer +=('{:10.6f}{:10}{:10.6f}\n'.format(final_pars[0],'',final_pars[1]))
	z = open(results_file + '.results', 'a')
	z.write(write_buffer)
	z.close()

	if plot:
		for i in plt.get_fignums():
			plt.figure(i)
			plt.savefig('Figures/%s%s%d.png' % (os.path.splitext(os.path.basename(filename))[0], '_fig', i), dpi = 200)
			plt.clf()
			plt.cla()
			plt.close()


	# Print execution time
	fullTimeScript = timeit.default_timer() - startTimeScript
	gp_backend.verboseprint("\nComplete execution time: {:10.5} usec".format(fullTimeScript))
	gp_backend.verboseprint('\n{:_^60}\n'.format('END'))
