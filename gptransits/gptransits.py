#!/usr/bin/env python3

''' Filipe Pereira - 2017
General procedure for fitting light curve data to a model and using
gaussian processes to fit the remaining stochastic noise
'''

import numpy as np
import matplotlib.pyplot as plt
# import scipy.stats
import timeit
import sys, os
import argparse
import logging

#Internal imports
# from backend import setup_priors
from model import *
from mcmc import *
import plot

# def main(dataFolder, resultsFolder, model, prior_settings, plot_flags, nwalkers, iterations, burnin):
def main(mean_model, gp_model, plot_flags, nwalkers, iterations, burnin):
	
	# Log start time
	init_time = timeit.default_timer()

	#------------------------------------------------------------------
	#	INPUT
	#------------------------------------------------------------------

	# Handle the input flags of the script
	parser = argparse.ArgumentParser(description='Parse arguments')
	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument('--file', '-f', type=str, dest='input_file', help='Filename with lightcurve data', required=False)
	group.add_argument('--list', '-l', type=str, dest='input_list', help='Filename with list of lightcurve files', required=False)
	parser.add_argument('--output', '-o', type=str, dest='output', help='Folder where to write the results. Folder will be created if it doesnt exist')
	parser.add_argument('--verbose', '-v', type=bool, default=True, dest='verbose', help='Enable verbose mode')
	args = parser.parse_args()

	if args.verbose:
		logging.basicConfig(format='%(message)s', level=logging.INFO)

	if args.input_file:
		filename = os.path.splitext(os.path.basename(args.input_file))[0]
	else:
		pass
	logging.info('Starting {:} ...'.format(filename))

	if args.output:
		if not os.path.isdir(args.output):
			os.mkdir(args.output)

	#------------------------------------------------------------------
	#	MAIN - TODO:Separate from arg parsing
	#------------------------------------------------------------------

	# Read data from input file and instanciate the GP using the time array and model
	data = np.loadtxt(args.input_file, unpack=True)
	model = Model(mean_model, gp_model, data)

	# Run Minimizationn
	#backend.run_minimization(data, priors, plot=plot)

	# Run MCMC
	samples, results = mcmc(model, nwalkers=nwalkers, iterations=iterations, burnin=burnin)

	# Replace model and gp parameters with median from results
	model.gp.gp_model.set_parameters(results[1])
	model.gp.set_parameters()

	# If verbose mode, display the values obtained for each parameter
	params, names = (model.gp.gp_model.get_parameters(), model.gp.gp_model.get_parameters_names())
	logging.info(''.join(["{:10}{:3}{:10.4f}\n".format(names[i], "-", params[i]) for i in range(len(params))]))

	#------------------------------------------------------------------
	#	OUTPUT
	#------------------------------------------------------------------

	if args.output:
		results_file = '{}/{}.out'.format(args.output, filename)
		if not os.path.exists(results_file):
			z = open(results_file, 'w')
		else:
			z = open(results_file, 'a')	
		
		# Write final parameters and uncertainties to output buffer
		output_buffer = '{:<12} {:}'.format('Filename:', filename)
		output_buffer += ''.join(['{:<12}'.format(parameter_name for parameter_name in model.gp.gp_model.get_parameters_names())])

		output_buffer += '\n'

		# Write buffer to results file
		
		z.write(output_buffer)
		z.close()

	#------------------------------------------------------------------
	#	PLOT
	#------------------------------------------------------------------
	
	if plot_flags['plot_gp']:
		plot.plot_gp(model, data)
	if plot_flags['plot_corner']:
		plot.plot_corner(model, samples)
	if plot_flags['plot_psd']:
		plot.plot_psd(model, data)
	if any(plot_flags.values()):
		plt.show()
		plt.close('all')

	# Print execution time
	execution_time = timeit.default_timer() - init_time
	logging.info("Complete execution time: {:.4f} usec".format(execution_time))
	logging.info('End\n')



# Loop to run all Diamonds stars. Need to move it elsewhere
"""
def diamondsRunAll():
	home = '/mnt/c/Users/Filipe/Downloads/work/phd'
	quarters = ['Q12', 'Q13', 'Q14', 'Q15', 'Q16']
	intervals = ['1', '2']
	for quarter in quarters:
		for interval in intervals:
			dataFolder = '{}/Data Diamonds GPs/{}.{}/lightcurve/'.format(home, quarter, interval)
			#resultsFolder = '{}/gptransits/gptransits/results/{}.{}_model1_new'.format(home, quarter, interval)
			resultsFolder = '{}/gptransits/gptransits/results/TEST'.format(home)
			print('Quarter {}.{} ...\n'.format(quarter, interval))
			main(dataFolder, resultsFolder)
"""