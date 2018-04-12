#!/usr/bin/env python3

''' Filipe Pereira - 2017
General procedure for fitting light curve data to a model and using
gaussian processes to fit the remaining stellar stochastic signals
'''

import gptransits
from component import *
from model import GPModel

# Parameters
plot_flags = {'plot_gp':0, 'plot_corner':0, 'plot_psd':0}

verbose = True

burnin = 500
iterations = 4000
nwalkers = 20

#--------------------------------------------------------------
#--------------------------------------------------------------

model = GPModel(OscillationBump(), Granulation(prior=[[20,200],[10,100]]), Granulation(prior=[[20,200],[90,200]]), WhiteNoise())

data_dir = './results/TEST/'
gptransits.main(data_dir, data_dir, model, plot_flags, nwalkers, iterations, burnin)