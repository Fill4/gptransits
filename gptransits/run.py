#!/usr/bin/env python3

''' Filipe Pereira - 2017
General procedure for fitting light curve data to a model and using
gaussian processes to fit the remaining stellar stochastic signals
'''

import gptransits
from component import *
from model import *

# Parameters
plot_flags = {'plot_gp':0, 'plot_corner':0, 'plot_psd':1}

verbose = True

burnin = 500
iterations = 2000
nwalkers = 20

#--------------------------------------------------------------
#--------------------------------------------------------------

# model = GPModel(OscillationBump(), Granulation(prior=[[20,200],[10,100]]), Granulation(prior=[[20,200],[90,200]]), WhiteNoise())
gp_model = GPModel(WhiteNoise(prior=[[1, 200]]))
mean_model = MeanModel()


data_dir = './results/TEST/'
gptransits.main(mean_model, gp_model, plot_flags, nwalkers, iterations, burnin)