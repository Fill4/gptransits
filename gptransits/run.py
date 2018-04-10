#!/usr/bin/env python3

''' Filipe Pereira - 2017
General procedure for fitting light curve data to a model and using
gaussian processes to fit the remaining stellar stochastic signals
'''

import gptransits
from components_v2 import *

# Parameters
plot_flags = {'plot_gp':0, 'plot_corner':0, 'plot_psd':0}

verbose = True

burnin = 200
iterations = 500
nwalkers = 16

#--------------------------------------------------------------
#--------------------------------------------------------------

model = Model(OscillationBump(), Granulation(), Granulation(), WhiteNoise())

data_dir = './results/TEST/'
gptransits.main(data_dir, data_dir, model, plot_flags, nwalkers, iterations, burnin)