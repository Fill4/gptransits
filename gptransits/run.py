#!/usr/bin/env python3

''' Filipe Pereira - 2017
Script with user controlled variables to define settings for gptransits
'''

import gptransits
from component import *
from model import *

# User-definable settings
settings = {
	plots = {
		"plot_gp" : False,
		"plot_corner" : False,
		"plot_psd" : True
	},
	"burnin" : 500,
	"iterations" : 2000,
	"nwalkers" : 20
}

#--------------------------------------------------------------
#--------------------------------------------------------------

# Model 2
# gp_model = GPModel(OscillationBump(), Granulation(prior=[[20,200],[10,100]]), Granulation(prior=[[20,200],[90,200]]), WhiteNoise())
# Model 1
# gp_model = GPModel(OscillationBump(), Granulation(), WhiteNoise(prior=[[1,400]]))
# Granulation
# gp_model = GPModel(Granulation())
# White Noise
gp_model = GPModel(WhiteNoise(prior=[[1,400]]))
mean_model = MeanModel()

gptransits.main(mean_model, gp_model, settings)