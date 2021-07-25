import sys

import numpy as np
from scipy.stats import uniform, norm, reciprocal

import batman
# import pysyzygy as ps

import logging
log = logging.getLogger(__name__)

# Define the transit models
class BatmanModel(object):
	def __init__(self, config):
		self.npars = 9
		self.parameter_names = np.array(['P', 't0', 'Rrat', 'aR', 'cosi', 'e', 'w', "u1", "u2"])
		self.parameter_latex_names = np.array(['Period', 'Epoch', r'$R_p / R_{\star}$', r'a / $R_{\star}$', r'$\cos (i)$', r'$e$', r'$w$', r"$u_1$", r"$u_2$"])
		self.parameter_units = np.array(["days", "days", "", "", "", "", "", "", ""])
		self.mask = np.full(self.npars, True)

		# Get component name
		if 'name' in config:
			self.name = config['name']
		else:
			self.name = 'BatmanTransit'
		
		# Check that we have some parameters to actualy configure the model 
		#TODO: Throw exceptions and return so that we can continue without crashing
		if not 'params' in config:
			log.exception("Can't define transit model. No parameters in the configuration")
			sys.exit()
		# Check that config dictionary as correct number of entries
		if self.npars != len(config['params']["values"].keys()):
			print(f"Transit model needs to have {self.npars} parameters")
			sys.exit()

		if "latex_names" in config['params']:
			if (self.npars != len(config['params']["latex_names"])):
				print(f"Transit parameters latex names must have size {self.npars}")
				sys.exit()
			self.parameter_latex_names = np.array(config['params']["latex_names"])
		if "units" in config['params']:
			if (self.npars != len(config['params']["units"])):
				print(f"Transit parameters units must have size {self.npars}")
				sys.exit()
			self.parameter_units = np.array(config['params']["units"])
		
		# Setup the priors and init parameters
		self.prior = []
		params = config['params']["values"]
		self.pars = np.zeros(self.npars)
		for i, pname in enumerate(self.parameter_names):
			if params[pname][0]:
				self.mask[i] = False
				self.pars[i] = params[pname][1]
			elif params[pname][2] == "uniform":
				self.prior.append(uniform(params[pname][3], params[pname][4] - params[pname][3]))
			elif params[pname][2] == "normal":
				self.prior.append(norm(params[pname][3], params[pname][4]))
			elif params[pname][2] == "reciprocal":
				if not params[pname][3] != 0:
					log.exception("Reciprocal prior cannot have a starting interval value of 0.0")
					sys.exit()
				self.prior.append(reciprocal(params[pname][3], params[pname][4]))
			else:
				log.exception(f"Parameter prior distribution chosen not recognized: {params[pname][2]}")
				sys.exit()
		self.prior = np.asarray(self.prior)
		self.ndim = np.sum(self.mask)

		# Init batman stuff
		self.batpars = batman.TransitParams()
		if self.mask[-1] == False and self.pars[-1] == 0.0:
			self.batpars.limb_dark = 'linear'
		else:
			self.batpars.limb_dark = "quadratic"
		log.debug(f"Transit Model using {self.batpars.limb_dark} limb darkening")

	def get_parameters_names(self):
		return self.parameter_names[self.mask]

	def get_parameters_latex(self):
		return self.parameter_latex_names[self.mask]

	def get_parameters_units(self):
		return self.parameter_units[self.mask]

	def init_model(self, time, cadence, supersample_factor):
		self.update_params(self.sample_prior()[0].T)
		exp_time = cadence - (cadence/supersample_factor)
		self.model = batman.TransitModel(self.batpars, time, supersample_factor=supersample_factor, exp_time=exp_time)

	def sample_prior(self, num=1):
		# return np.array([self.priors[key].rvs(num) for key in self.prior]).T
		return np.hstack([p.rvs([num, 1]) for p in self.prior])

	def update_params(self, params):
		self.pars[self.mask] = params
		self.batpars.per = self.pars[0]
		self.batpars.t0 = self.pars[1]
		self.batpars.rp = self.pars[2]
		self.batpars.a = self.pars[3]
		self.batpars.inc = self.pars[4]
		self.batpars.ecc = self.pars[5]
		self.batpars.w = self.pars[6]
		self.batpars.u = [self.pars[7], self.pars[8]]
		
		# ecosw, esinw
		# self.batpars.w = np.rad2deg(np.arctan(self.pars[5]/self.pars[6]))
		# self.batpars.ecc = np.sqrt(self.pars[5]**2 + self.pars[6]**2)
		# if self.batpars.ecc > 0.95:
		# 	self.batpars.ecc = 0.95

		# impact parameter
		# self.batpars.inc = np.rad2deg(np.arccos(self.pars[4]/self.pars[3]))

		# Kipping parametrization
		# self.batpars.u = [2 * np.sqrt(self.pars[7]) * self.pars[8], np.sqrt(self.pars[7]) * (1 - 2*self.pars[8])]

	def lnprior(self, params):
		# return sum([self.priors[key].logpdf(params[i]) for i, key in enumerate(self.priors)])
		return np.sum([self.prior[i].logpdf(params[i]) for i in range(self.prior.size)])

	def compute(self, params, time):
		self.update_params(params)
		flux = self.model.light_curve(self.batpars)
		return (flux-1)*1e6

