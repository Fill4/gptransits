import numpy as np
import celerite

from .component import Granulation, OscillationBump, WhiteNoise

import logging
log = logging.getLogger(__name__)

class GPModel(object):
	def __init__(self, config):
		self.component_vector = []
		# If there is a configuration list, init all the components
		for i in range(len(config)):
			try:
				component_type = config[i]["type"]
			except KeyError:
				log.exception("Need component type to define component")
				sys.exit()
			if component_type == "granulation":
				self.component_vector.append(Granulation(config[i]))
			elif component_type == "oscillation_bump":
				self.component_vector.append(OscillationBump(config[i]))
			elif component_type == "white_noise":
				self.component_vector.append(WhiteNoise(config[i]))
			else:
				log.exception(f"Component type defined not recognized: {component_type}")
				sys.exit()
		self.component_vector = np.asarray(self.component_vector)
		self.ndim = np.sum([component.npars for component in self.component_vector])

	def get_component_names(self):
		return np.array([component.name for component in self.component_vector])

	def get_parameters_celerite(self, params):
		i = 0
		celerite_params = np.array([])
		for component in self.component_vector:
			celerite_params = np.append(celerite_params, component.get_parameters_celerite(params[i:i+component.npars]))
			i += component.npars
		return celerite_params

	def get_parameters_names(self):
		return np.hstack([component.parameter_names for component in self.component_vector])

	def get_parameters_latex(self):
		return np.hstack([component.parameter_latex_names for component in self.component_vector])

	def get_parameters_units(self):
		return np.hstack([component.parameter_units for component in self.component_vector])

	def lnprior(self, params):
		lnprior = 0 
		i = 0
		for component in self.component_vector:
			lnprior += component.lnprior(params[i:i+component.npars])
			i += component.npars
		return lnprior

	def sample_prior(self, num=1):
		return np.hstack([component.sample_prior(num) for component in self.component_vector])

	def get_kernel(self, params):
		kernel = celerite.terms.TermSum()
		i = 0
		for component in self.component_vector:			
			kernel += component.get_kernel(params[i:i+component.npars])
			i += component.npars
		return kernel

	def get_psd(self, params, time, min_freq=0.0, nyquist_mult=1):

		days_to_microsec = (24*3600) / 1e6
		cadence = (time[1] - time[0]) * days_to_microsec
		nyquist = 1 / (2 * cadence)

		time_span = (time[-1] - time[0]) * days_to_microsec
		f_sampling = 1 / time_span

		freq = np.linspace(min_freq, nyquist*nyquist_mult, int((nyquist*nyquist_mult-min_freq)/f_sampling)+1)

		i = 0
		psd_vector = []
		for component in self.component_vector:
			psd_vector.append(component.get_psd(params[i:i+component.npars], freq, time.size, nyquist*nyquist_mult))
			i += component.npars
		return freq, np.asarray(psd_vector)
		