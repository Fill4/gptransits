

from __future__ import division, print_function

import numpy as np
from celerite.terms import SHOTerm

class Granulation(SHOTerm):
    r"""
    A term representing the granualtion of a solar-like oscillator. Uses the SHOTerm kernel from celerite

    Args:
        amplitude: Represented in ppm
        frequency: Represented in $mu$Hz
    """
    input_params = ("Amplitude", "Frequency")
    
    @property
    def Amplitude(self):
        return np.exp(self.log_S0)

    @property
    def Frequency(self):
        return np.exp(self.log_omega0)

    def __init__(self, *args, **kwargs):
        Q = 1/np.sqrt(np.pi)
        if len(args):
            if len(args) != 2:
                raise ValueError("expected {0} arguments but got {1}".format(2, len(args)))
            if len(kwargs):
                raise ValueError("parameters must be fully specified by arguments or keyword arguments, not both")
            
            super().__init__(np.log(args[0]), np.log(Q), np.log(args[1]))
            self.unfrozen_mask[1] = 0
        
        else:
            # Loop over the kwargs and set the parameter values
            params = {}
            for k in self.input_params:
                v = kwargs.pop(k, None)
                if v is None:
                    raise ValueError("missing parameter '{0}'".format(k))
                if k == "Amplitude":
                    params["log_S0"] = np.log(v)
                else:
                    params["log_omega0"] = np.log(v)
            params["log_Q"] = np.log(Q)
            super().__init__(**params)
            self.unfrozen_mask[1] = 0

    def get_parameter_vector(self, include_frozen=False):
        return np.exp(super().get_parameter_vector(include_frozen))

    def set_parameter_vector(self, vector, include_frozen=False):
        super().set_parameter_vector(np.log(vector), include_frozen)

    def __repr__(self):
        return r"Granulation( A: {0.Amplitude} ppm; w0: {0.Frequency} muHz)".format(self)