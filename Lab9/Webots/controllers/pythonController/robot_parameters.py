"""Robot parameters"""

import math
import numpy as np
import cmc_pylog as pylog


class RobotParameters(dict):
    """Robot parameters"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self, parameters):
        super(RobotParameters, self).__init__()

        # Initialise parameters
        self.n_body_joints = parameters.n_body_joints
        self.n_legs_joints = parameters.n_legs_joints
        self.n_joints = self.n_body_joints + self.n_legs_joints

        self.n_oscillators_body = 2*self.n_body_joints
        self.n_oscillators_legs = self.n_legs_joints
        self.n_oscillators = self.n_oscillators_body + self.n_oscillators_legs

        self.freqs = np.zeros(self.n_oscillators)
        self.coupling_weights = np.zeros([self.n_oscillators, self.n_oscillators])
        self.phase_bias = np.zeros([self.n_oscillators, self.n_oscillators])

        self.rates = np.zeros(self.n_oscillators)
        self.nominal_amplitudes = np.zeros(self.n_oscillators)
        self.update(parameters)

    def update(self, parameters):
        """Update network from parameters"""
        self.set_frequencies(parameters)  # f_i
        self.set_coupling_weights(parameters)  # w_ij
        self.set_phase_bias(parameters)  # theta_i
        self.set_amplitudes_rate(parameters)  # a_i
        self.set_nominal_amplitudes(parameters)  # R_i

    def set_frequencies(self, parameters):
        """Set frequencies"""
        frequency = parameters.freqs
        self.freqs = frequency * np.ones(2 * self.n_body_joints + self.n_legs_joints)

    def set_coupling_weights(self, parameters):
        """Set coupling weights"""
        weight = parameters.coupling_weights

        for i in range(self.n_body_joints):
            if i != self.n_body_joints-1:
                self.coupling_weights[i, i + 1] = weight
                self.coupling_weights[i + self.n_body_joints, i + self.n_body_joints + 1] = weight

                self.coupling_weights[i + 1, i] = weight
                self.coupling_weights[i + self.n_body_joints + 1, i + self.n_body_joints] = weight

            self.coupling_weights[i, i + self.n_body_joints] = weight
            self.coupling_weights[i + self.n_body_joints, i] = weight

    def set_phase_bias(self, parameters):
        """Set phase bias"""
        body_bias = parameters.axial_cpg_phase_bias
        limb_bias = parameters.limb_antiphase_bias

        for i in range(self.n_body_joints):
            if i != self.n_body_joints-1:
                self.phase_bias[i, i + 1] = -body_bias
                self.phase_bias[i + self.n_body_joints, i + self.n_body_joints + 1] = -body_bias

                self.phase_bias[i + 1, i] = body_bias
                self.phase_bias[i + self.n_body_joints + 1, i + self.n_body_joints] = body_bias

            self.phase_bias[i, i + self.n_body_joints] = limb_bias
            self.phase_bias[i + self.n_body_joints, i] = limb_bias

    def set_amplitudes_rate(self, parameters):
        """Set amplitude rates"""
        self.amplitudes_rate = parameters.amplitudes_rate

    def set_nominal_amplitudes(self, parameters):
        """Set nominal amplitudes"""
        amplitude = parameters.nominal_amplitudes
        self.nominal_amplitudes = amplitude * np.ones(2 * self.n_body_joints + self.n_legs_joints)

