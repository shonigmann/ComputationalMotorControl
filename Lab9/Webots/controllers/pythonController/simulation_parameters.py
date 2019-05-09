"""Simulation parameters"""
import numpy as np
import math

class SimulationParameters(dict):
    """Simulation parameters"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self, **kwargs):
        super(SimulationParameters, self).__init__()
        # Default parameters
        self.n_body_joints = 10
        self.n_legs_joints = 4
        self.simulation_duration = 30
        self.phase_lag = None
        self.amplitude_gradient = None

        vec_size = 2*self.n_body_joints + self.n_legs_joints

        self.coupling_weights = np.zeros([vec_size, vec_size])  # w_ij
        self.phase_bias = np.zeros([vec_size, vec_size])  # theta_ij

        self.amplitudes_rate = 20.0  # a_i
        self.freqs = np.ones(vec_size)*5 # f_i
        self.nominal_amplitudes = np.ones(vec_size)*0.5  # R_i

        # Try something else
        for i in range(self.n_body_joints):
            if i != self.n_body_joints-1:
                self.coupling_weights[i, i + 1] = 10
                self.coupling_weights[i + self.n_body_joints, i + self.n_body_joints + 1] = 10

                self.coupling_weights[i + 1, i] = 10
                self.coupling_weights[i + self.n_body_joints + 1, i + self.n_body_joints] = 10

                self.phase_bias[i, i + 1] = -0.2*math.pi
                self.phase_bias[i + self.n_body_joints, i + self.n_body_joints + 1] = -0.2*math.pi

                self.phase_bias[i + 1, i] = 0.2*math.pi
                self.phase_bias[i + self.n_body_joints + 1, i + self.n_body_joints] = 0.2*math.pi

            self.coupling_weights[i, i + self.n_body_joints] = 10
            self.coupling_weights[i + self.n_body_joints, i] = 10

            self.phase_bias[i, i + self.n_body_joints] = math.pi
            self.phase_bias[i + self.n_body_joints, i] = math.pi

        # Feel free to add more parameters (ex: MLR drive)
        # self.drive_mlr = ...
        # ...
        # Update object with provided keyword arguments
        self.update(kwargs)  # NOTE: This overrides the previous declarations

