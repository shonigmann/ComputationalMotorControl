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
        self.amplitude_gradient = 0.0

        self.coupling_weights = 10.0  # w_ij
        self.axialCPG_phase_bias = 2*math.pi/self.n_body_joints  # theta_ij
        self.antiphase_bias = math.pi

        self.amplitudes_rate = 20.0  # a_i
        self.freqs = 5 # f_i
        self.nominal_amplitudes = 0.5  # R_i

        self.smart = 0

        # Feel free to add more parameters (ex: MLR drive)
        # self.drive_mlr = ...
        # ...
        # Update object with provided keyword arguments
        self.update(kwargs)  # NOTE: This overrides the previous declarations

