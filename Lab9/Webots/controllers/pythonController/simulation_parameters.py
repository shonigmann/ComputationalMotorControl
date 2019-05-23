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

        self.coupling_weights = 10.0  # w_ij
        self.limb_body_weight = 30.0 #weight from limb to body
        self.body_phase_bias = 0.2*math.pi  # theta_ij
        self.body_limb_phase_bias = math.pi
        self.limb_phase_bias = math.pi

        self.amplitudes_rate = 20.0  # a_i
        self.freqs = 5 # f_i
        self.nominal_amplitudes = 0.5  # R_i for body
        self.nominal_limb_amplitudes = 0.0 # R_i for limbs

        # Parameters for 9d
        self.drive_left = 2.0
        self.drive_right = 2.0

        self.cv_body = [0.2, 0.3] #cv1, cv0
        self.cv_limb = [0.2, 0.0] #cv1, cv0
        self.cR_body = [0.065, 0.196] #cR1, cR0
        self.cR_limb = [0.131, 0.131] #cR1, cR0
        
        self.d_lim_limb = [1.0, 3.0]
        self.d_lim_body = [1.0, 5.0]
        
        self.v_sat = 0.0
        self.R_sat = 0.0
        
        self.use_drive_saturation = 0
        
        self.turn = 0.0
        self.reverse = 0
        self.shes_got_legs = 0

        # Parameters for 9c
        self.is_amplitude_gradient = None
        self.amplitude_gradient = 0.0
        self.smart = 0

        self.enable_transitions = False
        
        # Feel free to add more parameters (ex: MLR drive)
        # self.drive_mlr = ...
        # ...
        # Update object with provided keyword arguments
        self.update(kwargs)  # NOTE: This overrides the previous declarations

        if self.reverse != 0:
            self.body_phase_bias *= -1
            self.limb_phase_bias *= -1
            
        if self.turn != 0.0 and self.turn is not None: #set <0 for a right turn, >0 for a left turn. bounds should be [-2, 2] (or [-1, 1] for walking)
            self.drive_left += self.turn
            self.drive_right -= self.turn
            
        #if self.shes_got_legs != 0:
        #    self.cv_limb = [0.4, 0.0] #cv1, cv0
            
