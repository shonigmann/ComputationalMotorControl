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
        
        self.body_drive_left = 0.0
        self.body_drive_right = 0.0
        self.limb_drive_left = 0.0
        self.limb_drive_right = 0.0
        
        self.v_sat = 0.0
        self.R_sat = 0.0
        self.cv_body = [0,0]
        self.cv_limb = [0,0]
        self.cR_body = [0,0]
        self.cR_limb = [0,0]
        self.d_lim_body = [0,0]
        self.d_lim_limb = [0,0]
        
        self.use_drive_saturation = 0
        
        self.update(parameters)

    def update(self, parameters):
        """Update network from parameters"""
        self.set_frequencies(parameters)  # f_i
        self.set_coupling_weights(parameters)  # w_ij
        self.set_phase_bias(parameters)  # theta_i
        self.set_amplitudes_rate(parameters)  # a_i
        self.set_nominal_amplitudes(parameters)  # R_i
        self.set_drive_rates(parameters) # d
        self.set_saturation_params(parameters) # dlow, dhigh, cv1, cv0, cR1, cR0, Rsat
        self.saturate_params()

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
        body_bias = parameters.body_phase_bias
        limb_bias = parameters.limb_phase_bias

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
        self.nominal_amplitudes[2*self.n_body_joints:2*self.n_body_joints+self.n_legs_joints] = parameters.nominal_limb_amplitudes *np.ones(self.n_legs_joints)

    def set_drive_rates(self, parameters):
        self.body_drive_left = parameters.body_drive_left
        self.body_drive_right = parameters.body_drive_right
        self.limb_drive_left = parameters.limb_drive_left
        self.limb_drive_right = parameters.limb_drive_right
        
    def set_saturation_params(self, parameters):
        self.cv_body = parameters.cv_body #cv1, cv0
        self.cv_limb = parameters.cv_limb #cv1, cv0
        self.cR_body = parameters.cR_body #cR1, cR0
        self.cR_limb = parameters.cR_limb #cR1, cR0
        self.R_sat = parameters.R_sat 
        self.v_sat = parameters.v_sat
        self.d_lim_body = parameters.d_lim_body #dlow, dhigh
        self.d_lim_limb = parameters.d_lim_limb #dlow, dhigh
        self.use_drive_saturation = parameters.use_drive_saturation
        
    def saturate_params(self):
        if self.use_drive_saturation != 0:
            if self.body_drive_left <= self.d_lim_body[1] and self.body_drive_left > self.d_lim_body[0]:
                leftfreqs = self.cv_body[0]*self.body_drive_left+self.cv_body[1]
                leftAmps = self.cR_body[0]*self.body_drive_left + self.cR_body[1]
            else:
                leftfreqs = self.v_sat
                leftAmps = self.R_sat
            
            if self.body_drive_right <= self.d_lim_body[1] and self.body_drive_right > self.d_lim_body[0]:
                rightfreqs = self.cv_body[0]*self.body_drive_right+self.cv_body[1]
                rightAmps = self.cR_body[0]*self.body_drive_right + self.cR_body[1]
            else:
                rightfreqs = self.v_sat
                rightAmps = self.R_sat
            
            if self.limb_drive_left <= self.d_lim_limb[1] and self.limb_drive_left > self.d_lim_limb[0]:
                llfreqs = self.cv_limb[0]*self.limb_drive_left+self.cv_limb[1]
                llAmps = self.cR_body[0]*self.limb_drive_left + self.cR_body[1]
            else:
                llfreqs = self.v_sat
                llAmps = self.R_sat
            
            if self.limb_drive_right <= self.d_lim_body[1] and self.limb_drive_right > self.d_lim_body[0]:
                rlfreqs = self.cv_limb[0]*self.limb_drive_right+self.cv_limb[1]
                rlAmps = self.cR_body[0]*self.limb_drive_right + self.cR_body[1]
            else:
                rlfreqs = self.v_sat
                rlAmps = self.R_sat
            
            #note: Figure 1 lists the leg order as FL, FR, BL, BR which goes against the F->B, then L->R that was used in the spine. The spine notation will be used here for now
            self.freqs = np.concatenate([leftfreqs*np.ones(self.n_body_joints), rightfreqs*np.ones(self.n_body_joints), llfreqs*np.ones(self.n_legs_joints//2), rlfreqs*np.ones(self.n_legs_joints//2)])
            self.nominal_amplitudes = np.concatenate([leftAmps*np.ones(self.n_body_joints), rightAmps*np.ones(self.n_body_joints), llAmps*np.ones(self.n_legs_joints//2), rlAmps*np.ones(self.n_legs_joints//2)])
