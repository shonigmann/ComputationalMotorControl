"""Oscillator network ODE"""

import numpy as np

from solvers import euler, rk4
from robot_parameters import RobotParameters


def network_ode(_time, state, parameters):
    """Network_ODE

    returns derivative of state (phases and amplitudes)

    """
    phases = state[:parameters.n_oscillators]
    amplitudes = state[parameters.n_oscillators:2*parameters.n_oscillators]

    weights_size = parameters.coupling_weights[0].size
    d_phases = np.zeros(weights_size)
    for i in range(weights_size):
        d_phases[i] = 2*np.pi*parameters.freqs[i] + np.sum(amplitudes * parameters.coupling_weights[i] +
                                                           np.sin(phases[:] - phases[i] - parameters.phase_bias[i]))

    d_amplitudes = parameters.amplitudes_rate * (parameters.nominal_amplitudes - amplitudes)

    return np.concatenate([d_phases, d_amplitudes])


def motor_output(phases, amplitudes):
    """Motor output"""
    nb_body_joints = 10
    nb_legs_joints = 4
    q = np.zeros(nb_body_joints + nb_legs_joints)

    for i in range(nb_body_joints):
        q[i] = amplitudes[i] * (1 + np.cos(phases[i])) - amplitudes[i+10] * (1 + np.cos(phases[i + 10]))

    return q


class ODESolver(object):
    """ODE solver with step integration"""

    def __init__(self, ode, timestep, solver=rk4):
        super(ODESolver, self).__init__()
        self.ode = ode
        self.solver = solver
        self.timestep = timestep
        self._time = 0

    def integrate(self, state, *parameters):
        """Step"""
        diff_state = self.solver(
            self.ode,
            self.timestep,
            self._time,
            state,
            *parameters
        )
        self._time += self.timestep
        return diff_state

    def time(self):
        """Time"""
        return self._time


class RobotState(np.ndarray):
    """Robot state"""

    def __init__(self, *_0, **_1):
        super(RobotState, self).__init__()
        self[:] = 0.0

    @classmethod
    def salamandra_robotica_2(cls):
        """State of Salamandra robotica 2"""
        return cls(2*24, dtype=np.float64, buffer=np.zeros(2*24))

    @property
    def phases(self):
        """Oscillator phases"""
        return self[:24]

    @phases.setter
    def phases(self, value):
        self[:24] = value

    @property
    def amplitudes(self):
        """Oscillator phases"""
        return self[24:]

    @amplitudes.setter
    def amplitudes(self, value):
        self[24:] = value


class SalamanderNetwork(ODESolver):
    """Salamander oscillator network"""

    def __init__(self, timestep, parameters):
        super(SalamanderNetwork, self).__init__(
            ode=network_ode,
            timestep=timestep,
            solver=rk4  # Feel free to switch between Euler (euler) or
                        # Runge-Kutta (rk4) integration methods
        )
        # States
        self.state = RobotState.salamandra_robotica_2()
        # Parameters
        self.parameters = RobotParameters(parameters)
        # Set initial state
        self.state.phases = 1e-4*np.random.ranf(self.parameters.n_oscillators)

    def step(self):
        """Step"""
        self.state += self.integrate(self.state, self.parameters)

    def get_motor_position_output(self):
        """Get motor position"""
        return motor_output(self.state.phases, self.state.amplitudes)

