"""Exercise 9b"""

import numpy as np
import plot_results
import math

from run_simulation import run_simulation
from simulation_parameters import SimulationParameters


def exercise_9b(world, timestep, reset):
    """Exercise 9b"""
    # Parameters
    n_joints = 10
    parameter_set = [
        SimulationParameters(
            simulation_duration=5,
            drive=1,
            nominal_amplitudes=nominal_amplitudes,
            body_phase_bias=body_phase_bias,
            turn=1,
            # ...
        )
        for nominal_amplitudes in np.linspace(0.1, 1, num=3)
        for body_phase_bias in np.linspace(0.01*math.pi, 1*math.pi, num=3)
    ]

    # Grid search
    for simulation_i, parameters in enumerate(parameter_set):
        reset.reset()
        run_simulation(
            world,
            parameters,
            timestep,
            int(1000 * parameters.simulation_duration / timestep),
            logs="./logs/9b/simulation_{}.npz".format(simulation_i)
        )
