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
            simulation_duration=10,
            drive=1,
            amplitudes=test,
            phase_lag=np.ones(n_joints) * test2,
            turn=0,
            # ...
        )
        for test in np.linspace(0.5, 1.3, num=1)
        for test2 in np.linspace(0.25, 0.5, num=1)
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
