"""Exercise 9c"""

import numpy as np
from run_simulation import run_simulation
from simulation_parameters import SimulationParameters


def exercise_9c(world, timestep, reset):
    """Exercise 9c"""
    # Parameters
    n_joints = 10
    parameter_set = [
        SimulationParameters(
            simulation_duration=10,
            drive=1,
            nominal_amplitude=0.0,
            amplitude_gradient=rhead - rtail,
            turn=1,
            smart=0,
            # ...
        )
        for rhead in np.linspace(0.0, 0.5, num=5)
        for rtail in np.linspace(0.0, 0.5, num=5)
    ]

    # Grid search
    for simulation_i, parameters in enumerate(parameter_set):
        reset.reset()
        run_simulation(
            world,
            parameters,
            timestep,
            int(1000 * parameters.simulation_duration / timestep),
            logs="./logs/9c/simulation_{}.npz".format(simulation_i)
        )


