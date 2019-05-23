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
            nominal_amplitudes=0.0,
            is_amplitude_gradient=1,
            rhead=rhead,
            rtail=rtail,
            turn=1,
            # ...
        )
        for rhead in np.linspace(0.0, 0.25, num=2)
        for rtail in np.linspace(0.0, 0.5, num=2)
    ]

    # Grid search
    for simulation_i, parameters in enumerate(parameter_set):
        reset.reset()
        run_simulation(
            world,
            parameters,
            timestep,
            int(1000 * parameters.simulation_duration / timestep),
            logs="./logs/9c/random/simulation_{}.npz".format(simulation_i)
        )

