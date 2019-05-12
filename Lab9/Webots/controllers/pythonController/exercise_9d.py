"""Exercise 9d"""
import numpy as np
from run_simulation import run_simulation
from simulation_parameters import SimulationParameters
import math
import plot_results


def exercise_9d1(world, timestep, reset):
    """Exercise 9d1"""
    n_joints = 10
    parameter_set = [
        SimulationParameters(
            simulation_duration=10,
            use_drive_saturation=1,
            turn=0.1,
            # ...
        )
    ]

    # Grid search
    logs =""
    for simulation_i, parameters in enumerate(parameter_set):
        reset.reset()
        run_simulation(
            world,
            parameters,
            timestep,
            int(1000 * parameters.simulation_duration / timestep),
            logs="./logs/9d1/simulation_{}.npz".format(simulation_i)
        )

    plot_results.main("./logs/9d1/simulation_{}.npz".format(simulation_i))

def exercise_9d2(world, timestep, reset):
    """Exercise 9d2"""
    n_joints = 10
    parameter_set = [
        SimulationParameters(
            simulation_duration=5,
            use_drive_saturation=1,
            reverse = 1,
            # ...
        )
    ]

    # Grid search
    logs = ""
    for simulation_i, parameters in enumerate(parameter_set):
        reset.reset()
        run_simulation(
            world,
            parameters,
            timestep,
            int(1000 * parameters.simulation_duration / timestep),
            logs="./logs/9d2/simulation_{}.npz".format(simulation_i)
        )

    plot_results.main(logs)
    #TODO: Plot GPS Coordinates
    #TODO: Plot Spine Angles
    #TODO: Maybe compare spine angles to swimming forward?

