"""Exercise 9d"""
import numpy as np
from run_simulation import run_simulation
from simulation_parameters import SimulationParameters
import plot_results


def exercise_9d1(world, timestep, reset):  
    parameter_set = [
        SimulationParameters(
            simulation_duration=10,
            use_drive_saturation=1,
            shes_got_legs=1,
            cR_body = [0.05, 0.16], #cR1, cR0
            cR_limb = [0.131, 0.131], #cR1, cR0
            amplitudes_rate = 40,
            drive_left = 3.3,
            drive_right = 3.3,
            turn=turn
            # ...
        )
        for turn in np.linspace(-.2,.2, 7)
    ]

    # Grid search
    #logs =""
    for simulation_i, parameters in enumerate(parameter_set):
        reset.reset()
        run_simulation(
            world,
            parameters,
            timestep,
            int(1000 * parameters.simulation_duration / timestep),
            logs="./logs/9d1/simulation_{}.npz".format(simulation_i)
        )

def exercise_9d2(world, timestep, reset):
    """Exercise 9d2"""
    parameter_set = [
        SimulationParameters(
            simulation_duration=10,
            use_drive_saturation=1,
            reverse = reverse,
            turn = turn,
            shes_got_legs = 1,
            cR_body = [0.05, 0.16], #cR1, cR0
            cR_limb = [0.131, 0.131], #cR1, cR0
            drive_left = 3.3,
            drive_right = 3.3,
            # ...
        )
        for turn in np.linspace(-.2,.2, 3)            
        for reverse in np.linspace(0,1,2)
    ]

    # Grid search
    for simulation_i, parameters in enumerate(parameter_set):
        reset.reset()
        run_simulation(
            world,
            parameters,
            timestep,
            int(1000 * parameters.simulation_duration / timestep),
            logs="./logs/9d2/simulation_{}.npz".format(simulation_i)
        )

    plot_results.plot9d2()
    #TODO: Plot GPS Coordinates
    #TODO: Plot Spine Angles
    #TODO: Maybe compare spine angles to swimming forward?