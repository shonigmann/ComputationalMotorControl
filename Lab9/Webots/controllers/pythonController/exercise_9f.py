"""Exercise 9f"""

import numpy as np
from run_simulation import run_simulation
from simulation_parameters import SimulationParameters
import plot_results

def exercise_9f(world, timestep, reset):
    """Exercise 9f"""
    
    n_joints = 10
    
    parameter_set = [
        SimulationParameters(
            simulation_duration=10,
            use_drive_saturation=1,
            shes_got_legs=1,
            #cR_body = [0.05, 0.16], #cR1, cR0
            #cR_limb = [0.131, 0.131], #cR1, cR0
            #amplitudes_rate = 40,
            body_drive_left = 2.0,
            body_drive_right = 2.0,
            limb_drive_left = 2.0,
            limb_drive_right = 2.0,
            # ...
        )
        #for turn in np.linspace(-.2,.2, 7)
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
            logs="./logs/9f/simulation_{}.npz".format(simulation_i)
        )

    #plot_results.plot_turn_trajectory()
