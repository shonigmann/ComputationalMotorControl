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
            simulation_duration=15,
#            nominal_amplitudes=test,
            drive_right = test,
            drive_left = test,
            body_phase_bias=test2,
            use_drive_saturation = 1,
            turn=0,
            # ...
        )
        for test in np.linspace(4.0, 5.0, num=6)
            #As seen in Ijspeer paper fig.1.B. 
            
        for test2 in np.linspace(2*np.pi/10-0.2, 2*np.pi/10+0.2, num=7)
            #We know nature is set to 2*pi/length. So we could test from lower than 2*pi/lengh to above of it. 
            #Different proposals with 2*pi/lenght on the center value.
    ]

    # Grid search
    for simulation_i, parameters in enumerate(parameter_set):
        reset.reset()
        run_simulation(
            world,
            parameters,
            timestep,
            int(1000 * parameters.simulation_duration / timestep),
            logs="./logs/9b/test_{}.npz".format(simulation_i)
        )

