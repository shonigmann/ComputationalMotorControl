"""Exercise 9f"""

import numpy as np
from run_simulation import run_simulation
from simulation_parameters import SimulationParameters
import plot_results
import math

def exercise_9f(world, timestep, reset):
    """Exercise 9f"""
    
    n_joints = 10
    
    parameter_set = [
        SimulationParameters(
            simulation_duration=15,
            use_drive_saturation=1,
            shes_got_legs=1,
            cR_body = [0.05, 0.16], #cR1, cR0
            cR_limb = [0.131, 0.131], #cR1, cR0
            amplitudes_rate = 40,
            drive_left = 1.6,
            drive_right = 1.6,
            body_limb_phase_bias = phase_bias
            # ...
        )
        for phase_bias in np.linspace(0,2*math.pi, 9)
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
            logs="./logs/9f3/simulation_{}.npz".format(simulation_i)
        )

    #plot_results.plot_turn_trajectory()
