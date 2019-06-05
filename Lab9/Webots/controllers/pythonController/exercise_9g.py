"""Exercise 9g"""

# from run_simulation import run_simulation
import numpy as np
from run_simulation import run_simulation
from simulation_parameters import SimulationParameters
import plot_results


def exercise_9g(world, timestep, reset):
    """Exercise 9g"""
    n_joints = 10

    parameter_set = [
        SimulationParameters(
            simulation_duration=40,
            use_drive_saturation=1,
            shes_got_legs=1,
            cR_body=[0.052, 0.157],#0.052  # cR1, cR0
            cR_limb=[0.105, 0.105],#0.105  # cR1, cR0
            drive_left=2.0,
            drive_right=2.0,
            enable_transitions=True,
            # ...
        )
        # for turn in np.linspace(-.2,.2, 7)
    ]

    # Grid search
    # logs =""
    for simulation_i, parameters in enumerate(parameter_set):
        reset.reset()
        run_simulation(
            world,
            parameters,
            timestep,
            int(1000 * parameters.simulation_duration / timestep),
            logs="./logs/9g/simulation_{}.npz".format(simulation_i)
        )

    # plot_results.plot_turn_trajectory()
    plot_results.plot_9g()

