""" Lab 5 - Exercise 1 """

import matplotlib.pyplot as plt
import numpy as np

import cmc_pylog as pylog
from muscle import Muscle
from mass import Mass
from cmcpack import DEFAULT, parse_args
from cmcpack.plot import save_figure
from system_parameters import MuscleParameters, MassParameters
from isometric_muscle_system import IsometricMuscleSystem
from isotonic_muscle_system import IsotonicMuscleSystem

DEFAULT["label"] = [r"$\theta$ [rad]", r"$d\theta/dt$ [rad/s]"]

# Global settings for plotting
# You may change as per your requirement
plt.rc('lines', linewidth=2.0)
plt.rc('font', size=12.0)
plt.rc('axes', titlesize=14.0)     # fontsize of the axes title
plt.rc('axes', labelsize=14.0)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14.0)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14.0)    # fontsize of the tick labels

DEFAULT["save_figures"] = True


def exercise1a():
    """ Exercise 1a
    The goal of this exercise is to understand the relationship
    between muscle length and tension.
    Here you will re-create the isometric muscle contraction experiment.
    To do so, you will have to keep the muscle at a constant length and
    observe the force while stimulating the muscle at a constant activation."""

    # Defination of muscles
    parameters = MuscleParameters()
    pylog.warning("Loading default muscle parameters")
    pylog.info(parameters.showParameters())
    pylog.info("Use the parameters object to change the muscle parameters")

    # Create muscle object
    muscle = Muscle(parameters)

    pylog.warning("Isometric muscle contraction to be completed")

    # Instatiate isometric muscle system
    sys = IsometricMuscleSystem()

    # Add the muscle to the system
    sys.add_muscle(muscle)

    # You can still access the muscle inside the system by doing
    # >>> sys.muscle.L_OPT # To get the muscle optimal length
    muscle_stretches = np.arange(.18,.30,.001)

    muscle_active_forces = []
    muscle_passive_forces = []
    total_force = []
    
    # Evalute for a single muscle stretch
    for muscle_stretch in muscle_stretches:
       
        # Evalute for a single muscle stimulation
        muscle_stimulation = 1.
    
        # Set the initial condition
        x0 = [0.0, sys.muscle.L_OPT]
        # x0[0] --> muscle stimulation intial value
        # x0[1] --> muscle contracticle length initial value
    
        # Set the time for integration
        t_start = 0.0
        t_stop = 0.3
        time_step = 0.001
    
        time = np.arange(t_start, t_stop, time_step)
    
        # Run the integration
        result = sys.integrate(x0=x0,
                               time=time,
                               time_step=time_step,
                               stimulation=muscle_stimulation,
                               muscle_length=muscle_stretch)
        
        muscle_active_forces.append(result.active_force[-1])
        muscle_passive_forces.append(result.passive_force[-1])
        total_force.append(result.active_force[-1]+result.passive_force[-1])
        
    # Plotting
    plt.figure('Isometric muscle experiment')
    plt.plot(muscle_stretches, muscle_active_forces, label='active')
    plt.plot(muscle_stretches, muscle_passive_forces, label='passive')
    plt.plot(muscle_stretches, total_force, label='total')
    plt.title('Isometric muscle experiment')
    plt.xlabel('Length [m]')
    plt.ylabel('Muscle Force [N]')
    plt.legend(loc='upper right')
    plt.grid()
    
def exercise1b():
    
    a = 1
    b = 2
    
def exercise1c():
    """describe how fiber length influences the force-length curve. Compare 
    a muscle comprised of short muscle fibers to a muscle comprised of 
    long muscle fibers. Change the parameter, you can use 
    system_parameters.py::MuscleParameters before instantiating the muscle
    No more than 2 plots are required.
    """
    
    # Defination of muscles
    parameters = MuscleParameters()
    pylog.warning("Loading default muscle parameters")
    pylog.info(parameters.showParameters())
    pylog.info("Use the parameters object to change the muscle parameters")
    
    # Create muscle object
    muscle = Muscle(parameters)
    
    pylog.warning("Isometric muscle contraction to be completed")
    
    # Instatiate isometric muscle system
    sys = IsometricMuscleSystem()
    
    # Add the muscle to the system
    sys.add_muscle(muscle)
    
    # You can still access the muscle inside the system by doing
    # >>> sys.muscle.L_OPT # To get the muscle optimal length
    muscle_lengths = np.arange(.02,.22,.04)
    
    max_muscle_active_forces = []
    max_muscle_passive_forces = []
    max_total_force = []
    
    # Evalute for a single muscle stretch
    for muscle_length in muscle_lengths:
       
        parameters.l_opt = muscle_length
        muscle = Muscle(parameters)
    
        pylog.warning("Isometric muscle contraction to be completed")
        
        # Instatiate isometric muscle system
        sys = IsometricMuscleSystem()
        
        # Add the muscle to the system
        sys.add_muscle(muscle)
        muscle_stretches = np.arange(muscle_length,4*muscle_length,3*muscle_length/30)

        muscle_active_forces = []
        muscle_passive_forces = []
        total_force = []
        
        # Evalute for a single muscle stretch
        for muscle_stretch in muscle_stretches:
           
            # Evalute for a single muscle stimulation
            muscle_stimulation = 1.
        
            # Set the initial condition
            x0 = [0.0, sys.muscle.L_OPT]
            # x0[0] --> muscle stimulation intial value
            # x0[1] --> muscle contracticle length initial value
        
            # Set the time for integration
            t_start = 0.0
            t_stop = 0.3
            time_step = 0.001
        
            time = np.arange(t_start, t_stop, time_step)
        
            # Run the integration
            result = sys.integrate(x0=x0,
                                   time=time,
                                   time_step=time_step,
                                   stimulation=muscle_stimulation,
                                   muscle_length=muscle_stretch)
            
            muscle_active_forces.append(result.active_force[-1])
            muscle_passive_forces.append(result.passive_force[-1])
            total_force.append(result.active_force[-1]+result.passive_force[-1])
            
        # Plotting
        plt.figure('Isometric muscle experiment. L_opt = '+str(muscle_length))
        plt.plot(muscle_stretches, muscle_active_forces, label='active')
        plt.plot(muscle_stretches, muscle_passive_forces, label='passive')
        plt.plot(muscle_stretches, total_force, label='total')
        plt.title('Isometric muscle experiment')
        plt.xlabel('Length [m]')
        plt.ylabel('Muscle Force [N]')
        plt.legend(loc='upper right')
        plt.grid()
    

def exercise1d():
    """ Exercise 1d

    Under isotonic conditions external load is kept constant.
    A constant stimulation is applied and then suddenly the muscle
    is allowed contract. The instantaneous velocity at which the muscle
    contracts is of our interest."""

    # Defination of muscles
    muscle_parameters = MuscleParameters()
    print(muscle_parameters.showParameters())

    mass_parameters = MassParameters()
    print(mass_parameters.showParameters())

    # Create muscle object
    muscle = Muscle(muscle_parameters)

    # Create mass object
    mass = Mass(mass_parameters)

    pylog.warning("Isotonic muscle contraction to be implemented")

    # Instatiate isotonic muscle system
    sys = IsotonicMuscleSystem()

    # Add the muscle to the system
    sys.add_muscle(muscle)

    # Add the mass to the system
    sys.add_mass(mass)

    # You can still access the muscle inside the system by doing
    # >>> sys.muscle.L_OPT # To get the muscle optimal length

    # Evalute for a single load
    load = 100.

    # Evalute for a single muscle stimulation
    muscle_stimulation = 1.

    # Set the initial condition
    x0 = [0.0, sys.muscle.L_OPT,
          sys.muscle.L_OPT + sys.muscle.L_SLACK, 0.0]
    # x0[0] - -> activation
    # x0[1] - -> contractile length(l_ce)
    # x0[2] - -> position of the mass/load
    # x0[3] - -> velocity of the mass/load

    # Set the time for integration
    t_start = 0.0
    t_stop = 0.5
    time_step = 0.001
    time_stabilize = 0.2

    time = np.arange(t_start, t_stop, time_step)

    # Run the integration
    result = sys.integrate(x0=x0,
                           time=time,
                           time_step=time_step,
                           time_stabilize=time_stabilize,
                           stimulation=muscle_stimulation,
                           load=load)
    
    # Plotting
    plt.figure('Isotonic muscle experiment')
    plt.plot(result.time, result.v_ce)
    plt.title('Isotonic muscle experiment')
    plt.xlabel('Time [s]')
    plt.ylabel('Muscle contractilve velocity')
    plt.grid()


def exercise1():
#    exercise1a()
#    exercise1b()
    exercise1c()
#    exercise1d()

    if DEFAULT["save_figures"] is False:
        plt.show()
    else:
        figures = plt.get_figlabels()
        print(figures)
        pylog.debug("Saving figures:\n{}".format(figures))
        for fig in figures:
            plt.figure(fig)
            save_figure(fig)
            plt.close(fig)


if __name__ == '__main__':
    from cmcpack import parse_args
    parse_args()
    exercise1()

