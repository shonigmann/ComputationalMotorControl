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
    muscle_stretches = np.arange(.12,.30,.002)

    muscle_active_forces = []
    muscle_passive_forces = []
    total_force = []
    contractile_element_length = []
    
    
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
        contractile_element_length.append(result.l_ce[-1])
            
#        # Plotting
#        plt.figure('Isometric Muscle: Contractile Element Length vs Time')
#        plt.plot(time, result.l_ce, label='active')
#        plt.title('Isometric Muscle: Contractile Element Length vs Time')
#        plt.xlabel('Time [s]')
#        plt.ylabel('Contractile Element Length [m]]')
#        plt.legend(loc='upper right')
#        plt.grid()
#        
#        plt.figure('Isometric Muscle: Contractile Element Length vs Force')
#        plt.plot(result.l_ce, result.active_force, label='active')
#        plt.plot(result.l_ce, result.passive_force, label='passive')
#        plt.plot(result.l_ce, result.active_force+result.passive_force, label='total')
#        plt.title('Isometric Muscle: Contractile Element Length vs Force')
#        plt.xlabel('Contractile Element Length [m]')
#        plt.ylabel('Muscle Force [N]')
#        plt.legend(loc='upper right')
#        plt.grid()
        
    # Plotting
    plt.figure('Isometric Muscle: L_ce vs Force')
    plt.plot(contractile_element_length, muscle_active_forces, label='active')
    plt.plot(contractile_element_length, muscle_passive_forces, label='passive')
    plt.plot(contractile_element_length, total_force, label='total')
    plt.title('Isometric Muscle: Stretch Length vs Force')
    plt.xlabel('Contractile Element Length [m]')
    plt.ylabel('Muscle Force [N]')
    plt.legend(loc='upper right')
    plt.grid()


def exercise1b():
    
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
    stimulations = np.arange(.0,1.0,.03)
    stretches = np.arange(.12,.36,.06) #default length is 0.24

    for stretch in stretches:
        # Evalute for a single muscle stretch
        muscle_active_forces = []
        muscle_passive_forces = []
        total_force = []
        for stimulation in stimulations:
           
            # Evalute for a single muscle stimulation
            muscle_stimulation = stimulation
        
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
                                   muscle_length=stretch)
            
            muscle_active_forces.append(result.active_force[-1])
            muscle_passive_forces.append(result.passive_force[-1])
            total_force.append(result.active_force[-1]+result.passive_force[-1])
            
#            # Plotting results of individual trials to verify steady state assumption
#            plt.figure('Force over time with different stimulations and lengths %.2f' %stretch)
#            plt.plot(time, result.active_force, label='Active: Stim = %.2f'%stimulation + ' L = %.2f'%stretch)
#            plt.plot(time, result.passive_force, label='Passive: Stim = %.2f'%stimulation + ' L = %.2f'%stretch)
#            plt.plot(time, result.active_force+result.passive_force, label='Net: Stim = %.2f'%stimulation + ' L = %.2f'%stretch)
#            plt.title('Force over time with different stimulations and lengths')
#            plt.xlabel('Time [s]')
#            plt.ylabel('Active Muscle Force [N]')
#            plt.legend(loc='upper right')
#            plt.grid()
            
        # Plotting
        plt.figure('Isometric Muscle: Stimulation vs Force')
        plt.subplot(3,1,1)
        plt.plot(stimulations, muscle_active_forces, label='Stretch = %.2f'%stretch)
        plt.title('Isometric Muscle: Stimulation vs Force')
        plt.xlabel('Stimulation')
        plt.ylabel('Active Muscle Force [N]')
        plt.legend(loc='upper right')
        plt.grid()
        
        plt.subplot(3,1,2)
        plt.plot(stimulations, muscle_passive_forces, label='Stretch = %.2f'%stretch)
        plt.xlabel('Stimulation')
        plt.ylabel('Passive Muscle Force [N]')
        plt.legend(loc='upper right')
        plt.grid()
        
        plt.subplot(3,1,3)
        plt.plot(stimulations, total_force, label='Stretch = %.2f'%stretch)
        plt.xlabel('Stimulation')
        plt.ylabel('Total Muscle Force [N]')
        plt.legend(loc='upper right')
        plt.grid()
        
    
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
    muscle_lengths = np.arange(.1,.26,.03)
    
    max_muscle_active_forces = []
    max_muscle_passive_forces = []
    max_total_force = []
    max_force_stretch = []
    
    # Evalute for a single muscle stretch
    for muscle_length in muscle_lengths:
       
        parameters.l_opt = muscle_length
        muscle = Muscle(parameters)
    
        pylog.warning("Isometric muscle contraction to be completed")
        
        # Instatiate isometric muscle system
        sys = IsometricMuscleSystem()
        
        # Add the muscle to the system
        sys.add_muscle(muscle)
        if muscle_length<.16:
            start_stretch_length = .16
        else:
            start_stretch_length = muscle_length
        muscle_stretches = np.arange(start_stretch_length,1.2*muscle_length+.16,(1.2*muscle_length+.16-start_stretch_length)/40)

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
            
        max_muscle_active_forces.append(max(muscle_active_forces))
        
        active_max_index = muscle_active_forces.index(max(muscle_active_forces))
        max_muscle_passive_forces.append(muscle_active_forces[active_max_index])
        max_total_force.append(muscle_active_forces[active_max_index])
        max_force_stretch.append(muscle_stretches[active_max_index])
        
        # Plotting max force for each muscle length over different stretch values. Uncomment to see (adds ~8 plots)
#        plt.figure('Isometric muscle experiment. L_opt = %.2f'%(muscle_length))
#        plt.plot(muscle_stretches, muscle_active_forces, label='active')
#        plt.plot(muscle_stretches, muscle_passive_forces, label='passive')
#        plt.plot(muscle_stretches, total_force, label='total')
#        plt.title('Isometric muscle experiment 1C, L_opt = %.2f'%(muscle_length))
#        plt.xlabel('Stretch Length [m]')
#        plt.ylabel('Muscle Force [N]')
#        plt.legend(loc='upper left')
#        plt.grid()
        
        # Plotting active on its own
        plt.figure('Isometric muscle experiment, Active Force')
        plt.plot(muscle_stretches, muscle_active_forces, label=('L_opt = %.2f' %(muscle_length)))
        plt.title('Isometric muscle experiment: Active Force vs L_Opt')
        plt.xlabel('Stretch Length [m]')
        plt.ylabel('Active Muscle Force [N]')
        plt.legend(loc='upper left')
        plt.grid()
        
        # Plotting passive on its own
        plt.figure('Isometric muscle experiment, Passive Force')
        plt.plot(muscle_stretches, muscle_passive_forces, label=('L_opt = %.2f'%(muscle_length)))
        plt.title('Isometric muscle experiment: Passive Force vs L_Opt')
        plt.xlabel('Stretch Length [m]')
        plt.ylabel('Passive Muscle Force [N]')
        plt.legend(loc='upper left')
        plt.grid()
        
    # Plot max vals
    plt.figure('Isometric muscle experiment max Force')
    plt.plot(muscle_lengths, max_muscle_active_forces, label='active')
    #plt.plot(muscle_lengths, max_muscle_passive_forces, label='passive')
    #plt.plot(muscle_lengths, max_total_force, label='total')
    plt.title('Isometric muscle experiment 1C, Max')
    plt.xlabel('Muscle Optimal Length [m]')
    plt.ylabel('Max Muscle Force [N]')
    plt.legend(loc='upper left')
    plt.grid()

    # Plot max stretch lengths of max vals
    plt.figure('Isometric muscle experiment max Force stretch')
    plt.plot(muscle_lengths, max_force_stretch, label='active')
    #plt.plot(muscle_lengths, max_muscle_passive_forces, label='passive')
    #plt.plot(muscle_lengths, max_total_force, label='total')
    plt.title('Isometric muscle experiment 1C, Max Stretch')
    plt.xlabel('Muscle Optimal Length [m]')
    plt.ylabel('Muscle Stretch of Max Force [m]')
    plt.legend(loc='upper left')
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

    # Evaluate for a single muscle stimulation
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
    t_stop = 1.25
    time_step = 0.001
    time_stabilize = 0.2

    time = np.arange(t_start, t_stop, time_step)

    # ---------------------------------------------
    # Small load experiment
    # ---------------------------------------------
    load_table_small = np.arange(10, 200, 40)

    # Begin plotting
    plt.figure('Isotonic muscle experiment - load [10, 200] [N]')
    max_vce_small = ex1d_for(sys, x0, time, time_step, time_stabilize, muscle_stimulation, load_table_small)

    plt.title('Isotonic muscle experiment - load [10, 200] [N]')
    plt.xlabel('Time [s]')
    plt.ylabel('Muscle contractile velocity [m/s]')
    plt.legend(loc='upper right')
    plt.grid()

    # Plot steps
    steps = np.arange(0, len(load_table_small), 1)

    plt.figure('Steps - load [10, 200] [N]')
    plt.plot(steps, max_vce_small)
    plt.title('Steps - load [10, 200] [N]')
    plt.xlabel('Steps [-]')
    plt.ylabel('Max muscle contractile velocity [m/s]')
    plt.legend(loc='upper right')
    plt.grid()

    # ---------------------------------------------
    # Big load experiment
    # ---------------------------------------------
    load_table_big = np.arange(200, 2000, 400)

    # Begin plotting
    plt.figure('Isotonic muscle experiment - load [200, 2000] [N]')
    max_vce_big = ex1d_for(sys, x0, time, time_step, time_stabilize, muscle_stimulation, load_table_big)

    plt.title('Isotonic muscle experiment - load [200, 2000] [N]')
    plt.xlabel('Time [s]')
    plt.ylabel('Muscle contractile velocity [m/s]')
    plt.legend(loc='upper right')
    plt.grid()

    # Plot steps
    steps = np.arange(0, len(load_table_big), 1)

    plt.figure('Steps - load [200, 2000] [N]')
    plt.plot(steps, max_vce_big)
    plt.title('Steps - load [200, 2000] [N]')
    plt.xlabel('Steps [-]')
    plt.ylabel('Max muscle contractile velocity [m/s]')
    plt.legend(loc='upper right')
    plt.grid()


def ex1d_for(sys, x0, time, time_step, time_stabilize, muscle_stimulation, load):
    max_vce = []

    for l in load:
        # Run the integration
        result = sys.integrate(x0=x0,
                               time=time,
                               time_step=time_step,
                               time_stabilize=time_stabilize,
                               stimulation=muscle_stimulation,
                               load=l)

        # Recover the maximum velocity
        if result.l_mtc[-1] < (sys.muscle.L_OPT + sys.muscle.L_SLACK):
            max_vce.append(max(result.v_ce))
        else:
            max_vce.append(min(result.v_ce))

        plt.plot(result.time, result.v_ce, label='load {}'.format(l))

    return max_vce


def exercise1f():
    """ Exercise 1f

    What happens to the force-velocity relationship when the stimulation is varied
    between [0 - 1]?"""
    # Defination of muscles
    muscle_parameters = MuscleParameters()
    print(muscle_parameters.showParameters())

    mass_parameters = MassParameters()
    print(mass_parameters.showParameters())

    # Create muscle object
    muscle = Muscle(muscle_parameters)

    # Create mass object
    mass = Mass(mass_parameters)

    # Instantiate isotonic muscle system
    sys = IsotonicMuscleSystem()

    # Add the muscle to the system
    sys.add_muscle(muscle)

    # Add the mass to the system
    sys.add_mass(mass)

    # You can still access the muscle inside the system by doing
    # >>> sys.muscle.L_OPT # To get the muscle optimal length

    # Evaluate for a single load
    load = 100.

    # Set the initial condition
    x0 = [0.0, sys.muscle.L_OPT,
          sys.muscle.L_OPT + sys.muscle.L_SLACK, 0.0]
    # x0[0] - -> activation
    # x0[1] - -> contractile length(l_ce)
    # x0[2] - -> position of the mass/load
    # x0[3] - -> velocity of the mass/load

    # Set the time for integration
    t_start = 0.0
    t_stop = 1.25
    time_step = 0.001
    time_stabilize = 0.2

    time = np.arange(t_start, t_stop, time_step)

    # Evaluate for different muscle stimulations
    muscle_stimulation = np.arange(0, 1, 0.1)

    # Begin plotting
    plt.figure('Isotonic muscle experiment - stimulation [0, 1]')

    for s in muscle_stimulation:
        # Run the integration
        result = sys.integrate(x0=x0,
                               time=time,
                               time_step=time_step,
                               time_stabilize=time_stabilize,
                               stimulation=s,
                               load=load)

        plt.plot(result.time, result.v_ce, label='stimulation {}'.format(s))

    plt.xlabel('Time [s]')
    plt.ylabel('Muscle contractile velocity [m/s]')
    plt.legend(loc='upper right')
    plt.grid()


def exercise1():
    exercise1a()
#    exercise1b()
#    exercise1c()
#    exercise1d()
#    exercise1f()


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

