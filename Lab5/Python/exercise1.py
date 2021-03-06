""" Lab 5 - Exercise 1 """

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization! 

from matplotlib import cm

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

def plotXY(X,Y,X_Label,Y_Label,Y_Legend,Title,Figure):
    plt.figure(Figure)
    plt.plot(X, Y, label=Y_Legend)
    plt.title(Title)
    plt.xlabel(X_Label)
    plt.ylabel(Y_Label)
    plt.legend(loc='upper right')
    plt.grid()

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
    
    #muscle_tendon_forces = []  #Erase or comment
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
        
#        muscle_tendon_forces.append(result.tendon_force[-1])#Erase or comment
        muscle_active_forces.append(result.active_force[-1])
        muscle_passive_forces.append(result.passive_force[-1]) 
        total_force.append(result.active_force[-1]+result.passive_force[-1])
        contractile_element_length.append(result.l_ce[-1])
                
#        plotXY(time,result.l_ce,'Time [s]','Contractile Element Length [m]','Active',
#               'Isometric Muscle: Contractile Element Length vs Time',
#               'Isometric Muscle: Contractile Element Length vs Time')
        
#        plotXY(result.l_ce,result.active_force,'Contractile Element Length [m]','Active Force [N]','Active',
#               'Isometric Muscle: Contractile Element Length vs Force',
#               'Isometric Muscle: Contractile Element Length vs Force')
#        plt.plot(result.l_ce, result.passive_force, label='passive')
#        plt.plot(result.l_ce, result.active_force+result.passive_force, label='total')
        
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
    stimulations = np.arange(.0,1.0,.02)
    stretches = np.arange(.12,.36,.05) #default length is 0.24
    
    allStretches= []
    allStims = []
    allActForces = []
    allPassForces = []
    allNetForces = []
    
    for stretch in stretches:
        # Evalute for a single muscle stretch
        muscle_active_forces = []
        muscle_passive_forces = []
        total_force = []
        l_ce = []
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
            l_ce.append(result.l_ce[-1])
            allStretches.append(stretch)
            allStims.append(stimulation)
            allActForces.append(result.active_force[-1])
            allPassForces.append(result.passive_force[-1])
            allNetForces.append(total_force[-1])
            
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
        plt.plot(stimulations, muscle_active_forces, label='L_mtu = %.2f'%stretch)
        plt.title('Isometric Muscle: Stimulation vs Force')
        plt.xlabel('Stimulation')
        plt.ylabel('Active Force [N]')
        plt.legend(loc='upper right')
        plt.grid()
        plt.subplot(3,1,2)
        plt.plot(stimulations, muscle_passive_forces, label='L_mtu = %.2f'%stretch)
        plt.xlabel('Stimulation')
        plt.ylabel('Passive Force [N]')
        plt.legend(loc='upper right')
        plt.grid()
        
        plt.subplot(3,1,3)
        plt.plot(stimulations, total_force, label='L_mtu = %.2f'%stretch)
        plt.xlabel('Stimulation')
        plt.ylabel('Total Force [N]')
        plt.legend(loc='upper right')
        plt.grid()
        
    allActForces = np.array(allActForces).reshape((stretches.size,stimulations.size))
    allPassForces = np.array(allPassForces).reshape((stretches.size,stimulations.size))
    allNetForces = np.array(allNetForces).reshape((stretches.size,stimulations.size))
    stimulations, stretches = np.meshgrid(stimulations, stretches)
    
    fig1b = plt.figure('1b. Stim vs Active Force Surface Plot')
    ax = fig1b.gca(projection='3d')
    ax = fig1b.add_subplot(111, projection='3d')
    ax.plot_surface(stimulations,stretches,allActForces,cmap=cm.coolwarm,
                       linewidth=0.5, antialiased=False)
    ax.set_xlabel('Stimulation')
    ax.set_ylabel('Muscle Length (m)')
    ax.set_zlabel('Active Force (N)')
    plt.title('Stimulation vs Active Force')
    
    fig1b = plt.figure('1b. Stim vs Passive Force Surface Plot')
    ax = fig1b.gca(projection='3d')
    ax = fig1b.add_subplot(111, projection='3d')
    ax.plot_surface(stimulations,stretches,allPassForces,cmap=cm.coolwarm,
                       linewidth=0.5, antialiased=False)
    ax.set_xlabel('Stimulation')
    ax.set_ylabel('Muscle Length (m)')
    ax.set_zlabel('Passive Force (N)')
    plt.title('Stimulation vs Passive Force')
    
    fig1b = plt.figure('1b. Stim vs Total Force Surface Plot')
    ax = fig1b.gca(projection='3d')
    ax = fig1b.add_subplot(111, projection='3d')
    ax.plot_surface(stimulations,stretches,allNetForces,cmap=cm.coolwarm,
                       linewidth=0.5, antialiased=False)
    ax.set_xlabel('Stimulation')
    ax.set_ylabel('Muscle Length (m)')
    ax.set_zlabel('Net Force (N)')
    plt.title('Stimulation vs Total Force')
    
#    u = stimulations
#    v = stretches
#    u,v = np.meshgrid(u,v)
#    u = u.flatten()
#    v = v.flatten()
#    
#    points2D = np.vstack([u,v]).T
#    tri = Delaunay(points2D)
#    simplices = tri.simplices
#    
#    fig1b = FF.create_trisurf(np.array(allStims),np.array(allStretches),np.array(allActForces),
#                         colormap="Portland",
#                         simplices=simplices,
#                         title='Stim vs Force for Different Lengths')
#    
#    py.iplot(fig1b, filename='Stim vs Force for Different Lengths')
    # Add a color bar which maps values to colors.
    #fig1b.colorbar(surf, shrink=0.5, aspect=5)
    #plt.show()
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
        max_muscle_passive_forces.append(muscle_passive_forces[active_max_index])
        max_total_force.append(total_force[active_max_index])
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

#    print(max_muscle_active_forces)
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
    x0 = [0.0, sys.muscle.L_OPT, sys.muscle.L_OPT + sys.muscle.L_SLACK, 0.0]
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

    # Set max_vce
    max_vce = []

    # ---------------------------------------------
    # Small load experiment
    # ---------------------------------------------
    load_table_small = [5, 10, 20, 50, 100]

    # Begin plotting
    plt.figure('Isotonic muscle experiment - load [10, 200] [N]')
    max_vce_small = ex1d_for(sys, x0, time, time_step, time_stabilize,
                             muscle_stimulation, load_table_small,False)

    plt.title('Isotonic muscle experiment - load [5, 140] [N]')
    plt.xlabel('Time [s]')
    plt.ylabel('Muscle contractile velocity [m/s]')
    plt.legend(loc='upper right')
    plt.grid()

    # ---------------------------------------------
    # Big load experiment
    # ---------------------------------------------
    load_table_big = [150, 200, 220, 250, 500, 1000, 1500]

    # Begin plotting
    plt.figure('Isotonic muscle experiment - load [150, 1500] [N]')
    max_vce += ex1d_for(sys, x0, time, time_step, time_stabilize, muscle_stimulation, load_table_big, True)

    plt.title('Isotonic muscle experiment - load [150, 1500] [N]')
    plt.xlabel('Time [s]')
    plt.ylabel('Muscle contractile velocity [m/s]')
    plt.legend(loc='upper right')
    plt.grid()

    # ---------------------------------------------
    # Plot velocity - tension relation
    # ---------------------------------------------
    load = np.arange(5, 2500, 200)
    (max_vce, active_force) = ex1d_for(sys, x0, time, time_step, time_stabilize, muscle_stimulation, load, False)

    fig = plt.figure('Velocity - Tension')
    ax = fig.add_subplot(111)

    # Plot comments and line at 0 value
    min_val = 0.0
    if min(map(abs, max_vce)) not in max_vce:
        min_val = -min(map(abs, max_vce))
    else:
        min_val = min(map(abs, max_vce))

    xy = (load[max_vce.index(min_val)], min_val)
    xytext = (load[max_vce.index(min_val)]+50, min_val)
    ax.annotate('load = {:0.1f}'.format(152.2), xy=xy, xytext=xytext)

    plt.title('Velocity [m/s] - Tension [N]')
    plt.xlabel('Tension [N]')
    plt.ylabel('Velocity [m/s]')
    plt.grid()

    plt.plot(load, max_vce)
    plt.plot(load[max_vce.index(min_val)], min_val, 'o')


def ex1d_for(sys, x0, time, time_step, time_stabilize, muscle_stimulation, load, plot):
    max_vce = []
    active_force = []

    for l in load:
        # Run the integration
        result = sys.integrate(x0=x0,
                               time=time,
                               time_step=time_step,
                               time_stabilize=time_stabilize,
                               stimulation=muscle_stimulation,
                               load=l)

        # Recover the maximum velocity
        if abs(min(result.v_ce)) > max(result.v_ce):
            max_vce.append(min(result.v_ce))
            active_force.append(result.active_force[result.v_ce.argmin()])
        else:
            max_vce.append(max(result.v_ce))
            active_force.append(result.active_force[result.v_ce.argmax()])

        if plot:
            plt.plot(result.time, result.v_ce, label='load {}'.format(l))

    return max_vce, active_force


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
    x0 = [0.0, sys.muscle.L_OPT, sys.muscle.L_OPT + sys.muscle.L_SLACK, 0.0]
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
    # maximum force over stimulation
    # ---------------------------------------------
    
    # Evaluate for different muscle stimulation
    muscle_stimulation = np.arange(0, 1.1, 0.1)
    max_active_force = []
    max_passive_force = []
    max_sum_force = []

    # Begin plotting
    for s in muscle_stimulation:
        # Run the integration
        result = sys.integrate(x0=x0,
                               time=time,
                               time_step=time_step,
                               time_stabilize=time_stabilize,
                               stimulation=s,
                               load=load)

        if abs(min(result.active_force)) > max(result.active_force):
            max_active_force.append(min(result.active_force))
        else:
            max_active_force.append(max(result.active_force))

        if abs(min(result.passive_force)) > max(result.passive_force):
            max_passive_force.append(min(result.passive_force))
        else:
            max_passive_force.append(max(result.passive_force))

        max_sum_force.append(max_active_force[-1] + max_passive_force[-1])

    plt.figure('Isotonic muscle active force - stimulation [0, 1]')

    plt.plot(muscle_stimulation, max_active_force, label='maximum active force')
    plt.plot(muscle_stimulation, max_passive_force, label='maximum passive force')
    plt.plot(muscle_stimulation, max_sum_force, label='maximum sum force')

    plt.xlabel('Stimulation [-]')
    plt.ylabel('Muscle sum forces [N]')
    plt.legend(loc='upper right')
    plt.grid()

    # ---------------------------------------------
    # force - velocity over stimulation
    # ---------------------------------------------
    muscle_stimulation = np.arange(0, 1.1, 0.1)

    # Begin plotting
    for s in muscle_stimulation:
        # Run the integration
        result = sys.integrate(x0=x0,
                               time=time,
                               time_step=time_step,
                               time_stabilize=time_stabilize,
                               stimulation=s,
                               load=load)

        plt.figure('Isotonic muscle active force - velocity')
        plt.plot(result.v_ce[200:-1], result.active_force[200:-1], label='stimulation {:0.1f}'.format(s))

        plt.figure('Isotonic muscle passive force - velocity')
        plt.plot(result.v_ce[200:-1], result.passive_force[200:-1], label='stimulation {:0.1f}'.format(s))

        plt.figure('Isotonic muscle sum forces - velocity')
        plt.plot(result.v_ce[200:-1], result.active_force[200:-1] + result.passive_force[200:-1],
                 label='stimulation {:0.1f}'.format(s))

    plt.figure('Isotonic muscle active force - velocity')
    plt.xlabel('Velocity contractile element [m/s]')
    plt.ylabel('Active force [N]')
    plt.legend(loc='upper right')
    plt.grid()

    plt.figure('Isotonic muscle passive force - velocity')
    plt.xlabel('Velocity contractile element [m/s]')
    plt.ylabel('Passive force [N]')
    plt.legend(loc='upper right')
    plt.grid()

    plt.figure('Isotonic muscle sum forces - velocity')
    plt.xlabel('Velocity contractile element [m/s]')
    plt.ylabel('Sum forces [N]')
    plt.legend(loc='upper right')
    plt.grid()

    # ---------------------------------------------
    # Plot velocity - tension relation
    # ---------------------------------------------
    muscle_stimulation = np.arange(0, 1.1, 0.25)
    load = np.arange(5, 1500, 20)
    plt.figure('Velocity - Tension')

    # Begin plotting
    for s in muscle_stimulation:
        (max_vce, active_force) = ex1d_for(sys, x0, time, time_step, time_stabilize, s, load, False)
        plt.plot(load, max_vce, label="stimulation {:0.1f}".format(s))

    plt.title('Velocity [m/s] - Load [N]')
    plt.xlabel('Load [N]')
    plt.ylabel('Velocity [m/s]')
    plt.legend(loc='lower right')
    plt.grid()


def exercise1():
#    exercise1a()
#    exercise1b()
#    exercise1c()
    exercise1d()
    exercise1f()

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

