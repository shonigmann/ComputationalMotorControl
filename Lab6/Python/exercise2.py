""" Lab 6 Exercise 2

This file implements the pendulum system with two muscles attached

"""

from math import sqrt

import cmc_pylog as pylog
import numpy as np
from matplotlib import pyplot as plt

from cmcpack import DEFAULT
from cmcpack.plot import save_figure
from muscle import Muscle
from muscle_system import MuscleSytem
from neural_system import NeuralSystem
from pendulum_system import PendulumSystem
from system import System
from system_animation import SystemAnimation
from system_parameters import (MuscleParameters, 
                               PendulumParameters, NetworkParameters,)
from system_simulation import SystemSimulation

from poincare_crossings import poincare_crossings

# Global settings for plotting
# You may change as per your requirement
plt.rc('lines', linewidth=2.0)
plt.rc('font', size=12.0)
plt.rc('axes', titlesize=14.0)     # fontsize of the axes title
plt.rc('axes', labelsize=14.0)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=14.0)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14.0)    # fontsize of the tick labels

DEFAULT["save_figures"] = True

############Exercise 2A ###############################################
    #KPP: function called from exercise2() below
def fromtheta(muscle, a1, a2, name):
    
    muscle_length = []
    moment_arm = []
    thetas = np.arange(-np.pi/4, np.pi/4 , 0.001)
    
    for theta in thetas: 
        
        L = np.sqrt(a1**2+a2**2+2*a1*a2*np.sin(theta))
        h = a1*a2*np.cos(theta)/L
        
        muscle_length.append(L)
        moment_arm.append(h)
        
    plt.figure('Muscle %.1i v.s. Pendulum Angle' %(name))
    plt.plot(thetas, muscle_length, label ='Length')
    plt.title('Muscle %.1i v.s. Pendulum Angle' %(name))
    plt.xlabel('Theta [rad]')
    plt.ylabel('Distance [m]')
    plt.legend(loc='upper right')
    plt.grid() 
                
    plt.figure('Muscle %.1i v.s. Pendulum Angle' %(name))
    plt.plot(thetas, np.abs(moment_arm), label ='Moment Arm')
    plt.title('Muscle %.1i v.s. Pendulum Angle' %(name))
    plt.xlabel('Theta [rad]')
    plt.ylabel('Distance [m]')
    plt.legend(loc='upper right')
    plt.grid() 

    #calculate the total muscle force at each position to determine the max torque for the muscle
    passive_forces = []
    active_forces = []
    total_forces = []
    for length in muscle_length:                  
        passive_forces.append(muscle.compute_passive_force(length))
        active_forces.append(muscle.compute_active_force(length,0,1))
        total_forces = np.add(passive_forces,active_forces)
    
    total_torques = np.multiply(total_forces,moment_arm)
    
    plt.figure('Muscle Torque v.s. Pendulum Angle')
    plt.plot(thetas, np.abs(total_torques), label ='Muscle %.1i' %(name))
    plt.title('Max Muscle Torque v.s. Pendulum Angle')
    plt.xlabel('Theta [rad]')
    plt.ylabel('Muscle Torque (abs) [Nm]')
    plt.legend(loc='upper right')
    plt.grid() 
    
def exercise2():
    """ Main function to run for Exercise 2.

    Parameters
    ----------
        None

    Returns
    -------
        None
    """

    # Define and Setup your pendulum model here
    # Check PendulumSystem.py for more details on Pendulum class
    pendulum_params = PendulumParameters()  # Instantiate pendulum parameters
    pendulum_params.L = 0.5  # To change the default length of the pendulum
    pendulum_params.m = 1.  # To change the default mass of the pendulum
    pendulum = PendulumSystem(pendulum_params)  # Instantiate Pendulum object

    #### CHECK OUT PendulumSystem.py to ADD PERTURBATIONS TO THE MODEL #####

    pylog.info('Pendulum model initialized \n {}'.format(
        pendulum.parameters.showParameters()))

    # Define and Setup your pendulum model here
    # Check MuscleSytem.py for more details on MuscleSytem class
    M1_param = MuscleParameters()  # Instantiate Muscle 1 parameters
    M1_param.f_max = 1500  # To change Muscle 1 max force
    M2_param = MuscleParameters()  # Instantiate Muscle 2 parameters
    M2_param.f_max = 1500  # To change Muscle 2 max force
    M1 = Muscle(M1_param)  # Instantiate Muscle 1 object
    M2 = Muscle(M2_param)  # Instantiate Muscle 2 object
    # Use the MuscleSystem Class to define your muscles in the system
    muscles = MuscleSytem(M1, M2)  # Instantiate Muscle System with two muscles
    pylog.info('Muscle system initialized \n {} \n {}'.format(
        M1.parameters.showParameters(),
        M2.parameters.showParameters()))

    # Define Muscle Attachment points
    m1_origin = np.array([-0.17, 0.0])  # Origin of Muscle 1
    m1_insertion = np.array([0.0, -0.17])  # Insertion of Muscle 1

    m2_origin = np.array([0.17, 0.0])  # Origin of Muscle 2
    m2_insertion = np.array([0.0, -0.17])  # Insertion of Muscle 2

    # Attach the muscles
    muscles.attach(np.array([m1_origin, m1_insertion]),
                   np.array([m2_origin, m2_insertion]))
    
    ############Exercise 2A ###############################################
    # rigth after creating and attaching both muscles:
    
    print(m1_origin, m2_origin)
    m1a1 =abs( abs(m1_origin[0]) - abs(m1_origin[1]))
    m1a2 =abs( abs(m1_insertion[0]) - abs(m1_insertion[1]))

    m1a1 = m1_origin[0] - m1_origin[1]
    m1a2 = m1_insertion[0] - m1_insertion[1]
    m2a1 = m2_origin[0] - m2_origin[1]
    m2a2 = m2_insertion[0] - m2_insertion[1]

    print(m1a1, m1a2)
    fromtheta(M1, m1a1, m1a2, 1)
    fromtheta(M2, m2a1, m2a2, 2)
    
    # Create a system with Pendulum and Muscles using the System Class
    # Check System.py for more details on System class
    sys = System()  # Instantiate a new system
    sys.add_pendulum_system(pendulum)  # Add the pendulum model to the system
    sys.add_muscle_system(muscles)  # Add the muscle model to the system

    ##### Time #####
    t_max = 5  # Maximum simulation time

    time = np.arange(0., t_max, 0.002)  # Time vector

    ##### Model Initial Conditions #####
    x0_P = np.array([np.pi/4, 0.])  # Pendulum initial condition
    x0_P = np.array([0., 0.])  # Pendulum initial condition

    # Muscle Model initial condition
    x0_M = np.array([0., M1.L_OPT, 0., M2.L_OPT])

    x0 = np.concatenate((x0_P, x0_M))  # System initial conditions

    ##### System Simulation #####
    # For more details on System Simulation check SystemSimulation.py
    # SystemSimulation is used to initialize the system and integrate
    # over time

    sim = SystemSimulation(sys)  # Instantiate Simulation object

    # Add muscle activations to the simulation
    # Here you can define your muscle activation vectors
    # that are time dependent

    wave_h1 = np.sin(time*3)*1               #makes a sinusoidal wave from 'time'
    wave_h2 = np.sin(time*3 + np.pi)*1       #makes a sinusoidal wave from 'time'
    
    wave_h1[wave_h1<0] = 0      #formality of passing negative values to zero
    wave_h2[wave_h2<0] = 0      #formality of passing negative values to zero
    
    act1 = wave_h1.reshape(len(time), 1) #makes a vertical array like act1
    act2 = wave_h2.reshape(len(time), 1) #makes a vertical array like act1
    
    # Plotting the waveforms
    plt.figure('Muscle Activations')
    plt.title('Muscle Activation Functions')
    plt.plot(time, wave_h1, label='Muscle 1')
    plt.plot(time, wave_h2, label='Muscle 2')
    plt.xlabel('Time [s]')
    plt.ylabel('Muscle Excitation')
    plt.legend(loc='upper right')
    plt.grid()

    activations = np.hstack((act1, act2))

    # Method to add the muscle activations to the simulation
    sim.add_muscle_activations(activations)

    # Simulate the system for given time
    sim.initalize_system(x0, time)  # Initialize the system state

    #: If you would like to perturb the pedulum model then you could do
    # so by
    sim.sys.pendulum_sys.parameters.PERTURBATION = False
    # The above line sets the state of the pendulum model to zeros between
    # time interval 1.2 < t < 1.25. You can change this and the type of
    # perturbation in
    # pendulum_system.py::pendulum_system function

    # Integrate the system for the above initialized state and time
    sim.simulate()

    # Obtain the states of the system after integration
    # res is np.array [time, states]
    # states vector is in the same order as x0
    res = sim.results()

    # In order to obtain internal states of the muscle
    # you can access the results attribute in the muscle class
    muscle1_results = sim.sys.muscle_sys.Muscle1.results
    muscle2_results = sim.sys.muscle_sys.Muscle2.results

    # Plotting the results
    plt.figure('Pendulum_phase')
    plt.title('Pendulum Phase')
    plt.plot(res[:, 1], res[:, 2])
    plt.xlabel('Position [rad]')
    plt.ylabel('Velocity [rad.s]')
    plt.grid()
    
    # Plotting the results: Amplidute stimulation
    plt.figure('Amplidute stimulation')
    plt.title('Amplidute stimulation')
    plt.plot(time, res[:, 1], label = 'Stimul. 0.2')
    plt.xlabel('time [s]')
    plt.ylabel('Position [rad]')
    plt.legend(loc ='upper left')
    plt.grid()
    
    # Plotting the results: frequency stimulation
    plt.figure('Frequency stimulation')
    plt.title('Frequency stimulation')
    plt.plot(time, res[:, 1], label = 'w: 3 rad/s')
    plt.xlabel('time [s]')
    plt.ylabel('Position [rad]')
    plt.legend(loc ='upper left')
    plt.grid()

    poincare_crossings(res, -2, 1, "Pendulum")

    # To animate the model, use the SystemAnimation class
    # Pass the res(states) and systems you wish to animate
    simulation = SystemAnimation(res, pendulum, muscles)
    # To start the animation
    if DEFAULT["save_figures"] is False:
        simulation.animate()

    if not DEFAULT["save_figures"]:
        plt.show()
    else:
        figures = plt.get_figlabels()
        pylog.debug("Saving figures:\n{}".format(figures))
        for fig in figures:
            plt.figure(fig)
            save_figure(fig)
            plt.close(fig)
    #######################################################################

if __name__ == '__main__':
    from cmcpack import parse_args
    parse_args()
    exercise2()

