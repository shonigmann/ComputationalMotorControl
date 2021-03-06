""" Lab 6 Exercise 3
This file implements the pendulum system with two muscles attached driven
by a neural network
"""

import numpy as np
from matplotlib import pyplot as plt

import cmc_pylog as pylog
from cmcpack import DEFAULT
from cmcpack.plot import save_figure
from muscle import Muscle
from muscle_system import MuscleSytem
from neural_system import NeuralSystem
from pendulum_system import PendulumSystem
from system import System
from system_animation import SystemAnimation
from system_parameters import (MuscleParameters, NetworkParameters,
                               PendulumParameters)
from system_simulation import SystemSimulation

# Global settings for plotting
# You may change as per your requirement
plt.rc('lines', linewidth=2.0)
plt.rc('font', size=12.0)
plt.rc('axes', titlesize=14.0)     # fontsize of the axes title
plt.rc('axes', labelsize=14.0)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14.0)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14.0)    # fontsize of the tick labels


def exercise3():
  #  exercise3a()
    exercise3b()


def exercise3a():
    """ Main function to run for Exercise 3.

    Parameters
    ----------
        None

    Returns
    -------
        None
    """
    # Define and Setup your pendulum model here
    # Check Pendulum.py for more details on Pendulum class
    P_params = PendulumParameters()  # Instantiate pendulum parameters
    P_params.L = 0.5  # To change the default length of the pendulum
    P_params.m = 1.  # To change the default mass of the pendulum
    P_params.PERTURBATION = True
    pendulum = PendulumSystem(P_params)  # Instantiate Pendulum object

    #### CHECK OUT Pendulum.py to ADD PERTURBATIONS TO THE MODEL #####
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

    ##### Neural Network #####
    # The network consists of four neurons
    N_params = NetworkParameters()  # Instantiate default network parameters
    N_params.tau = [0.02, 0.02, 0.1, 0.1]
    N_params.b = [3.0, 3.0, -3.0, -3.0]
    N_params.D = 1.0 # To change a network parameter
    N_params.w = np.asarray([[0.0, -5.0, -5.0, 0.0],
                             [-5.0, 0.0, 0.0, -5.0],
                             [5.0, -5.0, 0.0, 0.0],
                             [-5.0, 5.0, 0.0, 0.0]])
    # Similarly to change w -> N_params.w = (4x4) array
    print(N_params.w)
    ############################# Exercise 3A  ######################
    N_params.w = np.transpose(np.asarray([[ 0, -1,  1, -1],
                                          [-1,  0, -1,  1],
                                          [-1,  0,  0,  0], 
                                          [ 0, -1,  0,  0]]))*5
    print(N_params.w, N_params.D, N_params.tau, N_params.b, N_params.exp )

    # Create a new neural network with above parameters
    neural_network = NeuralSystem(N_params)
    pylog.info('Neural system initialized \n {}'.format(
        N_params.showParameters()))

    # Create system of Pendulum, Muscles and neural network using SystemClass
    # Check System.py for more details on System class
    sys = System()  # Instantiate a new system
    sys.add_pendulum_system(pendulum)  # Add the pendulum model to the system
    sys.add_muscle_system(muscles)  # Add the muscle model to the system
    sys.add_neural_system(neural_network)  # Add the neural network to the system

    ##### Time #####
    t_max = 2.  # Maximum simulation time
    time = np.arange(0., t_max, 0.001)  # Time vector

    ##### Model Initial Conditions #####
    x0_P = np.array([[-0.5, 0], [-0.25, -0.25], [0., 0.], [0.5, 0]])  # Pendulum initial condition

    for i in x0_P:
        # Muscle Model initial condition
        x0_M = np.array([0., M1.L_OPT, 0., M2.L_OPT])

        x0_N = np.array([-1.5, 1, 2.5, 1])  # Neural Network Initial Conditions

        x0 = np.concatenate((i, x0_M, x0_N))  # System initial conditions

        ##### System Simulation #####
        # For more details on System Simulation check SystemSimulation.py
        # SystemSimulation is used to initialize the system and integrate
        # over time

        sim = SystemSimulation(sys)  # Instantiate Simulation object

#    sim.add_external_inputs_to_network(np.ones((len(time), 4)))
    
#    wave_h1 = np.sin(time*3)*2               #makes a sinusoidal wave from 'time'
#    wave_h2 = np.sin(time*3 + np.pi)*1       #makes a sinusoidal wave from 'time'
#    
#    wave_h1[wave_h1<0] = 0      #formality of passing negative values to zero
#    wave_h2[wave_h2<0] = 0      #formality of passing negative values to zero
#    
#    act1 = wave_h1.reshape(len(time), 1) #makes a vertical array like act1
#    act2 = wave_h2.reshape(len(time), 1) #makes a vertical array like act1
#    column = np.ones((len(time), 1))
    
#    ext_in = np.hstack((act1, column, act2, column))
    
#    sim.add_external_inputs_to_network(ext_in)
        sim.initalize_system(x0, time)  # Initialize the system state

        sim.sys.pendulum_sys.parameters.PERTURBATION = False

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

    # Plotting the results: Position(phase) vs time
    plt.figure('Pendulum Phase')
    plt.title('Pendulum Phase')
    plt.plot(res[:, 0], res[:,1])   #to plot pendulum Position (phase)
#    plt.plot(res[:, 0], time)   #to plot position
#    plt.plot(res[:, 0], res[:, -5:-1])  # to Plot neurons' states
    plt.xlabel('time [s]')
    plt.ylabel('Position [rad]')
    plt.grid()

    # Plotting the results: Velocity vs Position (phase)
    plt.figure('Pendulum Vel v.s. Phase')
    plt.title('Pendulum Vel v.s. Phase')
    plt.plot(res[:, 1], res[:,2])   #to plot Velocity vs Position (phase)
    plt.xlabel('Position [rad]')
    plt.ylabel('Velocity [rad.s]')
    plt.grid()

    # Plotting the results: Velocity vs time 
    plt.figure('Pendulum Velocity')
    plt.title('Pendulum Velocity')
    plt.plot(res[:, 0], res[:,2])   #to plot Velocity vs Position
    plt.xlabel('time [s]')
    plt.ylabel('Velocity [rad.s]')
    plt.grid()

    # Plotting the results: Output of the network
    plt.figure('Network output')
    plt.title('Network output')
    plt.plot(res[:, 0], res[:,-1], label='neuron1')   #to plot Velocity vs Position
    plt.plot(res[:, 0], res[:,-2], label='neuron2') 
    plt.plot(res[:, 0], res[:,-3], label='neuron3') 
    plt.plot(res[:, 0], res[:,-4], label='neuron4') 
    plt.xlabel('time [s]')
    plt.ylabel('Stimulation ')
    plt.legend(loc='upper right')
    plt.grid()
    


    if DEFAULT["save_figures"] is False:
        plt.show()
    else:
        figures = plt.get_figlabels()
        pylog.debug("Saving figures:\n{}".format(figures))
        for fig in figures:
            plt.figure(fig)
            save_figure(fig)
            plt.close(fig)

    # To animate the model, use the SystemAnimation class
    # Pass the res(states) and systems you wish to animate
    simulation = SystemAnimation(
        res,
        sim.sys.pendulum_sys,
        sim.sys.muscle_sys,
        sim.sys.neural_sys)
    # To start the animation
    simulation.animate()

def exercise3b():
    """ Main function to run for Exercise 3.

    Parameters
    ----------
        None

    Returns
    -------
        None
    """
    # Define and Setup your pendulum model here
    # Check Pendulum.py for more details on Pendulum class
    P_params = PendulumParameters()  # Instantiate pendulum parameters
    P_params.L = 0.5  # To change the default length of the pendulum
    P_params.m = 1.  # To change the default mass of the pendulum
    pendulum = PendulumSystem(P_params)  # Instantiate Pendulum object

    #### CHECK OUT Pendulum.py to ADD PERTURBATIONS TO THE MODEL #####

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

    ##### Neural Network #####
    # The network consists of four neurons
    N_params = NetworkParameters()  # Instantiate default network parameters
    N_params.D = 2.  # To change a network parameter
    # Similarly to change w -> N_params.w = (4x4) array
    print(N_params.w)
    ############################# Exercise 3A  ######################
    N_params.w = np.transpose(np.asarray([[ 0, -1,  1, -1],
                                          [-1,  0, -1,  1],
                                          [-1,  0,  0,  0], 
                                          [ 0, -1,  0,  0]]))*5
    print(N_params.w, N_params.D, N_params.tau, N_params.b, N_params.exp )

    

    # Create a new neural network with above parameters
    neural_network = NeuralSystem(N_params)
    pylog.info('Neural system initialized \n {}'.format(
        N_params.showParameters()))

    # Create system of Pendulum, Muscles and neural network using SystemClass
    # Check System.py for more details on System class
    sys = System()  # Instantiate a new system
    sys.add_pendulum_system(pendulum)  # Add the pendulum model to the system
    sys.add_muscle_system(muscles)  # Add the muscle model to the system
    # Add the neural network to the system
    sys.add_neural_system(neural_network)

    ##### Time #####
    t_max = 4.  # Maximum simulation time
    time = np.arange(0., t_max, 0.001)  # Time vector

    ##### Model Initial Conditions #####
    x0_P = np.array([0., 0.])  # Pendulum initial condition

    # Muscle Model initial condition
    x0_M = np.array([0., M1.L_OPT, 0., M2.L_OPT])

    x0_N = np.array([-0.5, 1, 0.5, 1])  # Neural Network Initial Conditions

    x0 = np.concatenate((x0_P, x0_M, x0_N))  # System initial conditions

    ##### System Simulation #####
    # For more details on System Simulation check SystemSimulation.py
    # SystemSimulation is used to initialize the system and integrate
    # over time

    sim = SystemSimulation(sys)  # Instantiate Simulation object

    # Add external inputs to neural network

#    sim.add_external_inputs_to_network(np.ones((len(time), 4)))
    
#    wave_h1 = np.sin(time*3)*2               #makes a sinusoidal wave from 'time'
#    wave_h2 = np.sin(time*3 + np.pi)*1       #makes a sinusoidal wave from 'time'
#    
#    wave_h1[wave_h1<0] = 0      #formality of passing negative values to zero
#    wave_h2[wave_h2<0] = 0      #formality of passing negative values to zero
#    
#    act1 = wave_h1.reshape(len(time), 1) #makes a vertical array like act1
#    act2 = wave_h2.reshape(len(time), 1) #makes a vertical array like act1
#    print (type(float(len(time))))
#    l = (len(time))
#    l = int(l/2.0)
#    print(l)
    column1 = np.zeros(( int(len(time)/2.) , 1))
    column2 = np.ones(( int(len(time)/2. ), 1))
    column = np.concatenate([column1, column2])
    print (column)
   

#    ext_in = np.hstack((act1, column, act2, column))    
    ext_in = np.hstack((column, column, column, column))
    
    sim.add_external_inputs_to_network(ext_in)
    sim.initalize_system(x0, time)  # Initialize the system state

    # Integrate the system for the above initialized state and time
    sim.simulate()

    # Obtain the states of the system after integration
    # res is np.array [time, states]
    # states vector is in the same order as x0
    res = sim.results()

    # Obtain the states of the system after integration
    # res is np.array [time, states]
    # states vector is in the same order as x0
    res = sim.results()

    # In order to obtain internal states of the muscle
    # you can access the results attribute in the muscle class
    muscle1_results = sim.sys.muscle_sys.Muscle1.results
    muscle2_results = sim.sys.muscle_sys.Muscle2.results

     # Plotting the results: Position(phase) vs time
    plt.figure('Pendulum Phase & Stimulation')
    plt.title('Pendulum Position vs Time with Changing External Input')
    switch_index = int(len(time)/2.)
    plt.plot(res[0:switch_index-1, 0], res[0:switch_index-1,1], label = 'ExtIn = 0')   #to plot pendulum Position (phase) before external input
    plt.plot(res[switch_index:-1, 0], res[switch_index:-1,1], label = 'ExtIn = 1')   #to plot pendulum Position (phase) before external input
    
    plt.legend(loc = 'upper right')
#    plt.plot(res[:, 0], time)   #to plot position
#    plt.plot(res[:, 0], res[:, -5:-1])  # to Plot neurons' states
    plt.xlabel('time [s]')
    plt.ylabel('Position [rad]')
    plt.grid()

    # Plotting the results: Velocity vs Position (phase)
    plt.figure('Pendulum Vel v.s. Phase & Stimulation')
    plt.title('Pendulum Position vs. Velocity Phase Plot with Changing External Input')
    plt.plot(res[0:switch_index-1, 1], res[0:switch_index-1,2], label = 'ExtIn = 0')   #to plot pendulum Position (phase) before external input
    plt.plot(res[switch_index:-1, 1], res[switch_index:-1,2], label = 'ExtIn = 1')   #to plot pendulum Position (phase) before external input
    plt.xlabel('Position [rad]')
    plt.ylabel('Velocity [rad.s]')
    plt.grid()
    plt.legend(loc = 'upper right')
    
    # Plotting the results: Velocity vs time 
    plt.figure('Pendulum Velocity vs Time for Changing External Input')
    plt.title('Pendulum Velocity & Stimulation')
    plt.plot(res[0:switch_index-1, 0], res[0:switch_index-1,2], label = 'ExtIn = 0')   #to plot pendulum Velocity before external input
    plt.plot(res[switch_index:-1, 0], res[switch_index:-1,2], label = 'ExtIn = 1')
    plt.legend(loc = 'upper right')
    plt.xlabel('time [s]')
    plt.ylabel('Velocity [rad.s]')
    plt.grid()

    # Plotting the results: Output of the network
    plt.figure('Network output & Stimulation')
    plt.title('Network Output for Changing External Input')
    plt.plot(res[0:switch_index-1, 0], res[0:switch_index-1,-4],'m', label='neuron1')   #to plot Velocity vs Position
    plt.plot(res[0:switch_index-1, 0], res[0:switch_index-1,-3],'g', label='neuron2') 
    plt.plot(res[0:switch_index-1, 0], res[0:switch_index-1,-2],'b', label='neuron3') 
    plt.plot(res[0:switch_index-1, 0], res[0:switch_index-1,-1],'r', label='neuron4') 
    plt.plot(res[switch_index:-1, 0], res[switch_index:-1,-4], color='orchid')   #to plot Velocity vs Position
    plt.plot(res[switch_index:-1, 0], res[switch_index:-1,-3],color='forestgreen') 
    plt.plot(res[switch_index:-1, 0], res[switch_index:-1,-2], color='royalblue') 
    plt.plot(res[switch_index:-1, 0], res[switch_index:-1,-1], color='indianred') 
    plt.xlabel('time [s]')
    plt.ylabel('Stimulation ')
    plt.legend(loc='upper left')
    plt.grid()

    if DEFAULT["save_figures"] is False:
        plt.show()
    else:
        figures = plt.get_figlabels()
        pylog.debug("Saving figures:\n{}".format(figures))
        for fig in figures:
            plt.figure(fig)
            save_figure(fig)
            plt.close(fig)

    # To animate the model, use the SystemAnimation class
    # Pass the res(states) and systems you wish to animate
    simulation = SystemAnimation(
        res,
        sim.sys.pendulum_sys,
        sim.sys.muscle_sys,
        sim.sys.neural_sys)
    # To start the animation
    simulation.animate()



if __name__ == '__main__':
    from cmcpack import parse_args
    parse_args()
    exercise3()

