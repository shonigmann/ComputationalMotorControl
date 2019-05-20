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
    #Grid points
    test = np.linspace( 0.5, 1.3, num=3)
    test2 = np.linspace(0.25, 0.5, num=3)
    Energy_matrix = np.zeros( ( len(test), len(test2) ) )
    Elements = len(test)*len(test2)
    
    parameter_set = [
        SimulationParameters(
            simulation_duration=12,
            drive=1,
            amplitudes= test_i,
            #phase_lag=np.zeros(n_joints),
            #nominal_amplitudes = [1,2,3], #test
            phase_lag = np.ones(n_joints)*test2_i,#[1,2,3], #test2
            turn=0,
           # print(phase_lag[1])
            # ...
        )
        for test_i in test
            for test2_i in test2
        #for test2 in np.linspace(0.01*math.pi, 1*math.pi, num=3)
    ]
    np.savez('./logs/9b/Energies.npz', Amplitudes=test, PhaseLag=test2)
    #Integral calculation: (Riemann Integral)
    
    

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
        
#        #CALCULATE ENERGY:
#        
#        #Load the files in order to read the data of each iteration and compute
#        #the energy or speed. (IMP: right now 250 iters=1 seg): in order to know
#        #how much to take out until salamander swims correctly. 
#        file = np.load("./logs/9b/test_{}.npz".format(simulation_i))
#        print(np.shape(file['joints']))
#        print(len(file['joints'][0])) #10
#        print(len(file['joints'][1])) #10
#        print(len(file['joints'][:])) #750
#        print(len(file['joints'][:,0,0])) #750
#        
#        #Go over every joint and compute the product: Torque*Angle
#        
#        Energy_joint = 0  #initialize the energy for every joint to 0 value.
#        Energy_animal = 0  #initialize the energy of the whole animal to 0. 
#        
#        # ['joints'][1] goes through all 10 joints. 
#        for i_joint in range(len(file['joints'][0,:,0])):
#            
#            print('i_joint: ', i_joint)
#            #print(len(file['joints'][1][0:5]))
#            #print(file['joints'][1][0:5])
#            
#             
#            
#            # ['joints'][0] goes through all iterations, i.e. 10seg = 2500 iters
#            for i_iter,nothing2 in enumerate(file['joints'][:,0,0]):    
#                #print('i_iter: ', i_iter)
#                #assign the torque value of the iteration
#                torque = file['joints'][i_iter, i_joint, 3]   
#                #print(torque)
#                #assign the increment of the angle displaced (it+1) - (it)
#                d_angle = file['joints'][i_iter+1, i_joint, 0] - file['joints'][i_iter, i_joint, 0]  
#                
#                #compute value of energy for every iteration, i.e. d_phi
#                Energy_iter = abs(torque * d_angle)
#                #append by summing to the previous value. Energy of joint for all iters 
#                Energy_joint = Energy_joint + Energy_iter   
#                #when the i_iter is one before the last: BREAK. Due to d_angle
#                
#                if i_iter+2 == len(file['joints'][:,0,0]):
#                    print('Energy of the joint: ',Energy_joint)
#                    break
#
#            Energy_animal = Energy_animal + Energy_joint
#            print(Energy_animal)
#    print('this is the energy of the animal: ', Energy_animal)
#    print(Energy_animal)
        
#NOTES OF WORK TO DO: 
#I should break before the last iter!!
#I should save the energy value somewhere to do a 2D plot after. Idea is: 2D for every set of phase lag and oscillation amplitude. 


#ID_J = {
#    "position": 0,
#    "velocity": 1,
#    "cmd": 2,
#    "torque": 3,
#    "torque_fb": 4,   --> WHAT IS TORQUE_FB?? 
#    "output": 5
#}
            
#        print(np.shape(file['joints']))   #(2500 iters en 10 seg,       10 joints    , 6 ID_J)
#        print('inic check')
#        #print(len(file['joints'][0]))
#        #print(file['joints'][:,3,0])
#        print('end check')
#        
        print('Simulation ', simulation_i,' of ', Elements)
#        print(Energy_matrix)
#        print(simulation_i)
        plot_results.main_9bEnergy(simulation_i, Elements, Energy_matrix)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        