"""Plot results"""

import os.path
import numpy as np
import math
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from cmc_robot import ExperimentLogger
from save_figures import save_figures
from parse_args import save_plots
import os

def plot_positions(times, link_data):
    """Plot positions"""
    for i, data in enumerate(link_data.T):
        plt.plot(times, data, label=["x", "y", "z"][i])
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Distance [m]")
    plt.grid(True)


def plot_trajectory(link_data):
    """Plot positions"""
    plt.plot(link_data[:, 0], link_data[:, 2])
    plt.xlabel("x [m]")
    plt.ylabel("z [m]")
    plt.axis("equal")
    plt.grid(True)

def plot_spine(timestep, joint_data, turn_rev, num_iter=5, plot_name="", subplot=0):
    # Plot spine angles
    
    #cut out transient
    joint_data = joint_data[1000:-1, :]
    
    times = np.arange(0, timestep*np.shape(joint_data)[0], timestep)
    
    plt.figure("Spine Angles "+plot_name)
    if subplot>0:
        plt.subplot(2,1,subplot)
    L_link = .1 #link length in cm
    x_spacing = .5
    index_step = np.shape(joint_data)[0]//num_iter
    
    #plot spine from fixed head position
    for i in range(num_iter):
        index = i*index_step
        x_offset = i*x_spacing
        joint_state = joint_data[index,0:10]
        
        spine_x = np.zeros([len(joint_state)+1,1]) + x_offset
        spine_y = np.zeros([len(joint_state)+1,1])
        for j in range(10):
            spine_x[j+1] = spine_x[j]+L_link*np.sin(joint_state[j])
            spine_y[j+1] = spine_y[j]-L_link*np.cos(joint_state[j])
        
        plt.plot(spine_x,spine_y, label="t = %.1f" % times[index])
        plt.plot(spine_x[0]*np.ones([2,1]), [spine_y[0],spine_y[-1]], color='r', linestyle='--')
        
  
    plt.title("Spine Angles ("+turn_rev+")")
    plt.xlabel("x [m]")
    plt.ylabel("z [m]")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)

def plot9d():
    plot9d1()
    plot9d2()

def plot9d1():
    #load files in folder
    file_number = 1
    for file in os.listdir('logs/9d1'):
        with np.load(os.path.join('logs/9d1/',file)) as data:
            #amplitude = data["amplitudes"]
            #phase_lag = data["phase_lag"]
            turn = data["turn"]*2
            link_data = data["links"][:, 0, :]
            joint_data = data["joints"][:,:,0]
        
            # Plot trajectory
            plt.figure("9d1: Trajectory")
            plt.plot(link_data[:, 0], link_data[:, 2], label="Drive Diff = %.2f" % turn)
            plt.xlabel("x [m]")
            plt.ylabel("z [m]")
            plt.legend()
            plt.axis("equal")
            plt.grid(True)
            
            # Plot spine angles
            #only plot 1 time per direction...   
            if file_number == 1:
                timestep = float(data["timestep"])
                turn_rev="Drive Diff = %.2f" % turn
                plot_spine(timestep, joint_data, turn_rev, 8, "d1", 1)
            if file_number == len(os.listdir('logs/9d1')):
                timestep = float(data["timestep"])
                turn_rev="Drive Diff = %.2f" % turn
                plot_spine(timestep, joint_data, turn_rev, 8, "d1", 2)
                
            file_number = file_number + 1


def plot9d2():
    """Plot positions"""
    epsilon = 0.0001
    subplot = 1
    for file in os.listdir('logs/9d2'):
        with np.load(os.path.join('logs/9d2/',file)) as data:
            #amplitude = data["amplitudes"]
            #phase_lag = data["phase_lag"]
            turn = data["turn"]*2
            reverse = data["reverse"]
            if(reverse!=0.0):
                reversed = "Rev:"
                rev_title = "Reverse"
            else:
                reversed = "Fwd:"
                rev_title = "Forward"
                
            link_data = data["links"][:, 0, :]
            joint_data = data["joints"][:,:,0]
        
            # Plot data
            plt.figure("9d2: Trajectory")
            plt.plot(link_data[:, 0], link_data[:, 2], label=reversed+" Drive Diff = %.2f" % turn)
            plt.xlabel("x [m]")
            plt.ylabel("z [m]")
            plt.legend()
            plt.axis("equal")
            plt.grid(True)
            
            # Plot spine angles            
            #only plot 1 time per direction...   
            if abs(turn) < epsilon:
                timestep = float(data["timestep"])
                plot_spine(timestep, joint_data, rev_title, 8, "d2", subplot)
                subplot = subplot+1


def plot_9c(plot=True):
    """Plot for exercise 9c"""
    # Load data
    pathfile = 'logs/9b/'
    num_files = len([f for f in os.listdir(pathfile)])

    gradient = np.zeros(num_files)
    energy_plot = np.zeros((5, 3))

    clean_val = 200

    for i in range(num_files):
        with np.load(pathfile + 'simulation_{}.npz'.format(i)) as data:
            timestep = float(data["timestep"])
            amplitude_gradient = data["amplitude_gradient"]
            link_data = data["links"][:, 0, :]
            angle = data["joints"][:, :, 0]
            angular_vel = data["joints"][:, :, 1]
            torque = data["joints"][:, :, 3]

        times = np.arange(0, timestep * np.shape(link_data)[0], timestep)
        speed = np.linalg.norm(link_data[clean_val:], axis=1)/times[clean_val:]

        # Plot speed data
        plt.figure("Speed vs Gradient amplitude")
        plt.plot(times[clean_val:], speed, label='Gradient {}'.format(amplitude_gradient))
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("Mean speed [m/s]")
        plt.grid(True)

        # Plot sum energy over a simulation
        gradient[i] = amplitude_gradient
        energy_plot[i] = np.sum(np.sum(np.abs(torque[:, :10]*angle[:, :10]), axis=1))

    """if pathfile is ('logs/9c/head_gradient_only/' or 'logs/9c/tail_gradient_only/'):
        # Plot energy data
        plt.figure("Energy vs Gradient amplitude")
        plt.plot(gradient, energy_plot)
        plt.legend()
        plt.xlabel("Gradient [-]")
        plt.ylabel("Energy [J]")
        plt.grid(True)"""

    energy_plot[:, 0] = np.linspace(0, 5, num=5)
    energy_plot[:, 1] = np.linspace(0, 5, num=5)
    energy_plot[:, 2] = np.linspace(0, 5, num=5)

    plot_2d(energy_plot, ['x', 'y'])

    # Show plots
    if plot:
        plt.show()
    else:
        save_figures()

            
def plot_9f():
    """Plot positions"""
    for file in os.listdir('logs/9f'):
        with np.load(os.path.join('logs/9f/', file)) as data:
            #amplitude = data["amplitudes"]
            #phase_lag = data["phase_lag"]        
            # Plot data
            n_joints = len(data["joints"][0,:,0])
            joint_data = data["joints"]
            timestep = float(data["timestep"])
            
            times = np.arange(0, timestep*np.shape(joint_data[1500:2500,:,:])[0], timestep)
            
            plt.figure("Phase Differences for: "+file)
            for j in range(2):
                for i in range((n_joints-4)//2):
                    #plt.plot(times, joint_data[1500:2500, i , 0]+i*np.pi, label = "x%d" % (i+1))
                    plt.subplot(2,1,j+1)
                    if i>0: 
                        plt.plot(times, joint_data[1500:2500, i+j*5 , 0]- joint_data[1500:2500,i+j*5-1,0], label = "x%d" % (i+j*5+1))
                    else:
                        plt.plot(times, joint_data[1500:2500, i+j*5 , 0]- joint_data[1500:2500,i+j*5,0], label = "x%d" % (i+j*5+1))
                plt.xlabel("t [s]")
                plt.ylabel("link phase lag [rad]")
                plt.legend()
                
            plt.figure("Phase for: "+file)
            for j in range(2):
                for i in range((n_joints-4)//2):
                    #plt.plot(times, joint_data[1500:2500, i , 0]+i*np.pi, label = "x%d" % (i+1))
                    plt.subplot(2,1,j+1)
                    if i>0: 
                        plt.plot(times, joint_data[1500:2500, i+j*5 , 0], label = "x%d" % (i+j*5+1))
                    else:
                        plt.plot(times, joint_data[1500:2500, i+j*5 , 0], label = "x%d" % (i+j*5+1))
                plt.xlabel("t [s]")
                plt.ylabel("link phase [rad]")
                plt.legend()
                  


def plot_9g():
    """Plot positions"""
    pathfile = 'logs/9g/'

    for file in os.listdir(pathfile):
        with np.load(pathfile + file) as data:
            links = data["links"][:, 0, :]
            joint_data = data["joints"]
            timestep = float(data["timestep"])

            times = np.arange(0, timestep * np.shape(joint_data)[0], timestep)

            plt.figure("Trajectory")
            plot_positions(times, link_data=links)
            plt.show()

            
def plot_2d(results, labels, n_data=300, log=False, cmap=None):
    """Plot result

    results - The results are given as a 2d array of dimensions [N, 3].

    labels - The labels should be a list of three string for the xlabel, the
    ylabel and zlabel (in that order).

    n_data - Represents the number of points used along x and y to draw the plot

    log - Set log to True for logarithmic scale.

    cmap - You can set the color palette with cmap. For example,
    set cmap='nipy_spectral' for high constrast results.

    """
    xnew = np.linspace(min(results[:, 0]), max(results[:, 0]), n_data)
    ynew = np.linspace(min(results[:, 1]), max(results[:, 1]), n_data)
    grid_x, grid_y = np.meshgrid(xnew, ynew)
    results_interp = griddata(
        (results[:, 0], results[:, 1]), results[:, 2],
        (grid_x, grid_y),
        method='linear'  # nearest, cubic
    )
    extent = (
        min(xnew), max(xnew),
        min(ynew), max(ynew)
    )
    plt.plot(results[:, 0], results[:, 1], "r.")
    imgplot = plt.imshow(
        results_interp,
        extent=extent,
        aspect='auto',
        origin='lower',
        interpolation="none",
        norm=LogNorm() if log else None
    )
    if cmap is not None:
        imgplot.set_cmap(cmap)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    cbar = plt.colorbar()
    cbar.set_label(labels[2])


def main(plot=True, file=None):
    """Main"""
    # Load data
    if file is None:
#        for file in os.listdir('logs/9d1'):
#            with np.load(os.path.join('logs/9d1/',file)) as data:
#                timestep = float(data["timestep"])
#                #amplitude = data["amplitudes"]
#                #phase_lag = data["phase_lag"]
#                link_data = data["links"][:, 0, :]
#                joints_data = data["joints"]
#                times = np.arange(0, timestep*np.shape(link_data)[0], timestep)
#            
#                # Plot data
#                plt.figure("Positions")
#                plot_positions(times, link_data)
#            
#                plt.figure("Trajectory")
#                plot_trajectory(link_data)
#                print(joints_data)
        #plot_turn_trajectory()
        #plot_reverse_trajectory()
        plot9d()
    else:
        with np.load(file) as data:
            timestep = float(data["timestep"])
            #amplitude = data["amplitudes"]
            #phase_lag = data["phase_lag"]
            link_data = data["links"][:, 0, :]
            joints_data = data["joints"]
            times = np.arange(0, timestep*np.shape(link_data)[0], timestep)
        
            # Plot data
            plt.figure("Positions")
            plot_positions(times, link_data)
        
            plt.figure("Trajectory")
            plot_trajectory(link_data)
            print(joints_data)


if __name__ == '__main__':
    #main(plot=not save_plots())
    main()

