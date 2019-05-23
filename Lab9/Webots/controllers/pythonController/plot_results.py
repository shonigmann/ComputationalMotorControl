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

def plot_turn_trajectory():
    """Plot positions"""
    for file in os.listdir('logs/9d1'):
        with np.load(os.path.join('logs/9d1/',file)) as data:
            #amplitude = data["amplitudes"]
            #phase_lag = data["phase_lag"]
            turn = data["turn"]*2
            link_data = data["links"][:, 0, :]
        
            # Plot data
            plt.figure("Trajectory")
            plt.plot(link_data[:, 0], link_data[:, 2], label="Drive Diff = %.2f" % turn)
            plt.xlabel("x [m]")
            plt.ylabel("z [m]")
            plt.legend()
            plt.axis("equal")
            plt.grid(True)

def plot_reverse_trajectory():
    """Plot positions"""
    for file in os.listdir('logs/9d2'):
        with np.load(os.path.join('logs/9d2/',file)) as data:
            #amplitude = data["amplitudes"]
            #phase_lag = data["phase_lag"]
            turn = data["turn"]*2
            reverse = data["reverse"]
            if(reverse!=0.0):
                reversed = "Rev:"
            else:
                reversed = "Fwd:"
                
            link_data = data["links"][:, 0, :]
        
            # Plot data
            plt.figure("Trajectory")
            plt.plot(link_data[:, 0], link_data[:, 2], label=reversed+" ; Drive Diff = %.2f" % turn)
            plt.xlabel("x [m]")
            plt.ylabel("z [m]")
            plt.legend()
            plt.axis("equal")
            plt.grid(True)


def plot_9c(plot=True):
    """Plot for exercise 9c"""
    # Load data
    pathfile = 'logs/9c/random/'
    num_files = len([f for f in os.listdir(pathfile)])

    gradient = np.zeros(num_files)
    speed_plot = np.zeros((num_files, 3))
    energy_plot = np.zeros((num_files, 3))
    speed_energy = np.zeros((num_files, 3))

    nb_body_joints = 10
    clean_val = 500

    for i in range(num_files):
        with np.load(pathfile + 'simulation_{}.npz'.format(i)) as data:
            timestep = float(data["timestep"])
            rhead = data["rhead"]
            rtail = data["rtail"]
            link_data = data["links"][:, 0, :]
            angle = data["joints"][:, :, 0]
            torque = data["joints"][:, :, 3]

        times = np.arange(0, timestep * np.shape(link_data)[0], timestep)
        speed = np.linalg.norm(link_data[clean_val:], axis=1)/times[clean_val:]

        # Plot sum energy over a simulation
        gradient[i] = (rhead - rtail) / nb_body_joints
        tot_energy = np.sum(torque[clean_val:-1, :nb_body_joints]*(angle[clean_val + 1:, :nb_body_joints]-angle[clean_val:-1, :nb_body_joints]))

        energy_plot[i, 0] = rhead
        energy_plot[i, 1] = rtail
        energy_plot[i, 2] = tot_energy

        speed_plot[i, 0] = rhead
        speed_plot[i, 1] = rtail
        speed_plot[i, 2] = speed[-1]

        speed_energy[i, 0] = rhead
        speed_energy[i, 1] = rtail
        speed_energy[i, 2] = speed[-1]/tot_energy

    # Plot energy data in 2D
    plt.figure("Energy vs Gradient amplitude")
    labels = ['rhead', 'rtail', 'Energy [J]']
    plot_2d(energy_plot, labels)

    plt.figure("Speed vs Gradient amplitude")
    labels = ['rhead', 'rtail', 'Mean speed [m/s]']
    plot_2d(speed_plot, labels)

    plt.figure("Speed/Energy vs Gradient amplitude")
    labels = ['rhead', 'rtail', 'Mean speed / energy [m/sJ]']
    plot_2d(speed_energy, labels)

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
            n_links = len(data["links"][0,:,0])
            joint_data = data["joints"]
            timestep = float(data["timestep"])
            
            times = np.arange(0, timestep*np.shape(joint_data)[0], timestep)
            
            plt.figure("Trajectory")
                        
            for i in range(n_links):
                plt.plot(times, joint_data[:, i , 0]+i*np.pi, label = "x%d" % (i+1))
            plt.xlabel("t [s]")
            plt.ylabel("link phase [rad]")
            plt.legend()
    plt.show()


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
        plot_9f()
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
    plot_9c()

