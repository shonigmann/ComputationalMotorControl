"""Plot results"""

import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from cmc_robot import ExperimentLogger
from save_figures import save_figures
from parse_args import save_plots


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


def main_9b(simulation_i, plot=True):
    """Main"""
    # Load data
    with np.load('logs/9b/simulation_{}.npz'.format(simulation_i)) as data:
        timestep = float(data["timestep"])
        amplitude = data["amplitudes"]
        phase_lag = data["phase_lag"]
        link_data = data["links"][:, 0, :]
        joints_data = data["joints"]
    times = np.arange(0, timestep*np.shape(link_data)[0], timestep)

    # Plot data
    plt.figure("Positions")
    plot_positions(times, link_data)

    #print(joints_data)

    # Show plots
    if plot:
        plt.show()
    else:
        save_figures()
            
def main_9bEnergy(simulation_i, Elements, Energy_matrix, plot=True):
    """Main"""
    # Load data
    with np.load("logs/9b/test_{}.npz".format(simulation_i)) as data:
#        timestep = float(data["timestep"])
#        amplitude = data["amplitudes"]
#        phase_lag = data["phase_lag"]
#        link_data = data["links"][:, 0, :]
        joints_data = data["joints"]
#    times = np.arange(0, timestep*np.shape(link_data)[0], timestep)
    
    #CALCULATE ENERGY
    torque_vector = joints_data[500:-1 , : , 3]
    d_angle = joints_data[ 501: , : , 0] - joints_data[ 500:-1 , : , 0]
    
    #print(np.shape(torque_vector))
    #print('Angle: ', np.shape(d_angle))
    
    Energy_joints = np.dot(abs(np.transpose(torque_vector)), abs(d_angle))
    #print(np.shape(Energy_joints))
    Energy_animal = np.trace(Energy_joints)
#    print('The energy of the animal ', simulation_i,' is: ', Energy_animal)
#    print('Elements: ', Elements)
    shape = np.shape(Energy_matrix)
    Energy_matrix = np.reshape(Energy_matrix, (1, Elements))
    Energy_matrix[0 , simulation_i] = Energy_animal
#    print(Energy_matrix)
    Energy_matrix = np.reshape(Energy_matrix, shape)
    print(Energy_matrix)
    np.savez('./logs/9b/Energies.npz', Energy=Energy_matrix)
    
    
    
#    # Plot data
#    plt.figure("Positions")
#    plot_positions(times, link_data)
#
#    #print(joints_data)
#
#    # Show plots
#    if plot:
#        plt.show()
#    else:
#            save_figures()

if __name__ == '__main__':
    #main(plot=not save_plots())
#    main_9bEnergy()
    main_9b(0)
    main_9b(1)
    main_9b(2)
    main_9b(3)
    
