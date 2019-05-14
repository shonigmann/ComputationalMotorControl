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


def plot9c(plot=True):
    """Plot for exercise 9c"""
    # Load data
    pathfile = 'logs/9c/'
    num_files = len([f for f in os.listdir(pathfile)])

    energy_plot = np.zeros((num_files-4, 2))

    clean_val = 200

    for i in range(num_files-4):
        with np.load(pathfile + 'simulation_{}.npz'.format(i)) as data:
            timestep = float(data["timestep"])
            amplitude_gradient = data["amplitude_gradient"]
            link_data = data["links"][:, 0, :]
            torque = data["joints"][:, :, 3]
            angular_vel = data["joints"][:, :, 1]
        times = np.arange(0, timestep * np.shape(link_data)[0], timestep)

        speed = np.linalg.norm(link_data[clean_val:], axis=1)/times[clean_val:]

        # Plot speed data
        plt.figure("Speed vs Gradient amplitude")
        plt.plot(times[clean_val:], speed, label='Gradient {}'.format(amplitude_gradient))
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("Mean speed [m/s]")
        plt.grid(True)

        energy_plot[i, 0] = amplitude_gradient
        energy_plot[i, 1] = np.sum(np.sum(torque[:, :10]*angular_vel[:, :10]))

    # Plot energy data
    plt.figure("Energy vs Gradient amplitude")
    plt.plot(energy_plot[:, 0], energy_plot[:, 1])
    plt.xlabel("Gradient [-]")
    plt.ylabel("Energy [J]")
    plt.grid(True)

    # Show plots
    if plot:
        plt.show()
    else:
        save_figures()


def main(plot=True):
    """Main"""
    plot9c()


if __name__ == '__main__':
    main(plot=not save_plots())

