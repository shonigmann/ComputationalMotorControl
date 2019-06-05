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


def plot_trajectoryDirect():
    """Plot positions"""
    pathfile = 'logs/9b/'
    with np.load(pathfile + 'test_7.npz') as data:
        #            timestep = float(data["timestep"])
        #            rhead = data["rhead"]
        #            rtail = data["rtail"]
        link_data = data["links"][:, 0, :]
    #            angle = data["joints"][:, :, 0]
    #            torque = data["joints"][:, :, 3]

    plt.plot(link_data[:, 0], link_data[:, 2])
    plt.xlabel("x [m]")
    plt.ylabel("z [m]")
    plt.axis("equal")
    plt.grid(True)


def plot_spine(timestep, joint_data, turn_rev, num_iter=5, plot_name="", subplot=0):
    # Plot spine angles
    # cut out transient
    num_points = np.shape(joint_data)[0]
    if num_points > 500:
        joint_data = joint_data[(num_points - 500):-1, :]

    times = np.arange(0, timestep * np.shape(joint_data)[0], timestep)

    plt.figure("Spine Frames " + plot_name)
    if subplot > 0:
        plt.subplot(2, 1, subplot)

    L_link = .1  # link length in cm
    x_spacing = .5
    index_step = np.shape(joint_data)[0] // num_iter

    # plot spine from fixed head position
    for i in range(num_iter):
        index = i * index_step
        x_offset = i * x_spacing
        joint_state = joint_data[index, 0:10]

        spine_x = np.zeros([len(joint_state) + 1, 1]) + x_offset
        spine_y = np.zeros([len(joint_state) + 1, 1])
        for j in range(10):
            spine_x[j + 1] = spine_x[j] + L_link * np.sin(joint_state[j])
            spine_y[j + 1] = spine_y[j] - L_link * np.cos(joint_state[j])

        plt.plot(spine_x, spine_y, label="t = %.1f" % times[index])
        plt.plot(spine_x[0] * np.ones([2, 1]), [spine_y[0], spine_y[-1]], color='r', linestyle='--')

    plt.title("Spine Angle Frames (" + turn_rev + ")")
    plt.xlabel("x [m]")
    plt.ylabel("z [m]")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)

    # integrate spine angle to uncover bias towards one side or another
    # first find zero as starting point
    integrated_angles = np.zeros([10, 1])
    period_indices = np.zeros([10, 2])

    for i in range(10):
        period_time = 0
        zeros = (np.multiply(joint_data[0:-1, i], joint_data[1:, i]) < 0)
        j = 0
        while zeros[j] != 1 or joint_data[j, i] < 0:
            j += 1
        zero_count = 0
        period_time -= times[j]
        period_indices[i, 0] = j
        # then integrate over one period
        while zero_count < 2:
            integrated_angles[i] += joint_data[j, i] * 360 / 2 / np.pi * timestep  # deg/s
            j += 1
            if zeros[j] == 1:
                zero_count += 1

        period_indices[i, 1] = j
        period_time += times[j]

    avg_turn_bias = np.average(integrated_angles) / period_time  # average offset in deg
    # print(turn_rev)
    # print(integrated_angles)
    print(period_time)
    plt.figure("Spine Angle Time Series" + plot_name)
    if subplot == 2:
        color_ = 'r'
        offset_time = 0.456
    else:
        color_ = 'b'
        offset_time = 0.0
    label_ = turn_rev

    times = np.arange(0, timestep * np.shape(joint_data)[0] * 2, timestep)
    # plot timeseries for each spine angle
    subplot = max(subplot, 1)
    for i in range(10):

        # phase shift all timeseries so that they start with the 1st joint
        tpi = period_indices[i, :]
        if period_indices[i, 0] < period_indices[0, 0] and turn_rev != "Reverse":
            tpi = tpi + period_indices[0, 1] - period_indices[0, 0]
        elif turn_rev == "Reverse":
            if i == 0:
                tpi = tpi  # + period_indices[0,1]-period_indices[0,0]
            if period_indices[i, 0] < period_indices[-1, 0]:
                tpi = tpi + period_indices[0, 1] - period_indices[0, 0]

        # plt.subplot(10,1,i+1)
        plt.subplot(10, 2, i * 2 + (subplot))
        joint_data_ = joint_data[int(period_indices[i, 0]):int(period_indices[i, 1]), i]
        times_ = times[int(tpi[0]):(int(tpi[0]) + np.shape(joint_data_)[0])]

        plt.plot(times_, joint_data_, color=color_, label=label_)
        if turn_rev != "Reverse":
            plt.plot([times[int(period_indices[-1, 0])], times[int(period_indices[-1, 0])] + period_time * 2],
                     np.zeros([2, 1]), color='k', linestyle='--')
        else:
            plt.plot([times[int(period_indices[0, 0])], times[int(period_indices[0, 0])] + period_time * 2],
                     np.zeros([2, 1]), color='k', linestyle='--')
        if i == 0:
            plt.title(
                "Spine Angle Timeseries for Different Turn Directions")  # \n %.1f deg avg. angle bias over period" % (turn_rev, avg_turn_bias))
        if i == 5:
            plt.ylabel("Joint Angle [rad]")
        if i < 9:
            plt.xticks([])
        plt.ylabel("J%d" % i)
    plt.xlabel("t [s]")
    plt.legend()


def plot_9d():
    plot_9d1()
    plot_9d2()


def plot_9d1():
    # load files in folder
    file_number = 1
    for file in os.listdir('logs/9d1'):
        with np.load(os.path.join('logs/9d1/', file)) as data:
            # amplitude = data["amplitudes"]
            # phase_lag = data["phase_lag"]
            turn = data["turn"] * 2
            link_data = data["links"][:, 0, :]
            joint_data = data["joints"][:, :, 0]

            # Plot trajectory
            plt.figure("9d1: Trajectory")
            plt.plot(link_data[400:, 0], link_data[400:, 2], label="Drive Diff = %.2f" % turn)
            plt.xlabel("x [m]")
            plt.ylabel("z [m]")
            plt.legend()
            plt.axis("equal")
            plt.grid(True)

            # Plot spine angles
            # only plot 1 time per direction...
            if file_number == 1:
                timestep = float(data["timestep"])
                turn_rev = "Drive Diff = %.2f" % turn
                plot_spine(timestep, joint_data[78:], turn_rev, 8, "d1", 1)
            if file_number == len(os.listdir('logs/9d1')):
                timestep = float(data["timestep"])
                turn_rev = "Drive Diff = %.2f" % turn
                plot_spine(timestep, joint_data[194:], turn_rev, 8, "d1", 2)

            # if file_number==1:
            # torques = data["joints"][500:-1,:,3]
            # angle_changes = data["joints"][501:,:,0]-data["joints"][500:-1,:,0]
            # energy = np.sum(np.multiply(torques,angle_changes))
            # print("energy:")
            # print(energy)
            file_number = file_number + 1


def plot_9d2():
    """Plot positions"""
    epsilon = 0.0001
    subplot = 1
    for file in os.listdir('logs/9d2'):
        with np.load(os.path.join('logs/9d2/', file)) as data:
            # amplitude = data["amplitudes"]
            # phase_lag = data["phase_lag"]
            turn = data["turn"] * 2
            reverse = data["reverse"]
            if (reverse != 0.0):
                reversed = "Rev:"
                rev_title = "Reverse"
            else:
                reversed = "Fwd:"
                rev_title = "Forward"

            link_data = data["links"][:, 0, :]
            joint_data = data["joints"][:, :, 0]

            # Plot data
            plt.figure("9d2: Trajectory")
            plt.plot(link_data[400:, 0], link_data[400:, 2], label=reversed + " Drive Diff = %.2f" % turn)
            plt.xlabel("x [m]")
            plt.ylabel("z [m]")
            plt.legend()
            plt.axis("equal")
            plt.grid(True)

            # Plot spine angles            
            # only plot 1 time per direction...
            if abs(turn) < epsilon:
                timestep = float(data["timestep"])
                plot_spine(timestep, joint_data, rev_title, 8, "d2", subplot)
                subplot = subplot + 1


def calc_9d_energy():
    # load files in folder
    file_number = 1
    for file in os.listdir('logs/9b'):
        with np.load(os.path.join('logs/9b/', file)) as data:
            link_data = data["links"][:, 0, :]
            joint_data = data["joints"][:, :, 0]

            torques = data["joints"][500:-1, :, 3]
            angle_changes = data["joints"][501:, :, 0] - data["joints"][500:-1, :, 0]
            energy = np.sum(np.multiply(torques, angle_changes))
            print("energy:")
            print(energy)
            file_number = file_number + 1


def plot_9b(plot=True):
    """Plot for exercise 9c"""
    # Load data
    pathfile = 'logs/9b/dr6_bPhases7_42/'
    num_files = len([f for f in os.listdir(pathfile)])

    gradient = np.zeros(num_files)

    energy_plot = np.zeros((num_files, 3))
    speed_plot = np.zeros((num_files, 3))
    cost_transport = np.zeros((num_files, 3))

    nb_body_joints = 10
    clean_val = 500

    for i in range(num_files):
        with np.load(pathfile + 'test_{}.npz'.format(i)) as data:
            timestep = float(data["timestep"])
            link_data = data["links"][:, 0, :]
            angle = data["joints"][:, :, 0]
            torque = data["joints"][:, :, 3]
            nominal_ampl = data['nominal_amplitudes']
            body_phase_bias = data['body_phase_bias']

        # Speed calculation
        times = np.arange(0, timestep * np.shape(link_data)[0], timestep)
        speed = np.linalg.norm(link_data[-1] - link_data[clean_val]) / (times[-1] - times[clean_val])

        # Plot sum energy over a simulation
        tot_energy = np.sum(torque[clean_val:-1, :nb_body_joints] * (
                    angle[clean_val + 1:, :nb_body_joints] - angle[clean_val:-1, :nb_body_joints]))
        energy_plot[i] = [nominal_ampl, body_phase_bias, tot_energy]
        speed_plot[i] = [nominal_ampl, body_phase_bias, speed]

        cost = np.linalg.norm(link_data[-1] - link_data[clean_val]) / tot_energy
        print(link_data[-1])
        print(cost)
        cost_transport[i] = [nominal_ampl, body_phase_bias, cost]

    print(energy_plot)
    print(speed_plot)

    #    # Plot energy data in 2D
    name1 = "Energy in grid: Amplitude vs phase"
    plt.figure(name1)
    plt.title(name1)
    labels = ['CPG Amplitude', 'CPG body phase bias [rad]', 'Energy [J]']
    plot_2d(energy_plot, labels)

    name2 = "Speed in grid: Amplitude vs phase"
    plt.figure(name2)
    plt.title(name2)
    labels = ['CPG Amplitude', 'CPG body phase bias [rad]', 'Speed [m/s]']
    plot_2d(speed_plot, labels)
    #
    name3 = "Cost of transport in grid: Amplitude vs phase"
    plt.figure(name3)
    plt.title(name3)
    labels = ['CPG Amplitude', 'CPG body phase bias [rad]', 'transport cost [m/s/J]']
    plot_2d(cost_transport, labels)

    # Show plots
    if plot:
        plt.show()
    else:
        save_figures()


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
        speed = np.linalg.norm(link_data[clean_val:], axis=1) / times[clean_val:]

        # Plot sum energy over a simulation
        gradient[i] = (rhead - rtail) / nb_body_joints
        tot_energy = np.sum(torque[clean_val:-1, :nb_body_joints] * (
                    angle[clean_val + 1:, :nb_body_joints] - angle[clean_val:-1, :nb_body_joints]))

        energy_plot[i, 0] = rhead
        energy_plot[i, 1] = rtail
        energy_plot[i, 2] = tot_energy

        speed_plot[i, 0] = rhead
        speed_plot[i, 1] = rtail
        speed_plot[i, 2] = speed[-1]

        speed_energy[i, 0] = rhead
        speed_energy[i, 1] = rtail
        speed_energy[i, 2] = speed[-1] / tot_energy

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
            # amplitude = data["amplitudes"]
            # phase_lag = data["phase_lag"]
            # Plot data
            n_joints = len(data["joints"][0, :, 0])
            joint_data = data["joints"]
            timestep = float(data["timestep"])

            times = np.arange(0, timestep * np.shape(joint_data[1500:2500, :, :])[0], timestep)

            plt.figure("Phase Differences for: " + file)
            for j in range(2):
                for i in range((n_joints - 4) // 2):
                    # plt.plot(times, joint_data[1500:2500, i , 0]+i*np.pi, label = "x%d" % (i+1))
                    plt.subplot(2, 1, j + 1)
                    if i > 0:
                        plt.plot(times, joint_data[1500:2500, i + j * 5, 0] - joint_data[1500:2500, i + j * 5 - 1, 0],
                                 label="x%d" % (i + j * 5 + 1))
                    else:
                        plt.plot(times, joint_data[1500:2500, i + j * 5, 0] - joint_data[1500:2500, i + j * 5, 0],
                                 label="x%d" % (i + j * 5 + 1))
                plt.xlabel("t [s]")
                plt.ylabel("link phase lag [rad]")
                plt.legend()

            plt.figure("Phase for: " + file)
            for j in range(2):
                for i in range((n_joints - 4) // 2):
                    # plt.plot(times, joint_data[1500:2500, i , 0]+i*np.pi, label = "x%d" % (i+1))
                    plt.subplot(2, 1, j + 1)
                    if i > 0:
                        plt.plot(times, joint_data[1500:2500, i + j * 5, 0], label="x%d" % (i + j * 5 + 1))
                    else:
                        plt.plot(times, joint_data[1500:2500, i + j * 5, 0], label="x%d" % (i + j * 5 + 1))
                plt.xlabel("t [s]")
                plt.ylabel("link phase [rad]")
                plt.legend()


def plot_9f_network():
    """Plot positions"""
    subplot = 1
    color = 'b'
    title = "Swimming"
    for file in os.listdir('logs/9f'):
        with np.load(os.path.join('logs/9f/', file)) as data:

            joints = data["joints"]

            # Plot data
            n_links = (len(joints[0, :, 0])) - 4
            network_output = joints[1500:2500, :, 2]
            timestep = float(data["timestep"])

            times = np.arange(0, timestep * np.shape(network_output)[0], timestep)

            for i in range(n_links):
                plt.subplot(10, 2, i * 2 + (subplot))

                plt.plot(times, network_output[:, i], color=color, label=title)

                plt.ylabel("J%d" % i)

                if i == 0:
                    plt.title(
                        "Joint Commands During %s" % title)  # \n %.1f deg avg. angle bias over period" % (turn_rev, avg_turn_bias))
                if i == 5:
                    plt.ylabel("Joint Command\nJ%d" % i)
                if i < 9:
                    plt.xticks([])
            plt.xlabel("t [s]")
            plt.legend()

            subplot = subplot + 1
            color = 'r'
            title = "Walking"


def plot_9f3():
    plt.figure("Salamander Walking with Different Spine-Limb Phase Offsets")
    num_files = 9
    i = 0
    clean_val = 1000
    speeds = np.zeros([num_files, 1])
    biases = np.zeros([num_files, 1])
    for file in os.listdir('logs/9f3'):
        with np.load(os.path.join('logs/9f3/', file)) as data:
            link_data = data["links"][:, 0, :]
            timestep = float(data["timestep"])

            times = np.arange(0, timestep * np.shape(link_data)[0], timestep)
            speed = np.linalg.norm(link_data[clean_val, :] - link_data[-1, :]) / (times[-1] - times[clean_val])
            speeds[i] = speed
            temp_bias = data["body_limb_phase_bias"]
            biases[i] = temp_bias
            i += 1

    plt.plot(biases, speeds)

    plt.title("Walking with Different Spine-Limb Phase Offsets")
    plt.xlabel("Body-Limb Phase Offset [rad]")
    plt.ylabel("Walking Speed [m/s]")


def plot_9f4():
    # plt.figure("Salamander Walking with Different Spine-Limb Phase Offsets")
    fig, ax1 = plt.subplots()
    num_files = 15
    i = 0
    clean_val = 1000
    speeds = np.zeros([num_files, 1])
    body_amp = np.zeros([num_files, 1])
    powers = np.zeros([num_files, 1])

    for file in os.listdir('logs/9f4'):
        with np.load(os.path.join('logs/9f4/', file)) as data:
            link_data = data["links"][:, 0, :]
            timestep = float(data["timestep"])

            times = np.arange(0, timestep * np.shape(link_data)[0], timestep)
            speed = np.linalg.norm(link_data[clean_val, :] - link_data[-1, :]) / (times[-1] - times[clean_val])
            speeds[i] = speed
            body_amplitude = data["nominal_amplitudes"]
            body_amp[i] = body_amplitude

            torques = data["joints"][clean_val:, :, 3]
            joint_vels = data["joints"][(clean_val):, :, 1]
            avg_power = np.average(np.multiply(torques, joint_vels))
            powers[i] = avg_power
            i += 1

    cost_of_transport = np.divide(speeds, powers)

    ax1.plot(body_amp, speeds, color='b', label='Speed')
    plt.title("Walking with Different Spine Curvatures")
    ax1.set_xlabel("Body Amplitude [rad]")
    ax1.set_ylabel("Walking Speed [m/s]")
    plt.legend()

    ax2 = ax1.twinx()
    ax2.set_ylabel("Cost of Transport (m/J))")
    ax2.plot(body_amp, cost_of_transport, color='r', label='COT')
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
        # plot_9d()
        # calc_9d_energy()
        # plot_9f()
        # plot_9f_network()
        # plot_9f3()
        # plot_9f4()
        plot_9g()
    else:
        with np.load(file) as data:
            timestep = float(data["timestep"])
            # amplitude = data["amplitudes"]
            # phase_lag = data["phase_lag"]
            link_data = data["links"][:, 0, :]
            joints_data = data["joints"]
            times = np.arange(0, timestep * np.shape(link_data)[0], timestep)

            # Plot data
            plt.figure("Positions")
            plot_positions(times, link_data)

            plt.figure("Trajectory")
            plot_trajectory(link_data)
            print(joints_data)


if __name__ == '__main__':
    main(plot=not save_plots())
