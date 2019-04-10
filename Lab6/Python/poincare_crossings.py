# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 22:28:47 2019

@author: Simon
"""
def poincare_crossings(res, threshold, crossing_index, figure):
    """ Study poincaré crossings """
    ci = crossing_index

    # Extract state of first trajectory
    state = np.array(res.state[0])

    # Crossing index (index corrsponds to last point before crossing)
    idx = np.argwhere(np.diff(np.sign(state[:, ci] - threshold)) < 0)
    # pylog.debug("Indices:\n{}".format(idx))  # Show crossing indices

    # Linear interpolation to find crossing position on threshold
    # Position before crossing
    pos_pre = np.array([state[index[0], :] for index in idx])
    # Position after crossing
    pos_post = np.array([state[index[0]+1, :] for index in idx])
    # Position on threshold
    pos_treshold = [
        (
            (threshold - pos_pre[i, 1])/(pos_post[i, 1] - pos_pre[i, 1])
        )*(
            pos_post[i, 0] - pos_pre[i, 0]
        ) + pos_pre[i, 0]
        for i, _ in enumerate(idx)
    ]

    # Plot
    # Figure limit cycle variance
    plt.figure(figure)
    plt.plot(pos_treshold, "o-")
    val_min = np.sort(pos_treshold)[2]
    val_max = np.sort(pos_treshold)[-2]
    bnd = 0.3*(val_max - val_min)
    plt.ylim([val_min-bnd, val_max+bnd])
    plt.xlabel(u"Number of Poincaré section crossings")
    plt.ylabel("Value for Neuron 1 (Neuron 2 = {})".format(threshold))
    plt.grid(True)

    # Figure limit cycle
    plt.figure(figure+"_phase")
    plt.plot([val_min-0.3, val_max+0.3], [threshold]*2, "gx--")
    for pos in pos_treshold:
        plt.plot(pos, threshold, "ro")

    # Save plots if option activated
    if DEFAULT["save_figures"] is True:
        from cmcpack.plot import save_figure
        save_figure(figure)
        save_figure(figure+"_phase")

        # Zoom on limit cycle
        plt.figure(figure+"_phase")
        plt.xlim([val_min-bnd, val_max+bnd])
        plt.ylim([threshold-1e-7, threshold+1e-7])
        save_figure(figure=figure+"_phase", name=figure+"_phase_zoom")

    return idx

