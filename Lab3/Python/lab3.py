#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Lab 3 """


import numpy as np
import matplotlib.pyplot as plt

import cmc_pylog as pylog
from cmcpack import parse_args, integrate_multiple, DEFAULT


class LeakyIntegratorParameters(object):
    """ Leaky-integrator neuron parameters """

    def __init__(self, tau, D, b, w, exp=np.exp):
        super(LeakyIntegratorParameters, self).__init__()
        self.tau = np.array(tau)  # Time constant
        self.D = np.array(D)
        self.b = np.array(b)
        self.w = np.array(w)  # Weights
        self.exp = exp  # Exponential
        return

    def __str__(self):
        """ String used when printing instantiated object """
        return self.msg()

    def list(self):
        """ Return list of parameters """
        return self.tau, self.D, self.b, self.w, self.exp

    def msg(self):
        """ Parameters information message """
        return (
            "Leaky integrator parameters:"
            "\nTau: {}"
            "\nD:   {}"
            "\nb:   {}"
            "\nw:   {}"
            "\nExp: {}"
        ).format(*self.list())


def two_li_ode(y, t, params):
    """ Derivative function of a network of 2 leaky integrator neurons

    y is the vector of membrane potentials (variable m in lecture equations)
    yd the derivative of the vector of membrane potentials
    """
    # Extract parameters
    tau, D, b, w, exp = params.list()

    # Update the firing rates:
    x = [1/(1+exp(-D*(y[0]+b[0]))), 1/(1+exp(-D*(y[1]+b[1])))]

    # IMPLEMENT THE DIFFERENTIAL EQUATION FOR THE MEMBRANE POTENTIAL
    # Compute the dentritic sums for both neurons
    dend_sum = [w[0][0]*x[0]+w[1][0]*x[1], w[1][1]*x[1]+w[0][1]*x[0]]

    # Compute the membrane potential derivative:
    yd = [1/tau[0]*(-y[0]+dend_sum[0]),1/tau[1]*(-y[1]+dend_sum[1])]
   
    #pylog.debug("x: {}\ndend_sum: {}\nyd: {}".format(x, dend_sum, yd))
    return yd


def two_coupled_li_neurons(y_0, t_max, dt, params, figure="Phase"):
    """ Two mutually coupled leaky-integrator neurons with self connections """
    res = integrate_multiple(
        two_li_ode,
        y_0,
        np.arange(0, t_max, dt),
        args=(params,)
    )
    labels = ["Neuron 1", "Neuron 2"]
    res.plot_state(figure+"_state", label=False, subs_labels=labels)
    res.plot_phase(figure+"_phase", scale=0.05, label=labels)
    return res


def exercise5():
    """ Lab 3 - Exrecise 5 """
    # Fixed parameters of the neural network
    tau = [0.05, 0.05] 
    D = 1
    # Additional parameters
    b = [0, 0]
    w = [[1, 0], [0, 1]]
    # All system parameters packed in object for integration
    params = LeakyIntegratorParameters(tau, D, b, w)
    # Initial conditions
    y_0 = [[0, 0]]  # Values of the membrane potentials of the two neurons
    dt = 1e-4
    t_max = 30  # Set total simulation time

    # Integration (make sure to implement)
    two_coupled_li_neurons(y_0, t_max, dt, params, "Case1")

    # Two stable fixed points and one saddle node
    pylog.warning("Implement two stable fixed points and one saddle node")
    tau = [1, 1]
    D = 1
    b = [-3.4, -2.5]
    w = [[5.25, -1], [1, 5.25]]
    # All system parameters packed in object for integration
    params = LeakyIntegratorParameters(tau, D, b, w)
    y_0 = [[0, 0]]  # Values of the membrane potentials of the two neurons
    two_coupled_li_neurons(y_0, t_max, dt, params, "Case2")

    # Limit cycle
    """
    pylog.warning("Implement limit cycle")
    """
    tau = [0.1, 0.1]
    D = 1
    b = [-2.75, -1.75]
    w = [[4.5, -1], [1, 4.5]]
    # All system parameters packed in object for integration
    params = LeakyIntegratorParameters(tau, D, b, w)
    y_0 = [[0, 0]]  # Values of the membrane potentials of the two neurons
    two_coupled_li_neurons(y_0, t_max, dt, params, "Case3")
    
    pylog.warning(u"Implement Poincare analysis of limit cycle")

    # Limit cycle (small), one stable fixed point and one saddle node
    pylog.warning(
        "Implement a system with:"
        "\n- One limit cycle (small)"
        "\n- One stable fixed point"
        "\n- One saddle node"
    )
    
    pylog.warning(u"Implement Poincare analysis of limit cycle")

    if DEFAULT["save_figures"] is False:
        plt.show()

    return


def main():
    """ Lab 3 exercise """
    pylog.info("Runnig exercise 5")
    exercise5()
    return


if __name__ == "__main__":
    parse_args()
    main()

