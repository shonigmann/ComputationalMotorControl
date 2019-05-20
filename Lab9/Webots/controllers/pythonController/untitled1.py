# -*- coding: utf-8 -*-
"""
Created on Mon May 13 18:28:09 2019

@author: Kilian
"""

import numpy as np
import plot_results
import math

from run_simulation import run_simulation
from simulation_parameters import SimulationParameters


#ID_J = {
#    "position": 0,
#    "velocity": 1,
#    "cmd": 2,
#    "torque": 3,
#    "torque_fb": 4,
#    "output": 5
#}


#file = np.load("./logs/9b/simulation_{}.npz".format(simulation_i))
file = np.load("./logs/9b/Energies.npz")
print(np.shape(file['Amplitudes']), '  ', np.shape(file['PhaseLag'])  )   #(2500 iters en 10 seg,       10 joints    , 6 ID_J)
print(file['Amplitudes'][0], '  ', file['PhaseLag'][2]  )
#print(file['joints'][])
# 2500 iteraciones para 10 seg