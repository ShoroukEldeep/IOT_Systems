# We import the algorithm (You can use from pyade import * to import all of them)
from pyade.newJso import new_jso
import numpy as np
import os
from rnn import train
import json
from collections import namedtuple

def run():
    algorithm = new_jso()
    params = algorithm.get_default_params(dim=9)
    params['bounds'] = np.array([[1,64],
                                [1,3],
                                [5,25],
                                [0,3],
                                [256,512],
                                [128,256],
                                [32,128],
                                [0,2],
                                [0,25]])

    params['opts'] = (0, 0)
    params['func'] = lambda x, y: train(x) - y

    solution, fitness = algorithm.apply(**params)
    print(solution,fitness)
