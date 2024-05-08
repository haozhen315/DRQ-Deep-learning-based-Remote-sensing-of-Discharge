import numpy as np
import torch.nn as nn


def awei_index(blue, green, nir, swir1, swir2, sh=True):
    '''
    :param blue: blue band
    :param green: green band
    :param nir: near infrared band
    :param swir1: shortwave infrared band 1
    :param swir2: shortwave infrared band 2
    :param sh: whether to use the shadow-resistant version of AWEI
    return: AWEI index
    '''
    if sh:
        awei = blue + 2.5 * green - 1.5 * (nir + swir1) - 0.25 * swir2
    else:
        awei = 4 * (green - swir1) - (0.25 * nir + 2.75 * swir2)
    return awei


def kling_gupta_efficiency(observed, simulated, pp=False):
    '''
    :param observed: observed discharge
    :param simulated: simulated discharge
    :param pp: whether to print the correlation, variability, and bias
    return: Kling-Gupta Efficiency
    '''
    observed = np.array(observed)
    observed = np.float32(observed)
    simulated = np.array(simulated)
    simulated = np.float32(simulated)

    obs_mean = np.mean(observed)
    sim_mean = np.mean(simulated)
    obs_std = np.std(observed)
    sim_std = np.std(simulated)

    correlation = np.corrcoef(observed, simulated)[0, 1]

    variability = sim_std / obs_std

    bias = sim_mean / obs_mean

    kge = 1 - np.sqrt((correlation - 1) ** 2 + (variability - 1) ** 2 + (bias - 1) ** 2)
    if pp:
        print(f'correlation: {correlation}, variability: {variability}, bias: {bias}')
        return kge, correlation, variability, bias

    return kge


def absoluteFilePaths(directory):
    import os
    def absoluteFilePaths(directory):
        for dirpath, _, filenames in os.walk(directory):
            for f in filenames:
                yield os.path.abspath(os.path.join(dirpath, f))

    return list(absoluteFilePaths(directory))
