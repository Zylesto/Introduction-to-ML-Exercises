'''
Created on 04.10.2016

@author: Daniel Stromer
@modified by Charly, Christian, Max (23.12.2020)
'''
import numpy as np
import matplotlib.pyplot as plt
# do not import more modules!


def calculate_R_Distance(Rx, Ry):
    '''
    calculate similarities of Ring features
    :param Rx: Ring features of Person X
    :param Ry: Ring features of Person Y
    :return: Similiarity index of the two feature vectors
    '''
    return np.sum(np.abs(Rx - Ry)) / len(Rx)


def calculate_Theta_Distance(Thetax, Thetay):
    '''
    calculate similarities of Fan features
    :param Thetax: Fan features of Person X
    :param Thetay: Fan features of Person Y
    :return: Similiarity index of the two feature vectors
    '''
    Lxx = np.sum((Thetax - np.sum(Thetax) / len(Thetax)) ** 2)
    Lyy = np.sum((Thetay - np.sum(Thetay) / len(Thetay)) ** 2)
    Lxy = np.sum((Thetax - np.sum(Thetax) / len(Thetax)) * (Thetay - np.sum(Thetay) / len(Thetay)))

    return (1 - (Lxy * Lxy) / (Lxx * Lyy)) * 100
