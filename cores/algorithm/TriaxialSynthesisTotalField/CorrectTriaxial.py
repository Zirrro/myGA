# -*- coding: utf-8 -*-
# @Time    : 2023/6/30 14:03
# @Author  : Salieri
# @FileName: transfer.py
# @Software: PyCharm
# @Comment : correct data with 9 coefficients

import numpy as np
from numpy import cos as cosd
from numpy import sin as sind


def CorrectTriaxial(inputdata, alpha, beta, gamma, k01, k02, k03, offset01, offset02, offset03):
    alpha = alpha * np.pi / 180
    beta = beta * np.pi / 180
    gamma = gamma * np.pi / 180
    inputdata_t = (inputdata - np.array([offset01, offset02, offset03])) / np.array([k01, k02, k03])
    carray = np.array([[cosd(alpha) * cosd(gamma), sind(gamma), sind(alpha) * cosd(gamma)],
                       [0, cosd(beta), sind(beta)],
                       [0, 0, 1]])

    Baxis = np.linalg.inv(carray).dot(inputdata_t.T)
    Btotal = np.sqrt(np.sum(np.square(Baxis), axis=0))
    dataresult = np.hstack((Baxis.T, Btotal.reshape(-1, 1)))
    return dataresult
