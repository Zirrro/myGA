# -*- coding: utf-8 -*-
# @Time    : 2023/6/29 16:00
# @Author  : Salieri
# @FileName: transfer.py
# @Software: PyCharm
# @Comment : Transfer .mat file to ndarray

import scipy.io as sio
import numpy as np

def get_mat_variable_name(mat_file_path):
    mat_info = sio.whosmat(mat_file_path)
    variable_name = str([entry[0] for entry in mat_info])[2:-2]
    return variable_name
def mat_to_nparray(mat_file_path):
    mat_data = sio.loadmat(mat_file_path)
    np_array = np.array(mat_data[get_mat_variable_name(mat_file_path)])
    return np_array

