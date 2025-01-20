import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D

current_directory = os.getcwd()
print(current_directory)

qx = np.linspace(-20, 20, 21)
qy = np.linspace(-20, 20, 21)
#Qx,Qy = np.meshgrid(qx, qy)
t_array = np.linspace(1.5, 51, 34)
#print(t_array)

# Load the data from the file

def load_data(repo):
    with open(repo, 'r') as file:
        data = file.readlines()
    print(len(data))
    DATA = []
    for i in range(0,len(data)):
        a = data[i]
        string_values = a.split()
        float_values = [float(value.replace('E', 'e')) for value in string_values]
        np_array = np.array(float_values)
        DATA.append(np_array)
        data_np = np.array(DATA)
    return data_np

# dir = 'C:/Users/hajdu/OneDrive - VUT/__PHD_data/%_Projects/%_EMLD/%_EMLD_Data/Simulations/%Dyndiff-tilt-dep/241007_FeRh_beam-tilt_back-up/struct-001/01X/1000/'

def load_dyndiff(dir):
    # struct-001/01X/
    # struct-001/11X/

    # struct-011/0-1X/
    # struct-011/10X/

    # struct-111/1-3X/
    # struct-111/-1-3X/

    repo_1 = dir + '1_e1'
    repo_2 = dir + '2_e1'
    repo_3 = dir + '3_e1'
    repo_7 = dir + '7_e1'
    repo_8 = dir + '8_e1'
    repo_9 = dir + '9_e1'

    DATA_1 = load_data(repo_1)
    DATA_2 = load_data(repo_2)
    DATA_3 = load_data(repo_3)
    DATA_7 = load_data(repo_7)
    DATA_8 = load_data(repo_8)
    DATA_9 = load_data(repo_9)

    reshaped_array_1 = DATA_1.reshape(41, 41, 17)
    reshaped_array_2 = DATA_2.reshape(41, 41, 17)
    reshaped_array_3 = DATA_3.reshape(41, 41, 17)
    reshaped_array_7 = DATA_7.reshape(41, 41, 17)
    reshaped_array_8 = DATA_8.reshape(41, 41, 17)
    reshaped_array_9 = DATA_9.reshape(41, 41, 17)

    # EMLD_x = reshaped_array_1 - 0.5*(reshaped_array_2+reshaped_array_3) #)/(reshaped_array_1+reshaped_array_2+reshaped_array_3)
    # EMLD_y = reshaped_array_2 - 0.5*(reshaped_array_1+reshaped_array_3)
    # EMLD_z = reshaped_array_3 - 0.5*(reshaped_array_2+reshaped_array_1)


    reshaped_array_22 = np.rot90(reshaped_array_1, k=1, axes=(0, 1))
    reshaped_array_11 = np.rot90(reshaped_array_2, k=1, axes=(0, 1))
    reshaped_array_33 = np.rot90(reshaped_array_3, k=1, axes=(0, 1))
    reshaped_array_77 = np.rot90(reshaped_array_7, k=1, axes=(0, 1))
    reshaped_array_88 = np.rot90(reshaped_array_8, k=1, axes=(0, 1))
    reshaped_array_99 = np.rot90(reshaped_array_9, k=1, axes=(0, 1))
    # EMLD_xx = reshaped_array_11 - 0.5*(reshaped_array_22 + reshaped_array_33) #)/(reshaped_array_11+reshaped_array_22+reshaped_array_33)


    A = [reshaped_array_11,reshaped_array_22,reshaped_array_33,reshaped_array_77,reshaped_array_88,reshaped_array_99]
    return A