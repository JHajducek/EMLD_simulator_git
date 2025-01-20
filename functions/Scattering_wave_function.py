# Scattering wave function

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from matplotlib.colors import LinearSegmentedColormap

def load_data_wf(repo):
    with open(repo, 'r') as file:
        data = file.readlines()
    print(len(data))
    DATA = np.zeros([640, 640, 4])
    for i in range(0,len(data)):
        a = data[i]
        string_values = a.split()
        if len(string_values) == 6:
            x_coords = int(string_values[0])
            y_coords = int(string_values[1])
            float_values = [float(value.replace('E', 'e')) for value in string_values[2:5]]
            #print(x_coords, y_coords)
            for j in range(0,3):
                DATA[x_coords-1, y_coords-1, j] = float_values[j]

    return DATA

def plot_data_wf():
    repo_1 = 'C:/Users/hajdu/Desktop/EMCD-sim/240407_FeRh_conv_full_test/mult_02mrad_118ZA_300kV_15nm/14.6nm.plt'

    DATA_1 = load_data_wf(repo_1)
    print(np.shape(DATA_1))
    # ================
    a_0 = 5.29177210903*10**(-11) # m (Bohr radius)
    a_FeRh = 2.987  # A
    n = 10          # rep
    Lambda = 1.96809    # pm
    # ================
    r_Max = a_FeRh * n / 2  # A
    k_a = 0.17704 / a_0 # m^-1

    alpha_Max = 2 * np.pi * k_a * Lambda * 10**(-12) * 1000 * n / 2 # mrad

    fig, axs = plt.subplots(2, 3, figsize=(14, 9))

    axs[0,0].imshow(DATA_1[:,:,0], extent=[-r_Max, r_Max, -r_Max, r_Max], cmap = 'viridis')
    axs[0,0].set_title('Re{psi(x,y)}')
    axs[0,0].set_xlabel('x (A)')
    axs[0,0].set_ylabel('y (A)')

    axs[0,1].imshow(DATA_1[:,:,1], extent=[-r_Max, r_Max, -r_Max, r_Max], cmap = 'viridis')
    axs[0,1].set_title('Im{psi(x,y)}')
    axs[0,1].set_xlabel('x (A)')
    axs[0,1].set_ylabel('y (A)')

    abs_real = DATA_1[:,:,0]**2 + DATA_1[:,:,1]**2
    axs[0,2].imshow(abs_real, extent=[-r_Max, r_Max, -r_Max, r_Max], cmap = 'viridis')
    axs[0,2].set_title('abs{psi(x,y)}^2')
    axs[0,2].set_xlabel('x (A)')
    axs[0,2].set_ylabel('y (A)')

    axs[1,0].imshow(DATA_1[:,:,2], extent=[-alpha_Max, alpha_Max, -alpha_Max, alpha_Max], cmap = 'viridis')
    axs[1,0].set_title('Re{psi(k_x,k_y)}')
    axs[1,0].set_xlabel('k_x (mrad)')
    axs[1,0].set_ylabel('k_y (mrad)')

    axs[1,1].imshow(DATA_1[:,:,3], extent=[-alpha_Max, alpha_Max, -alpha_Max, alpha_Max], cmap = 'viridis')
    axs[1,1].set_title('Re{psi(k_x,k_y)}')
    axs[1,1].set_xlabel('k_x (mrad)')
    axs[1,1].set_ylabel('k_y (mrad)')

    abs_recip = DATA_1[:,:,2]**2 + DATA_1[:,:,3]**2

    # Equalize color scale
    vmin = np.min(abs_recip)
    vmax = np.max(abs_recip)

    # Apply gamma correction to the image
    gamma = 5
    gamma_corrected_image = np.power((abs_recip - vmin) / (vmax - vmin), 1/gamma)

    axs[1,2].imshow(gamma_corrected_image, extent=[-alpha_Max, alpha_Max, -alpha_Max, alpha_Max], cmap = 'viridis')
    axs[1,2].set_title('abs{psi(k_x,k_y)}^2')
    axs[1,2].set_xlabel('k_x (mrad)')
    axs[1,2].set_ylabel('k_y (mrad)')

    plt.show()
