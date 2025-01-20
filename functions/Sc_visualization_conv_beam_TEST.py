import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from matplotlib.colors import LinearSegmentedColormap

def load_data_Sc(repo):
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

# 240407_FeRh_conv_full_test\dynd_test
def plot_data_Sc():
    repo_1 = 'C:/Users/hajdu/Desktop/EMCD-sim/240407_FeRh_conv_full_test/dynd_test/e1_conv'

    DATA_1 = load_data_Sc(repo_1)
    print(np.shape(DATA_1))

    reshaped_array_1 = DATA_1.reshape(26, 26, 9)
    print(np.shape(reshaped_array_1))

    # Create a figure and subplots
    fig, axs = plt.subplots(3, 3, figsize=(14, 9))
    # cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    # Flatten the axs array to easily iterate over subplots
    axs = axs.flatten()

    qx = np.linspace(100, 150, 21)
    qy = np.linspace(-150, -100, 21)

    colors = [(0, 'blue'), (0.5, 'white'), (1, 'red')]  # (value, color)
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)

    # Plot data on each subplot
    for i, ax in enumerate(axs):
        arr = reshaped_array_1[:,:,int(i)]
        max(max(row) for row in arr)

        # vmin = min(reshaped_array_1[:,:,int(i)])
        # vmax = max(reshaped_array_1[:,:,int(i)])
        vmin = min(min(row) for row in arr)
        vmax = max(max(row) for row in arr)

        if abs(vmin) > vmax:
            vmax_f = abs(vmin)
        else:
            vmax_f = abs(vmax)

        im = ax.imshow(reshaped_array_1[:,:,int(i)], extent=[qx[0], qx[-1], qy[0], qy[-1]], cmap = cmap, vmin = -vmax_f, vmax = vmax_f)#='viridis')
        # if i == 0:  # If first subplot, create colorbar
        cbar = fig.colorbar(im, ax=ax)
        ax.set_title(f'Matrix index: {i+1}')
        ax.set_xlabel('q_x (mrad)')
        ax.set_ylabel('q_y (mrad)')

    plt.subplots_adjust(left=0.05, right=0.9, top=0.9, bottom=0.1, wspace=0.3, hspace=0.3)

    # Adjust layout to prevent overlap
    #plt.tight_layout()

    # Show the plot
    plt.show()




