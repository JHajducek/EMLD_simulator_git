from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from functions.DynDiff_load import load_data, load_dyndiff
from functions.Matrix_treatment import rotation_matrix_x, rotation_matrix_y, rotation_matrix_z, orientation_to_rotation_matrix, ROTATE_EULER
from functions.Matrix_treatment import rotate_vector_map, reverse_rotate_vector_map
from functions.matrices_JR import L2_down_coeffs, L3_down_coeffs, L2_up_coeffs, L3_up_coeffs, A_trans
from functions.Sc_visualization_conv_beam_TEST import load_data_Sc, plot_data_Sc
from functions.Scattering_wave_function import load_data_wf, plot_data_wf
from functions.Plotting_01X_11X import plot_tilt_thc_dep
from functions.Sphere_coverage_Euler_angles import sphere_plot

# Comment in the new branch
# Few more tests

Atrans = A_trans()

# print(np.conj(Atrans))
# print(L3_up_coeffs(0, 0, 0))
# print(L3_down_coeffs(0, 0, 0))
# print(L2_up_coeffs(0, 0, 0))
# print(L2_down_coeffs(0, 0, 0))


dir = 'C:/Users/hajdu/OneDrive - VUT/__PHD_data/%_Projects/%_EMLD/%_EMLD_Data/Simulations/%Dyndiff-tilt-dep/241007_FeRh_beam-tilt_back-up/struct-001/01X/1000/'
A = load_dyndiff(dir)
reshaped_array_1, reshaped_array_2, reshaped_array_3, reshaped_array_7, reshaped_array_8, reshaped_array_9 = A
print(len(A))


# L3_up_A_11 = []
# L3_up_A_22 = []
# L3_up_A_33 = []
# L3_up_B_11 = []
# L3_up_B_22 = []
# L3_up_B_33 = []
# # beta_M = np.linspace(0, 180, 360)
# # for i in range(0, len(beta_M)):
# #     beta_s = beta_M[i]
# #     L3_up_A = L3_up_coeffs(0, beta_s, 0)[0]
# #     L3_up_B = L3_up_coeffs(0, beta_s, 0)[1]
# alpha_M = np.linspace(0, 180, 360)
# for i in range(0, len(alpha_M)):
#     alpha_s = alpha_M[i]
#     L3_up_A = L3_up_coeffs(alpha_s, 90, 0)[0]
#     L3_up_B = L3_up_coeffs(alpha_s, 90, 0)[1]
#     print(L3_up_B)
#     print(Atrans)
#     print(Atrans.T)
#     print(np.conj(Atrans).T)
#     AA = np.conj(Atrans).T @ L3_up_A @ Atrans
#     BB = np.conj(Atrans).T @ L3_up_B @ Atrans
#     print(np.real(BB[0,0]), np.real(BB[1,1]), np.real(BB[2,2]))
#     print(np.real(AA[0,0]), np.real(AA[1,1]), np.real(AA[2,2]))
#     L3_up_A_11.append(np.real(AA[0,0]))
#     L3_up_A_22.append(np.real(AA[1,1]))
#     L3_up_A_33.append(np.real(AA[2,2]))
#     L3_up_B_11.append(np.real(BB[0,0]))
#     L3_up_B_22.append(np.real(BB[1,1]))
#     L3_up_B_33.append(np.real(BB[2,2]))

# plt.plot(alpha_M, L3_up_A_11, label = 'L3_up_A_11')
# plt.plot(alpha_M, L3_up_A_22, label = 'L3_up_A_22')
# plt.plot(alpha_M, L3_up_A_33, label = 'L3_up_A_33')
# plt.plot(alpha_M, L3_up_B_11, label = 'L3_up_B_11')
# plt.plot(alpha_M, L3_up_B_22, label = 'L3_up_B_22')
# plt.plot(alpha_M, L3_up_B_33, label = 'L3_up_B_33')

# # plt.plot(beta_M, L3_up_A_11, label = 'L3_up_A_11')
# # plt.plot(beta_M, L3_up_A_22, label = 'L3_up_A_22')
# # plt.plot(beta_M, L3_up_A_33, label = 'L3_up_A_33')
# # plt.plot(beta_M, L3_up_B_11, label = 'L3_up_B_11')
# # plt.plot(beta_M, L3_up_B_22, label = 'L3_up_B_22')
# # plt.plot(beta_M, L3_up_B_33, label = 'L3_up_B_33')
# plt.legend()
# plt.show()

qx = np.linspace(-20, 20, 41)
qy = np.linspace(-20, 20, 41)
Qx,Qy = np.meshgrid(qx, qy)

def electron_wavelength(kinetic_energy_eV):
    # Constants
    h = 6.626e-34  # Planck's constant (J·s)
    c = 3.0e8  # Speed of light (m/s)
    eV_to_J = 1.602e-19  # Conversion factor from eV to Joules
    E0_eV = 511000  # Rest energy of electron in eV

    # Total energy and momentum in eV
    total_energy_eV = kinetic_energy_eV + E0_eV
    momentum_eV = np.sqrt(total_energy_eV**2 - E0_eV**2)

    # Convert momentum from eV/c to SI units (kg·m/s)
    momentum_SI = momentum_eV * eV_to_J / c

    # Wavelength (meters)
    wavelength = h / momentum_SI
    return wavelength

# Example: Kinetic energy of 1 MeV (1,000,000 eV)
kinetic_energy_eV = 300000
wavelength = electron_wavelength(kinetic_energy_eV)

print(f"Kinetic Energy: {kinetic_energy_eV} eV")
print(f"Relativistic Wavelength: {wavelength:.3e} meters")


E0 = 300000
dE = 708
Q_IP = np.sqrt(Qx**2 + Qy**2)
Q_IP_recipm = Q_IP/1000/electron_wavelength(E0)
k0 = 2*np.pi/electron_wavelength(E0)
k = 2*np.pi/electron_wavelength(E0-dE)
Qz_recipm = k0 - np.sqrt(k**2 - Q_IP_recipm**2)
Qz = Qz_recipm*electron_wavelength(E0)*1000

Qx_norm = Qx/np.sqrt(Qx**2 + Qy**2 + Qz**2)
Qy_norm = Qy/np.sqrt(Qx**2 + Qy**2 + Qz**2)
Qz_norm = Qz/np.sqrt(Qx**2 + Qy**2 + Qz**2)

q_par = np.array([Qx_norm[20,20],Qy_norm[20,20],Qz_norm[20,20]])
q_perp = np.array([Qx_norm[20,0],Qy_norm[20,0],Qz_norm[20,0]])

# ======== SCATTERING MAP PLOTTING ============

# alpha, beta, gamma = 45, 90, 0

def scattering_map(alpha, beta, gamma):

    L3_up_A = L3_up_coeffs(alpha, beta, gamma)[0]
    L3_up_B = L3_up_coeffs(alpha, beta, gamma)[1]

    MM_L3_up_A = np.conj(Atrans).T @ L3_up_A @ Atrans
    MM_L3_up_B = np.conj(Atrans).T @ L3_up_B @ Atrans
    # print(np.real(BB[0,0]), np.real(BB[1,1]), np.real(BB[2,2]))

    x_map = Qx_norm
    y_map = Qy_norm
    z_map = Qz_norm

    x_rot, y_rot, z_rot = rotate_vector_map(x_map, y_map, z_map, alpha, beta, gamma)

    # print(x_rot)
    # print(y_rot)
    # print(z_rot)

    x_rot_2, y_rot_2, z_rot_2 = reverse_rotate_vector_map(x_rot*np.sqrt(MM_L3_up_A[0,0]), y_rot*np.sqrt(MM_L3_up_A[1,1]), z_rot*np.sqrt(MM_L3_up_A[2,2]), alpha, beta, gamma)
    print(MM_L3_up_A[0,0], MM_L3_up_A[0,0], MM_L3_up_A[0,0])
    # 001 case (as a reference)

    MM_L3_up_A_001 = np.array([
        [68.75, 0, 0],
        [0, 68.75, 0],
        [0, 0, 91.67]])
    
    # NN_11 = np.rot90(reshaped_array_1[:,:,10])
    # NN_22 = np.rot90(reshaped_array_2[:,:,10])
    # NN_33 = np.rot90(reshaped_array_3[:,:,10])

    NN_11 = reshaped_array_1[:,:,10]
    NN_22 = reshaped_array_2[:,:,10]
    NN_33 = reshaped_array_3[:,:,10]

    x_rot_2_001, y_rot_2_001, z_rot_2_001 = x_map*np.sqrt(MM_L3_up_A_001[0,0]), y_map*np.sqrt(MM_L3_up_A_001[1,1]), z_map*np.sqrt(MM_L3_up_A_001[2,2])
    x_sc_001, y_sc_001, z_sc_001 = x_rot_2_001**2 *NN_11/100, y_rot_2_001**2 *NN_22/100, z_rot_2_001**2 *NN_33/100
    I_tot_001 = x_sc_001 + y_sc_001 + z_sc_001

    I_tot = x_rot_2**2 *NN_11/100 + y_rot_2**2 *NN_22/100 + z_rot_2**2 *NN_33/100
    return I_tot, I_tot_001, x_rot, x_rot_2

# Function to update the 3D plot based on Euler angles and the selected orientation
def update_arrow():
    # Get Euler angles for both arrows
    # roll_red = float(entry_roll_red.get())
    # pitch_red = float(entry_pitch_red.get())
    # yaw_red = float(entry_yaw_red.get())
    # roll_blue = float(entry_roll_blue.get())
    # pitch_blue = float(entry_pitch_blue.get())
    # yaw_blue = float(entry_yaw_blue.get())

    roll_red = 0
    pitch_red = 0
    yaw_red = 0
    roll_blue = 0
    pitch_blue = 0
    yaw_blue = 0

    # Get Euler angles for the secondary coordinate system
    alpha = float(entry_alpha.get())
    beta = float(entry_beta.get())
    gamma = float(entry_gamma.get())

    # Get selected orientation
    orientation = orientation_var.get()

    # Get the rotation matrix for the selected orientation
    R_orientation = orientation_to_rotation_matrix(orientation)

    # Compute the rotation matrix for the red arrow
    R_red = ROTATE_EULER(roll_red, pitch_red, yaw_red)
    # Compute the rotation matrix for the blue arrow
    R_blue = ROTATE_EULER(roll_blue, pitch_blue, yaw_blue)
    # Compute the rotation matrix for the secondary coordinate system
    R_secondary = ROTATE_EULER(alpha, beta, gamma)

    # Update the 3D plot
    ax3d.cla()  # Clear the current axes
    ax3d.set_xlim([-0.5, 0.5])
    ax3d.set_ylim([-0.5, 0.5])
    ax3d.set_zlim([-0.5, 0.5])

    # Plot coordinate system (sketch with arrows and labels)
    ax3d.quiver(0, 0, 0, 0.5, 0, 0, color='black', linewidth=1, label='X')
    ax3d.quiver(0, 0, 0, 0, 0.5, 0, color='black', linewidth=1, label='Y')
    ax3d.quiver(0, 0, 0, 0, 0, 0.5, color='black', linewidth=1, label='Z')

    ax3d.text(0.5, 0, 0, 'X', fontsize=12, color='black')
    ax3d.text(0, 0.5, 0, 'Y', fontsize=12, color='black')
    ax3d.text(0, 0, 0.5, 'Z', fontsize=12, color='black')

    # Plot the secondary coordinate system (orange)
    # ax3d.quiver(0, 0, 0, 0.5, 0, 0, color='orange', linewidth=1, label='X\'')
    # ax3d.quiver(0, 0, 0, 0, 0.5, 0, color='orange', linewidth=1, label='Y\'')
    # ax3d.quiver(0, 0, 0, 0, 0, 0.5, color='orange', linewidth=1, label='Z\'')

    ax3d.text(0.5, 0, 0, 'X\'', fontsize=12, color='orange')
    ax3d.text(0, 0.5, 0, 'Y\'', fontsize=12, color='orange')
    ax3d.text(0, 0, 0.5, 'Z\'', fontsize=12, color='orange')

    # # Apply the secondary rotation to the coordinate system (rotate the grey coordinate axes)
    # ax3d.quiver(0, 0, 0, 0.5, 0, 0, color='orange', linewidth=1)
    # ax3d.quiver(0, 0, 0, 0, 0.5, 0, color='orange', linewidth=1)
    # ax3d.quiver(0, 0, 0, 0, 0, 0.5, color='orange', linewidth=1)


    # Plot cube vertices
    vertices = np.array([[0-0.25, 0-0.25, 0-0.25], [0.5-0.25, 0-0.25, 0-0.25], [0.5-0.25, 0.5-0.25, 0-0.25], [0-0.25, 0.5-0.25, 0-0.25],
                         [0-0.25, 0-0.25, 0.5-0.25], [0.5-0.25, 0-0.25, 0.5-0.25], [0.5-0.25, 0.5-0.25, 0.5-0.25], [0-0.25, 0.5-0.25, 0.5-0.25]])
    
    # Apply orientation rotation to the cube vertices
    rotated_vertices = np.dot(vertices, R_orientation.T)

    # Plot the cube edges after rotation
    _ = [ax3d.plot([rotated_vertices[0][0], rotated_vertices[1][0]], [rotated_vertices[0][1], rotated_vertices[1][1]], [rotated_vertices[0][2], rotated_vertices[1][2]], color="black"),
         ax3d.plot([rotated_vertices[1][0], rotated_vertices[2][0]], [rotated_vertices[1][1], rotated_vertices[2][1]], [rotated_vertices[1][2], rotated_vertices[2][2]], color="black"),
         ax3d.plot([rotated_vertices[2][0], rotated_vertices[3][0]], [rotated_vertices[2][1], rotated_vertices[3][1]], [rotated_vertices[2][2], rotated_vertices[3][2]], color="black"),
         ax3d.plot([rotated_vertices[3][0], rotated_vertices[0][0]], [rotated_vertices[3][1], rotated_vertices[0][1]], [rotated_vertices[3][2], rotated_vertices[0][2]], color="black"),
         ax3d.plot([rotated_vertices[4][0], rotated_vertices[5][0]], [rotated_vertices[4][1], rotated_vertices[5][1]], [rotated_vertices[4][2], rotated_vertices[5][2]], color="black"),
         ax3d.plot([rotated_vertices[5][0], rotated_vertices[6][0]], [rotated_vertices[5][1], rotated_vertices[6][1]], [rotated_vertices[5][2], rotated_vertices[6][2]], color="black"),
         ax3d.plot([rotated_vertices[6][0], rotated_vertices[7][0]], [rotated_vertices[6][1], rotated_vertices[7][1]], [rotated_vertices[6][2], rotated_vertices[7][2]], color="black"),
         ax3d.plot([rotated_vertices[7][0], rotated_vertices[4][0]], [rotated_vertices[7][1], rotated_vertices[4][1]], [rotated_vertices[7][2], rotated_vertices[4][2]], color="black"),
         ax3d.plot([rotated_vertices[0][0], rotated_vertices[4][0]], [rotated_vertices[0][1], rotated_vertices[4][1]], [rotated_vertices[0][2], rotated_vertices[4][2]], color="black"),
         ax3d.plot([rotated_vertices[1][0], rotated_vertices[5][0]], [rotated_vertices[1][1], rotated_vertices[5][1]], [rotated_vertices[1][2], rotated_vertices[5][2]], color="black"),
         ax3d.plot([rotated_vertices[2][0], rotated_vertices[6][0]], [rotated_vertices[2][1], rotated_vertices[6][1]], [rotated_vertices[2][2], rotated_vertices[6][2]], color="black"),
         ax3d.plot([rotated_vertices[3][0], rotated_vertices[7][0]], [rotated_vertices[3][1], rotated_vertices[7][1]], [rotated_vertices[3][2], rotated_vertices[7][2]], color="black")]

    # Apply the orientation to the red and blue arrows
    arrow_red = np.array([1, 0, 0])
    arrow_red_rot = np.dot(R_orientation, np.dot(R_red, arrow_red))
    arrow_blue = np.array([0, 1, 0])
    arrow_blue_rot = np.dot(R_orientation, np.dot(R_blue, arrow_blue))

    # Plot the red and blue arrows after rotation
    ax3d.quiver(0, 0, 0, arrow_red_rot[0], arrow_red_rot[1], arrow_red_rot[2], color='red', length=0.5)
    ax3d.quiver(0, 0, 0, arrow_blue_rot[0], arrow_blue_rot[1], arrow_blue_rot[2], color='blue', length=0.5)

    # Plot the secondary (orange) arrows for X', Y', Z'
    arrow_secondary_x = np.array([1, 0, 0])
    arrow_secondary_y = np.array([0, 1, 0])
    arrow_secondary_z = np.array([0, 0, 1])
    
    # arrow_secondary_x_rot = np.dot(R_secondary, arrow_secondary_x)
    # arrow_secondary_y_rot = np.dot(R_secondary, arrow_secondary_y)
    # arrow_secondary_z_rot = np.dot(R_secondary, arrow_secondary_z)
    # R = np.dot(rotation_matrix_z, np.dot(R_y_beta, rotation_matrix_z))
    # arrow_secondary_x_rot = np.dot(R_orientation, np.dot(rotation_matrix_z(alpha),np.dot(rotation_matrix_y(beta), np.dot(rotation_matrix_z(gamma), arrow_secondary_x))))
    # arrow_secondary_y_rot = np.dot(R_orientation, np.dot(rotation_matrix_z(alpha),np.dot(rotation_matrix_y(beta), np.dot(rotation_matrix_z(gamma), arrow_secondary_y))))
    # arrow_secondary_z_rot = np.dot(R_orientation, np.dot(rotation_matrix_z(alpha),np.dot(rotation_matrix_y(beta), np.dot(rotation_matrix_z(gamma), arrow_secondary_z))))

    R = ROTATE_EULER(alpha, beta, gamma)
    arrow_secondary_x_rot = R @ arrow_secondary_x
    arrow_secondary_y_rot = R @ arrow_secondary_y
    arrow_secondary_z_rot = R @ arrow_secondary_z


    # Plot the secondary coordinate system
    ax3d.quiver(0, 0, 0, arrow_secondary_x_rot[0], arrow_secondary_x_rot[1], arrow_secondary_x_rot[2], color='black', length=0.5, linewidths=3)
    ax3d.quiver(0, 0, 0, arrow_secondary_y_rot[0], arrow_secondary_y_rot[1], arrow_secondary_y_rot[2], color='orange', length=0.5)
    ax3d.quiver(0, 0, 0, arrow_secondary_z_rot[0], arrow_secondary_z_rot[1], arrow_secondary_z_rot[2], color='green', length=0.5, linewidths=5)


    semiangle_deg = 5  # Semiangle of the cone in degrees
    semiangle_rad = np.radians(semiangle_deg)  # Convert to radians
    height = 0.7  # Height of the cone
    num_points = 100  # Nu  mber of points along the height and circumference

    # Generate the cone's coordinates
    z = np.linspace(0, height, num_points)  # Height from 0 to 'height'
    theta = np.linspace(0, 2 * np.pi, num_points)  # Angle around the z-axis
    Z, Theta = np.meshgrid(z, theta)  # Create a grid of height and angle

    # Calculate the radius of the cone at each height
    radius = np.tan(semiangle_rad) * Z  # Radius increases with height based on the semiangle

    # Parametric equations for the cone surface
    X = radius * np.cos(Theta)  # X coordinates
    Y = radius * np.sin(Theta)  # Y coordinates
    ax3d.plot_surface(X, Y, Z, rstride=1, cstride=1, color='b', alpha=0.1)
    ax3d.plot_surface(-X, -Y, -Z, rstride=1, cstride=1, color='b', alpha=0.1)

    alpha_s = np.linspace(0, 2 * np.pi, 100)  # Azimuthal angle (0 to 2pi)
    beta_s = np.linspace(0, np.pi, 100)      # Polar angle (0 to pi)
    alpha_grid, beta_grid = np.meshgrid(alpha_s, beta_s)

    matrix_00_component = np.zeros_like(alpha_grid)

# Iterate over the grid of alpha and beta to evaluate the function
    # for i in range(len(alpha_s)):
    #     for j in range(len(beta_s)):
    #         alpha = alpha_grid[i, j]
    #         beta = beta_grid[i, j]
            
    #         # Get the (0,0) component from L3_up_coeffs
    #         L3_up_A  = L3_up_coeffs(alpha, beta, gamma)[0]
    #         L3_up_B  = L3_up_coeffs(alpha, beta, gamma)[1]
    #         AA = np.conj(Atrans).T @ L3_up_A @ Atrans
    #         BB = np.conj(Atrans).T @ L3_up_B @ Atrans
    #         matrix_00_component[i, j] = np.real(AA[0,0])  # Access the (0,0) component of the first matrix


    # L3_up_A = L3_up_coeffs(alpha_grid, beta_grid, gamma)[0]
    # L3_up_B = L3_up_coeffs(alpha_grid, beta_grid, gamma)[1]
    # print(L3_up_B)
    # print(Atrans)
    # print(Atrans.T)
    # print(np.conj(Atrans).T)
    # AA = np.conj(Atrans).T @ L3_up_A @ Atrans
    # BB = np.conj(Atrans).T @ L3_up_B @ Atrans

    # matrix_00_component = AA[0, 0]
    # print(np.real(BB[0,0]), np.real(BB[1,1]), np.real(BB[2,2]))
    # print(np.real(AA[0,0]), np.real(AA[1,1]), np.real(AA[2,2]))
    # L3_up_A_11_a.append(np.real(AA[0,0]))

    # Define the function f(alpha, beta)
    # f = (np.sin(2 * alpha_grid) * np.cos(2 * beta_grid)+2)/8

    # # Convert to Cartesian coordinates for 3D plotting
    r = matrix_00_component/300  # r is the value of the function
    x = r * np.sin(beta_grid) * np.cos(alpha_grid)
    y = r * np.sin(beta_grid) * np.sin(alpha_grid)
    z = r * np.cos(beta_grid)

    # Plot the 3D surface
    # fig = plt.figure(figsize=(8, 6))
    # contour = ax3d.contourf(alpha_grid, beta_grid, matrix_00_component/100, cmap='viridis',alpha=0.3)
    # surf = ax3d.plot_surface(x, y, z, facecolors=plt.cm.viridis((f - f.min()) / (f.max() - f.min())),edgecolor='none', alpha=0.3)

    # Surface plot
    # surf = ax3d.plot_surface(x, y, z, facecolors=plt.cm.viridis((f - f.min()) / (f.max() - f.min())),edgecolor='none',
                            # alpha=0.3)

    # Update the canvas for 3D plot
    canvas_3d.draw()

# Function to update the 2D plot with the new sin(x + yaw_red) * sin(y + yaw_blue)
def update_2d_plot():

    alpha = float(entry_alpha.get())
    beta = float(entry_beta.get())
    gamma = float(entry_gamma.get())

    I_tot_var, I_tot_001_var, x_rot_var, x_rot_2_var = scattering_map(alpha, beta, gamma)
    SC_MAP_var = I_tot_var - I_tot_001_var

    L3_up_A_11_a = []
    L3_up_A_22_a = []
    L3_up_A_33_a = []
    L3_up_B_11_a = []
    L3_up_B_22_a = []
    L3_up_B_33_a = []
    # beta_M = np.linspace(0, 180, 360)
    # for i in range(0, len(beta_M)):
    #     beta_s = beta_M[i]
    #     L3_up_A = L3_up_coeffs(0, beta_s, 0)[0]
    #     L3_up_B = L3_up_coeffs(0, beta_s, 0)[1]

    alpha_M = np.linspace(-180, 180, 360)
    for i in range(0, len(alpha_M)):
        alpha_s = alpha_M[i]
        L3_up_A = L3_up_coeffs(alpha_s, beta, gamma)[0]
        L3_up_B = L3_up_coeffs(alpha_s, beta, gamma)[1]
        # print(L3_up_B)
        # print(Atrans)
        # print(Atrans.T)
        # print(np.conj(Atrans).T)
        AA = np.conj(Atrans).T @ L3_up_A @ Atrans
        BB = np.conj(Atrans).T @ L3_up_B @ Atrans
        # print(np.real(BB[0,0]), np.real(BB[1,1]), np.real(BB[2,2]))
        # print(np.real(AA[0,0]), np.real(AA[1,1]), np.real(AA[2,2]))
        L3_up_A_11_a.append(np.real(AA[0,0]))
        L3_up_A_22_a.append(np.real(AA[1,1]))
        L3_up_A_33_a.append(np.real(AA[2,2]))
        L3_up_B_11_a.append(np.real(BB[0,0]))
        L3_up_B_22_a.append(np.real(BB[1,1]))
        L3_up_B_33_a.append(np.real(BB[2,2]))
    
    L3_up_A_11_b = []
    L3_up_A_22_b = []
    L3_up_A_33_b = []
    L3_up_B_11_b = []
    L3_up_B_22_b = []
    L3_up_B_33_b = []

    beta_M = np.linspace(-180, 180, 360)
    for i in range(0, len(beta_M)):
        beta_s = beta_M[i]
        L3_up_A = L3_up_coeffs(alpha, beta_s, gamma)[0]
        L3_up_B = L3_up_coeffs(alpha, beta_s, gamma)[1]
        # print(L3_up_B)
        # print(Atrans)
        # print(Atrans.T)
        # print(np.conj(Atrans).T)
        AA = np.conj(Atrans).T @ L3_up_A @ Atrans
        BB = np.conj(Atrans).T @ L3_up_B @ Atrans
        # print(np.real(BB[0,0]), np.real(BB[1,1]), np.real(BB[2,2]))
        # print(np.real(AA[0,0]), np.real(AA[1,1]), np.real(AA[2,2]))
        L3_up_A_11_b.append(np.real(AA[0,0]))
        L3_up_A_22_b.append(np.real(AA[1,1]))
        L3_up_A_33_b.append(np.real(AA[2,2]))
        L3_up_B_11_b.append(np.real(BB[0,0]))
        L3_up_B_22_b.append(np.real(BB[1,1]))
        L3_up_B_33_b.append(np.real(BB[2,2]))


    # I_tot, I_tot_001, x_rot, x_rot_2 = scattering_map(45, 90, 0)
    # SC_MAP = I_tot - I_tot_001
    

    colors = [(0, 'blue'), (0.5, 'white'), (1, 'red')]  # (value, color)
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
    vmax_f = 1
    
    # Update the imshow map
    axs[0, 0].axis('off')

    # Update the imshow map
    polar_ax.cla()  # Clear previous imshow plot
    polar_ax.plot(np.radians(alpha_M), L3_up_A_11_a, label='L3_up_A_11', color='blue')
    polar_ax.plot(np.radians(alpha_M), L3_up_A_22_a, label='L3_up_A_22', color='green')
    polar_ax.plot(np.radians(alpha_M), L3_up_A_33_a, label='L3_up_A_33', color='red')
    polar_ax.plot(np.radians(alpha_M), L3_up_B_11_a, label='L3_up_B_11', color='black')
    polar_ax.plot(np.radians(alpha_M), L3_up_B_22_a, label='L3_up_B_22', color='cyan')
    polar_ax.plot(np.radians(alpha_M), L3_up_B_33_a, label='L3_up_B_33', color='orange')
    polar_ax.legend(loc='upper left', bbox_to_anchor=(-0.1, 1.1), bbox_transform=polar_ax.transAxes, fontsize=6)
    polar_ax.set_title("alpha-dep coeffs")

# Add a green arrow along the long axis

    arrow_secondary_z = np.array([0, 0, 1])
    R = ROTATE_EULER(alpha, beta, gamma)
    arrow_secondary_z_rot = R @ arrow_secondary_z


    max_angle_IP = np.arctan2(arrow_secondary_z_rot[1],arrow_secondary_z_rot[0])
    max_radius_IP = 50*np.sqrt((arrow_secondary_z_rot[0]**2 +arrow_secondary_z_rot[1]**2)/(arrow_secondary_z_rot[0]**2 +arrow_secondary_z_rot[1]**2 + arrow_secondary_z_rot[2]**2))
    radii = np.linspace(0, 100, 100)
    polar_ax.plot([max_angle_IP] * len(radii), radii, linestyle=':', color='black')
    polar_ax.plot([max_angle_IP + np.pi] * len(radii), radii, linestyle=':', color='black')
    polar_ax.annotate('', 
                xy=(max_angle_IP, max_radius_IP), 
                xytext=(max_angle_IP - np.pi, max_radius_IP), 
                arrowprops=dict(facecolor='green', shrink=0, width=2, headwidth=10))
    polar_ax.grid(True)

    axs[0, 1].axis('off')

    # Update the imshow map
    polar_ax_2.cla()  # Clear previous imshow plot
    polar_ax_2.plot(np.radians(beta_M), L3_up_A_11_b, label='L3_up_A_11', color='blue')
    polar_ax_2.plot(np.radians(beta_M), L3_up_A_22_b, label='L3_up_A_22', color='green')
    polar_ax_2.plot(np.radians(beta_M), L3_up_A_33_b, label='L3_up_A_33', color='red')
    polar_ax_2.plot(np.radians(beta_M), L3_up_B_11_b, label='L3_up_B_11', color='black')
    polar_ax_2.plot(np.radians(beta_M), L3_up_B_22_b, label='L3_up_B_22', color='cyan')
    polar_ax_2.plot(np.radians(beta_M), L3_up_B_33_b, label='L3_up_B_33', color='orange')
    polar_ax_2.legend(loc='upper left', bbox_to_anchor=(-0.1, 1.1), bbox_transform=polar_ax.transAxes, fontsize=6)
    polar_ax_2.set_theta_offset(np.pi / 2)  # Rotate by 90 degrees
    polar_ax_2.set_theta_direction(-1)  # Make it counterclockwise
    polar_ax_2.set_title("beta-dep coeffs")

    # Add a green arrow along the long axis

    arrow_secondary_z = np.array([0, 0, 1])
    R = ROTATE_EULER(alpha, beta, gamma)
    arrow_secondary_z_rot = R @ arrow_secondary_z

    # max_angle_OOP = np.arctan2(np.sqrt(arrow_secondary_z_rot[0]**2 + arrow_secondary_z_rot[1]**2), arrow_secondary_z_rot[2])
    max_angle_OOP = np.radians(beta)
    max_radius_OOP = 50*np.sqrt((arrow_secondary_z_rot[2]**2)/(arrow_secondary_z_rot[0]**2 +arrow_secondary_z_rot[1]**2 + arrow_secondary_z_rot[2]**2))
    radii = np.linspace(0, 100, 100)
    polar_ax_2.plot([max_angle_OOP] * len(radii), radii, linestyle=':', color='black')
    polar_ax_2.plot([max_angle_OOP+ np.pi] * len(radii), radii, linestyle=':', color='black')
    polar_ax_2.annotate('', 
                xy=(max_angle_OOP, max_radius_OOP), 
                xytext=(max_angle_OOP - np.pi, max_radius_OOP), 
                arrowprops=dict(facecolor='green', shrink=0, width=2, headwidth=10))
    polar_ax_2.grid(True)


    # c = axs[0, 1].imshow(np.real(x_rot_2_var), extent=[qx[0], qx[-1], qy[-1], qy[0]], cmap = cmap, vmin = -vmax_f, vmax = vmax_f)
    # axs[0, 1].cla()  # Clear previous imshow plot
    # c = axs[0, 1].imshow(np.real(x_rot_var), extent=[qx[0], qx[-1], qy[-1], qy[0]], cmap = cmap, vmin = -vmax_f, vmax = vmax_f)
    # axs[0, 1].set_title('$\delta [dA/dE]$, $L_{2}$', fontsize=12)
    # axs[0, 1].set_title('$\delta [dB/dE]$, $L_{2}$', fontsize=12)
    
    # # Update the colorbar only if it's not already there
    # if not hasattr(update_2d_plot, 'colorbar_added') or not update_2d_plot.colorbar_added:
    #     fig2d.colorbar(c, ax=axs[0, 1])
    #     update_2d_plot.colorbar_added = True

    # Update the imshow map
    axs[1, 0].cla()  # Clear previous imshow plot
    vmax_f = 100






    c = axs[1, 0].imshow(np.real(I_tot_001_var), extent=[qx[0], qx[-1], qy[-1], qy[0]], cmap = cmap, vmin = -vmax_f, vmax = vmax_f)
    axs[1, 0].set_title('$\delta [dA/dE]$, $L_{3}$', fontsize=12)
    
    # # Update the colorbar only if it's not already there
    # if not hasattr(update_2d_plot, 'colorbar_added') or not update_2d_plot.colorbar_added:
    #     fig2d.colorbar(c, ax=axs[1, 0])
    #     update_2d_plot.colorbar_added = True

    # Update the imshow map
    axs[1, 1].cla()  # Clear previous imshow plot


    vmax_f = 10
    c = axs[1, 1].imshow(np.real(SC_MAP_var), extent=[qx[0], qx[-1], qy[-1], qy[0]], cmap = cmap, vmin = -vmax_f, vmax = vmax_f)
    axs[1, 1].set_title('$\delta [dB/dE]$, $L_{3}$', fontsize=12)
    
    # # Update the colorbar only if it's not already there
    # if not hasattr(update_2d_plot, 'colorbar_added') or not update_2d_plot.colorbar_added:
    #     fig2d.colorbar(c, ax=axs[1, 1])
    #     update_2d_plot.colorbar_added = True

    # Update the canvas for 2D plot
    canvas_2d.draw()

# Function to save the figure
def save_figure():
    file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Files", "*.png"), ("All Files", "*.*")])
    if file_path:
        fig3d.savefig(file_path)

def plot_tilt_thc_dep_or():
    tilt = tilt_dep_var.get()
    par = par_tilt_thc_var.get()
    return plot_tilt_thc_dep(tilt, par)

# Create the main window
root = tk.Tk()
root.title("EMLD SIMULATOR v01")

# Create a Notebook widget for tabs
notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True)

# Create a frame for the inputs and buttons
# frame = tk.Frame(root)
# frame.pack(pady=10)

frame = ttk.Frame(notebook, padding=10)
notebook.add(frame, text="General M")

settings_frame = ttk.Frame(notebook, padding=10)
notebook.add(settings_frame, text="Settings")

dyndiff_proc_frame = ttk.Frame(notebook, padding=10)
notebook.add(dyndiff_proc_frame, text="Dyndiff processing")

polycrystal_frame = ttk.Frame(notebook, padding=10)
notebook.add(polycrystal_frame, text="Polycrystal processing")

homog_sphere = tk.Button(polycrystal_frame, text="Homog sphere plot", command=sphere_plot)
homog_sphere.grid(row=0, column=0, padx=5, pady=5)

dyndiff_proc_frame.grid_columnconfigure(2, weight=0)  # Make column 1 fixed

load_dd = tk.Button(dyndiff_proc_frame, text="Load Dyndiff (from conv beam)", command=plot_data_Sc)
load_dd.grid(row=0, column=0, columnspan=2, pady=5)

load_sc_wv = tk.Button(dyndiff_proc_frame, text="Load sc_wave (from conv beam mult)", command=plot_data_wf)
load_sc_wv.grid(row=1, column=0, columnspan=2, pady=5)

tilt_dep_var = tk.StringVar()
tilt_dep_var.set('01X') # Default orientation
tilt_dep_menu = tk.OptionMenu(dyndiff_proc_frame, tilt_dep_var, '01X', '11X')
tilt_dep_menu.grid(row=2, column=0, padx=5, pady=5)

par_tilt_thc_var = tk.StringVar()
par_tilt_thc_var.set('tilt') # Default orientation
par_tilt_thc = tk.OptionMenu(dyndiff_proc_frame, par_tilt_thc_var, 'tilt', 'thc')
par_tilt_thc.grid(row=2, column=1, padx=5, pady=5)

# par_tilt_thc_var
# tilt = tilt_dep_var.get()

load_tilt_dep = tk.Button(dyndiff_proc_frame, text="Plot tilt/thc dep (FeRh)", command = plot_tilt_thc_dep_or)
load_tilt_dep.grid(row=3, column=0, columnspan=2, pady=5)

dir_fig = 'C:/Users/hajdu/Desktop/EMLD_visualization/EMLD_sim_v01/figures/EMCD-EMLD-scheme.png'
image = Image.open(dir_fig)  # Replace with the actual path to your image
print(np.shape(image))

image = image.resize((int(np.shape(image)[1]/2), int(np.shape(image)[0]/2)))#, Image.ANTIALIAS)  # Resize the image to fit
photo = ImageTk.PhotoImage(image)
image_label = ttk.Label(dyndiff_proc_frame, image=photo)
image_label.image = photo  # Keep a reference to avoid garbage collection
image_label.grid(row=4, column=0, columnspan=2,  sticky="nsew")

# dyndiff_proc_frame.grid_columnconfigure(2, maxsize=100)
 
# Add labels and entry widgets for Euler angles for both arrows (two columns)
# tk.Label(frame, text="Roll (red arrow):").grid(row=0, column=0, padx=5, pady=5)
# entry_roll_red = tk.Entry(frame)
# entry_roll_red.grid(row=0, column=1, padx=5, pady=5)

# tk.Label(frame, text="Pitch (red arrow):").grid(row=1, column=0, padx=5, pady=5)
# entry_pitch_red = tk.Entry(frame)
# entry_pitch_red.grid(row=1, column=1, padx=5, pady=5)

# tk.Label(frame, text="Yaw (red arrow):").grid(row=2, column=0, padx=5, pady=5)
# entry_yaw_red = tk.Entry(frame)
# entry_yaw_red.grid(row=2, column=1, padx=5, pady=5)

# tk.Label(frame, text="Roll (blue arrow):").grid(row=3, column=0, padx=5, pady=5)
# entry_roll_blue = tk.Entry(frame)
# entry_roll_blue.grid(row=3, column=1, padx=5, pady=5)

# tk.Label(frame, text="Pitch (blue arrow):").grid(row=4, column=0, padx=5, pady=5)
# entry_pitch_blue = tk.Entry(frame)
# entry_pitch_blue.grid(row=4, column=1, padx=5, pady=5)

# tk.Label(frame, text="Yaw (blue arrow):").grid(row=5, column=0, padx=5, pady=5)
# entry_yaw_blue = tk.Entry(frame)
# entry_yaw_blue.grid(row=5, column=1, padx=5, pady=5)

# Add labels and entry widgets for Euler angles for the secondary system (α, β, γ)
tk.Label(frame, text="Alpha (secondary coord):").grid(row=6, column=0, padx=5, pady=5)
entry_alpha = tk.Entry(frame)
entry_alpha.grid(row=6, column=1, padx=5, pady=5)

tk.Label(frame, text="Beta (secondary coord):").grid(row=7, column=0, padx=5, pady=5)
entry_beta = tk.Entry(frame)
entry_beta.grid(row=7, column=1, padx=5, pady=5)

tk.Label(frame, text="Gamma (secondary coord):").grid(row=8, column=0, padx=5, pady=5)
entry_gamma = tk.Entry(frame)
entry_gamma.grid(row=8, column=1, padx=5, pady=5)

# Dropdown for cube orientation selection
orientation_var = tk.StringVar()
orientation_var.set('001')  # Default orientation
orientation_menu = tk.OptionMenu(frame, orientation_var, '001', '011', '111')
orientation_menu.grid(row=6, column=2, padx=5, pady=5)

# Button to update the 3D plot
update_button = tk.Button(frame, text="Update 3D Plot", command=update_arrow)
update_button.grid(row=7, column=2, padx=5, pady=5)

# Button to update the 2D plot
update_2d_button = tk.Button(frame, text="Update 2D Plot", command=update_2d_plot)
update_2d_button.grid(row=8, column=2, padx=5, pady=5)

# Button to save the figure
save_button = tk.Button(frame, text="Save Figure", command=save_figure)
save_button.grid(row=9, column=2, padx=5, pady=5)

# Create the figure and canvas for 3D and 2D plots
fig3d = plt.Figure(figsize=(7, 7), dpi=100)
ax3d = fig3d.add_subplot(111, projection='3d')

alpha = np.linspace(0, 2 * np.pi, 100)  # Azimuthal angle (0 to 2pi)
beta = np.linspace(0, np.pi, 100)      # Polar angle (0 to pi)
alpha_grid, beta_grid = np.meshgrid(alpha, beta)

# Define the function f(alpha, beta)
f = (np.sin(2 * alpha_grid) * np.cos(2 * beta_grid)+2)/3

# Create the figure and canvas for the 2D plot
fig2d, axs = plt.subplots(2, 2, figsize=(7, 7), subplot_kw={})
polar_ax = fig2d.add_subplot(2, 2, 1, polar=True)
polar_ax_2 = fig2d.add_subplot(2, 2, 2, polar=True)

axs[0, 0].axis('off')
axs[0, 1].axis('off')
axs[1, 0].imshow(np.random.random((100, 100)), cmap='coolwarm')
axs[1, 1].imshow(np.random.random((100, 100)), cmap='coolwarm')
# polar_ax = fig2d.add_subplot(1, 2, 1, polar=True)

canvas_3d = FigureCanvasTkAgg(fig3d, master=frame)
canvas_widget = canvas_3d.get_tk_widget()
canvas_widget.grid(row=10, column=0, columnspan=2, sticky="nsew")

# canvas_3d = FigureCanvasTkAgg(fig3d, master=frame)
# canvas_3d.get_tk_widget().pack(side=tk.LEFT)

canvas_2d = FigureCanvasTkAgg(fig2d, master=frame)
canvas_widget2 = canvas_2d.get_tk_widget()
canvas_widget2.grid(row=10, column=2, columnspan=2, sticky="nsew")


# Create canvas to display 2D plot
# canvas_2d = FigureCanvasTkAgg(fig2d, master=frame)
# canvas_2d.get_tk_widget().pack(side=tk.LEFT)


# Create a figure and axis for 3D plot
# fig3d = plt.figure(figsize=(8, 6))
# ax3d = fig3d.add_subplot(111, projection='3d')
ax3d.set_title("3D Arrow and Cube", fontsize=14)

# Run the Tkinter main loop
root.mainloop()
