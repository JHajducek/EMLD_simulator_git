import numpy as np
import matplotlib.pyplot as plt

def load_plot_tilt_dep(dir,file, sel_thickness, title):
    thickness, tilt, intensity = np.loadtxt(dir + file, unpack=True)
    # Select thickness for plotting
    selected_thickness = [sel_thickness]
    tilt_1 = np.arctan(1/tilt)
    # Plot tilt dependence for each selected thickness
    for t in selected_thickness:
        mask = thickness == t
        tilt_1 = 1000*np.arctan(1/tilt[mask])
        plt.plot(tilt_1, intensity[mask], 'o-', label=title)

def load_plot_thc_dep(dir,file, sel_tilt, title):
    thickness, tilt, intensity = np.loadtxt(dir + file, unpack=True)
    # Select thickness for plotting
    selected_tilt = [sel_tilt]
    tilt_1 = np.arctan(np.sqrt(2)/tilt)
    # if tilt == 1000:                      # plot correctly for the 0 tilt !!! (1000 in the txt file)
    #     tilt_1 = 0
    # Plot tilt dependence for each selected thickness
    for t in selected_tilt:
        mask = tilt == t
        tilt_1 = 1000*np.arctan(np.sqrt(2)/tilt[mask])
        plt.plot(thickness[mask], intensity[mask], 'o-', label=title)

def plot_tilt_thc_dep(direction):
    if direction == 0:
        dir = 'C:/Users/hajdu/OneDrive - VUT/Magnetic_TEM_Dream_Team/EMLD-simulations/240913_FeRh_beam-tilt_txt-files/'
        thickness = 10.5
        # thickness = 15.0
        # thickness = 25.5
        # sel_tilt = 1000 # [001]
        sel_tilt = 1000    # [001]
        # sel_tilt = 32    # [0132]
        # sel_tilt = 8    # [018]

        file_EMCD_x = 'EMCD_01x_RDAS-data/EMCD_01x_X-comp-norm_RDAS/diff.txt'
        file_EMCD_y = 'EMCD_01x_RDAS-data/EMCD_01x_Y-comp-norm_RDAS/diff.txt'
        file_EMCD_z = 'EMCD_01x_RDAS-data/EMCD_01x_Z-comp-norm_RDAS/diff.txt'

        # title_EMCD_x = 'EMCD_x-10.5nm'
        # title_EMCD_y = 'EMCD_y-10.5nm'
        # title_EMCD_z = 'EMCD_z-10.5nm'

        title_EMCD_x = 'EMCD_x-[001]'
        title_EMCD_y = 'EMCD_y-[001]'
        title_EMCD_z = 'EMCD_z-[001]'

        file_EMLD_x = 'EMLD_01x_RDAS-data/EMLD_01x_X-comp-norm_RDAS/diff.txt'
        file_EMLD_y = 'EMLD_01x_RDAS-data/EMLD_01x_Y-comp-norm_RDAS/diff.txt'
        file_EMLD_z = 'EMLD_01x_RDAS-data/EMLD_01x_Z-comp-norm_RDAS/diff.txt'

        # title_EMLD_x = 'EMLD_x-10.5nm'
        # title_EMLD_y = 'EMLD_y-10.5nm'
        # title_EMLD_z = 'EMLD_z-10.5nm'

        title_EMLD_x = 'EMLD_x-[001]'
        title_EMLD_y = 'EMLD_y-[001]'
        title_EMLD_z = 'EMLD_z-[001]'

        plt.figure(figsize=(8,6))

        # Thickness dep ============================================================

        load_plot_thc_dep(dir,file_EMCD_x,sel_tilt, title_EMCD_x)
        load_plot_thc_dep(dir,file_EMCD_y,sel_tilt, title_EMCD_y)
        load_plot_thc_dep(dir,file_EMCD_z,sel_tilt, title_EMCD_z)
        load_plot_thc_dep(dir,file_EMLD_x,sel_tilt, title_EMLD_x)
        load_plot_thc_dep(dir,file_EMLD_y,sel_tilt, title_EMLD_y)
        load_plot_thc_dep(dir,file_EMLD_z,sel_tilt, title_EMLD_z)

        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
        plt.xlabel('Thickness (nm)')
        plt.ylim(-4, 4)
        plt.ylabel('S1-S2 (a.u.)')
        plt.title('Tilt dependence of the EMLD/EMCD signal in FM')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Tilt dep ============================================================

        # load_plot_tilt_dep(dir,file_EMCD_x,thickness, title_EMCD_x)
        # load_plot_tilt_dep(dir,file_EMCD_y,thickness, title_EMCD_y)
        # load_plot_tilt_dep(dir,file_EMCD_z,thickness, title_EMCD_z)
        # load_plot_tilt_dep(dir,file_EMLD_x,thickness, title_EMLD_x)
        # load_plot_tilt_dep(dir,file_EMLD_y,thickness, title_EMLD_y)
        # load_plot_tilt_dep(dir,file_EMLD_z,thickness, title_EMLD_z)

        # plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
        # plt.xlabel('Tilt in 01x dir (mrad)')
        # plt.ylim(-4, 4)
        # plt.xlim(-15, 360)
        # plt.ylabel('S1-S2 (a.u.)')
        # plt.title('Tilt dependence of the EMLD/EMCD signal in FM')
        # plt.legend()
        # plt.grid(True)
        # plt.show()
    if direction == 1:
        dir = 'C:/Users/hajdu/OneDrive - VUT/Magnetic_TEM_Dream_Team/EMLD-simulations/240913_FeRh_beam-tilt_txt-files/'
        # thickness = 10.5
        thickness = 15.0
        # thickness = 25.5
        # sel_tilt = 1000 # [001]
        # sel_tilt = 1000    # [11256]
        # sel_tilt = 32    # [1132]
        sel_tilt = 8    # [118]

        file_EMCD_x = 'EMCD_11x_RDAS-data/EMCD_11x_X-comp-norm_RDAS/diff.txt'
        file_EMCD_y = 'EMCD_11x_RDAS-data/EMCD_11x_Y-comp-norm_RDAS/diff.txt'
        file_EMCD_z = 'EMCD_11x_RDAS-data/EMCD_11x_Z-comp-norm_RDAS/diff.txt'

        title_EMCD_x = 'EMCD_x-15.0nm'
        title_EMCD_y = 'EMCD_y-15.0nm'
        title_EMCD_z = 'EMCD_z-15.0nm'

        # title_EMCD_x = 'EMCD_x-[118]'
        # title_EMCD_y = 'EMCD_y-[118]'
        # title_EMCD_z = 'EMCD_z-[118]'

        file_EMLD_x = 'EMLD_11x_RDAS-data/EMLD_11x_X-comp-norm_RDAS/diff.txt'
        file_EMLD_y = 'EMLD_11x_RDAS-data/EMLD_11x_Y-comp-norm_RDAS/diff.txt'
        file_EMLD_z = 'EMLD_11x_RDAS-data/EMLD_11x_Z-comp-norm_RDAS/diff.txt'

        title_EMLD_x = 'EMLD_x-15.0nm'
        title_EMLD_y = 'EMLD_y-15.0nm'
        title_EMLD_z = 'EMLD_z-15.0nm'

        # title_EMLD_x = 'EMLD_x-[118]'
        # title_EMLD_y = 'EMLD_y-[118]'
        # title_EMLD_z = 'EMLD_z-[118]'

        plt.figure(figsize=(8,6))

        # load_plot_thc_dep(dir,file_EMCD_x,sel_tilt, title_EMCD_x)
        # load_plot_thc_dep(dir,file_EMCD_y,sel_tilt, title_EMCD_y)
        # load_plot_thc_dep(dir,file_EMCD_z,sel_tilt, title_EMCD_z)
        # load_plot_thc_dep(dir,file_EMLD_x,sel_tilt, title_EMLD_x)
        # load_plot_thc_dep(dir,file_EMLD_y,sel_tilt, title_EMLD_y)
        # load_plot_thc_dep(dir,file_EMLD_z,sel_tilt, title_EMLD_z)

        # plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
        # plt.xlabel('Thickness (nm)')
        # plt.ylim(-4, 4)
        # plt.ylabel('S1-S2 (a.u.)')
        # plt.title('Tilt dependence of the EMLD/EMCD signal in FM')
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        load_plot_tilt_dep(dir,file_EMCD_x,thickness, title_EMCD_x)
        load_plot_tilt_dep(dir,file_EMCD_y,thickness, title_EMCD_y)
        load_plot_tilt_dep(dir,file_EMCD_z,thickness, title_EMCD_z)
        load_plot_tilt_dep(dir,file_EMLD_x,thickness, title_EMLD_x)
        load_plot_tilt_dep(dir,file_EMLD_y,thickness, title_EMLD_y)
        load_plot_tilt_dep(dir,file_EMLD_z,thickness, title_EMLD_z)

        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
        plt.xlabel('Tilt in 11x dir (mrad)')
        plt.ylim(-4, 4)
        plt.ylabel('S1-S2 (a.u.)')
        plt.title('Tilt dependence of the EMLD/EMCD signal in FM')
        plt.legend()
        plt.grid(True)
        plt.show()