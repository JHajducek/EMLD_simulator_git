import numpy as np
from numpy import random, cos, sin, sqrt, pi
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

# WORKING!
def rand_sphere(n):
    """n points distributed evenly on the surface of a unit sphere""" 
    z = 2 * random.rand(n) - 1   # uniform in -1, 1
    t = 2 * pi * random.rand(n)   # uniform in 0, 2*pi
    x = sqrt(1 - z**2) * cos(t)
    y = sqrt(1 - z**2) * sin(t)
    return x, y, z

# WORKING!
def fibonacci_sphere(samples=1000):

    points = []
    X = []
    Y = []
    Z = []
    phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        X.append(x)
        Y.append(y)
        Z.append(z)
        points.append((x, y, z))

    return np.array(X),np.array(Y),np.array(Z)

def sphere_plot():

    fig = plt.figure(figsize=(15, 5))

    # Generate different sets of angles for each subplot
    num_points = 1000

    # Subplot 1
    ax1 = fig.add_subplot(131, projection='3d')
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 40)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    ax1.scatter(x_sphere, y_sphere, z_sphere)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Evenly distributed angles')

    ax2 = fig.add_subplot(132, projection='3d')
    x, y, z = rand_sphere(1000)
    ax2.scatter(x, y, z)
    ax2.set_title('Randomized sphere')

    ax3 = fig.add_subplot(133, projection='3d')
    x, y, z = fibonacci_sphere(1000)
    ax3.scatter(x, y, z, color='red', s=20)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title('Fibonacci sphere')
    # ax.scatter(x, y, z, color='red', s=20)
    plt.show()