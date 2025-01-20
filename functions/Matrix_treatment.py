import numpy as np


def rotation_matrix_z(alpha):
    alpha = np.radians(alpha)
    return np.array([
        [np.cos(alpha), -np.sin(alpha), 0],
        [np.sin(alpha), np.cos(alpha), 0],
        [0, 0, 1]
    ])
def rotation_matrix_y(beta):
    beta = np.radians(beta)
    return np.array([
        [np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)]
    ])

def rotation_matrix_x(gamma):
    gamma = np.radians(gamma)
    return np.array([
        [1, 0, 0],
        [0, np.cos(gamma), -np.sin(gamma)],
        [0, np.sin(gamma), np.cos(gamma)]
    ])

# Function to compute the rotation matrix for given Euler angles (roll, pitch, yaw)
# def euler_to_rotation_matrix(alpha, beta, gamma):
#     R = rotation_matrix_x(gamma) @ rotation_matrix_y(beta) @ rotation_matrix_z(alpha)
#     return R

def ROTATE_EULER(alpha, beta, gamma):
    # R = rotation_matrix_x(gamma) @ rotation_matrix_y(beta) @ rotation_matrix_z(alpha)
    R = np.dot(rotation_matrix_z(alpha), np.dot(rotation_matrix_y(beta), rotation_matrix_z(gamma)))
    return R

def rotate_vector_map(x_map, y_map, z_map, alpha, beta, gamma):
    # Get the rotation matrix
    R = ROTATE_EULER(alpha, beta, gamma)

    # Stack the component arrays into a 3xN array (each column is a vector)
    vectors = np.stack((x_map.ravel(), y_map.ravel(), z_map.ravel()), axis=0)

    # Apply the rotation
    rotated_vectors = R @ vectors

    # Reshape back to original shape
    x_rot, y_rot, z_rot = rotated_vectors

    return x_rot.reshape(x_map.shape), y_rot.reshape(y_map.shape), z_rot.reshape(z_map.shape)

def inverse_rotation_matrix(alpha, beta, gamma):
    # Rotation matrix for Euler angles alpha, beta, gamma
    R = ROTATE_EULER(alpha, beta, gamma)
    
    # Inverse of the rotation matrix
    R_inv = np.linalg.inv(R)
    return R_inv

def reverse_rotate_vector(rotated_vector, alpha, beta, gamma):
    # Get the inverse rotation matrix
    R_inv = inverse_rotation_matrix(alpha, beta, gamma)

    # Apply the inverse rotation
    original_vector = R_inv @ rotated_vector

    return original_vector

def reverse_rotate_vector_map(x_map, y_map, z_map, alpha, beta, gamma):
    # Get the rotation matrix
    R_inv = inverse_rotation_matrix(alpha, beta, gamma)

    # Stack the component arrays into a 3xN array (each column is a vector)
    vectors = np.stack((x_map.ravel(), y_map.ravel(), z_map.ravel()), axis=0)

    # Apply the rotation
    rotated_vectors = R_inv @ vectors

    # Reshape back to original shape
    x_rot, y_rot, z_rot = rotated_vectors
    return x_rot.reshape(x_map.shape), y_rot.reshape(y_map.shape), z_rot.reshape(z_map.shape)



# Function to get rotation matrix for selected cube orientation
def orientation_to_rotation_matrix(orientation):
    if orientation == '001':
        # Align 001 with z and 100 with x (identity rotation for simplicity)
        return np.eye(3)
    elif orientation == '011':
        # Align 011 with z and 100 with x
        return np.array([[1, 0, 0],
                         [0, 1/np.sqrt(2), -1/np.sqrt(2)],
                         [0, 1/np.sqrt(2), 1/np.sqrt(2)]])
    elif orientation == '111':
        # Align 111 with z and -211 with x
        return np.array([[-0.8165, 0,  0.5774],
                         [0.4082,  -0.7071, 0.5774],
                         [0.4082,  0.7071,  0.5774]])
    else:
        return np.eye(3)