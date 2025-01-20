import numpy as np

def A_trans():
    Atrans = (1 / 2) * np.sqrt(3 / np.pi) * np.array([
            [1 / np.sqrt(2), 1j / np.sqrt(2), 0],
            [0, 0, 1],
            [-1 / np.sqrt(2), 1j / np.sqrt(2), 0]
        ])
    return Atrans

def L2_down_coeffs(alpha, beta, gamma):
    alpha = np.radians(alpha)
    beta = np.radians(beta)
    gamma = np.radians(gamma)
    """
    Calculate the coefficient matrices for A and B in the L2Down matrix.
    """
    # Pre-compute trigonometric terms for efficiency
    sin_beta = np.sin(beta)
    cos_beta = np.cos(beta)
    cos_2beta = np.cos(2 * beta)
    cos_4beta = np.cos(4 * beta)
    cos_4alpha = np.cos(4 * alpha)
    sin_4alpha = np.sin(4 * alpha)

    # Common term
    common_term = 8 * cos_4alpha * sin_beta**4 + 4 * cos_2beta + 7 * cos_4beta

    # Coefficients for A
    A_matrix = np.array([
        [
            -81 + 3 * common_term,
            6 * np.sqrt(2) * np.exp(1j * gamma) * sin_beta * (-4j * sin_4alpha * sin_beta**2 - (cos_4alpha - 9) * cos_beta + (cos_4alpha + 7) * np.cos(3 * beta)),
            12 * np.exp(2j * gamma) * sin_beta**2 * (7 * cos_2beta + cos_4alpha * (cos_2beta + 3) + 4j * cos_beta * sin_4alpha + 5)
        ],
        [
            6 * np.sqrt(2) * np.exp(-1j * gamma) * sin_beta * (4j * sin_4alpha * sin_beta**2 + (cos_4alpha - 9) * cos_beta - (cos_4alpha + 7) * np.cos(3 * beta)),
            -30 - 6 * common_term,
            6 * np.sqrt(2) * np.exp(1j * gamma) * sin_beta * (4j * sin_4alpha * sin_beta**2 + (cos_4alpha - 9) * cos_beta - (cos_4alpha + 7) * np.cos(3 * beta))
        ],
        [
            12 * np.exp(-2j * gamma) * sin_beta**2 * (7 * cos_2beta + cos_4alpha * (cos_2beta + 3) - 4j * cos_beta * sin_4alpha + 5),
            6 * np.sqrt(2) * np.exp(-1j * gamma) * sin_beta * (-4j * sin_4alpha * sin_beta**2 + (cos_4alpha - 9) * cos_beta - (cos_4alpha + 7) * np.cos(3 * beta)),
            15 + 3 * common_term
        ]
    ])

    # Coefficients for B
    B_matrix = np.array([
        [
            -63 - 3 * common_term,
            -6 * np.sqrt(2) * np.exp(1j * gamma) * sin_beta * (-4j * sin_4alpha * sin_beta**2 - (cos_4alpha - 9) * cos_beta + (cos_4alpha + 7) * np.cos(3 * beta)),
            -12 * np.exp(2j * gamma) * sin_beta**2 * (7 * cos_2beta + cos_4alpha * (cos_2beta + 3) + 4j * cos_beta * sin_4alpha + 5)
        ],
        [
            -6 * np.sqrt(2) * np.exp(-1j * gamma) * sin_beta * (4j * sin_4alpha * sin_beta**2 + (cos_4alpha - 9) * cos_beta - (cos_4alpha + 7) * np.cos(3 * beta)),
            -2 + 6 * common_term,
            -6 * np.sqrt(2) * np.exp(1j * gamma) * sin_beta * (4j * sin_4alpha * sin_beta**2 + (cos_4alpha - 9) * cos_beta - (cos_4alpha + 7) * np.cos(3 * beta))
        ],
        [
            -12 * np.exp(-2j * gamma) * sin_beta**2 * (7 * cos_2beta + cos_4alpha * (cos_2beta + 3) - 4j * cos_beta * sin_4alpha + 5),
            -6 * np.sqrt(2) * np.exp(-1j * gamma) * sin_beta * (-4j * sin_4alpha * sin_beta**2 + (cos_4alpha - 9) * cos_beta - (cos_4alpha + 7) * np.cos(3 * beta)),
            1 - 3 * common_term
        ]
    ])

    return A_matrix, B_matrix

def L3_down_coeffs(alpha, beta, gamma):
    alpha = np.radians(alpha)
    beta = np.radians(beta)
    gamma = np.radians(gamma)

    # Pre-compute trigonometric terms
    sin_beta = np.sin(beta)
    cos_beta = np.cos(beta)
    sin_4alpha = np.sin(4 * alpha)
    cos_4alpha = np.cos(4 * alpha)
    cos_2beta = np.cos(2 * beta)
    cos_4beta = np.cos(4 * beta)
    sin_beta_4 = sin_beta**4
    
    # Common term
    common_term = 8 * cos_4alpha * sin_beta_4 + 4 * cos_2beta + 7 * cos_4beta

    # Coefficient matrices
    CA = np.array([
        [-81 + 3 * common_term, 6 * np.sqrt(2) * np.exp(1j * gamma) * sin_beta * (-4j * sin_4alpha * sin_beta**2), 12 * np.exp(2j * gamma) * sin_beta**2 * (7 * cos_2beta + cos_4alpha * (cos_2beta + 3) + 5 + 4j * cos_beta * sin_4alpha)],
        [6 * np.sqrt(2) * np.exp(-1j * gamma) * sin_beta * (4j * sin_4alpha * sin_beta**2), -318 - 6 * common_term, 6 * np.sqrt(2) * np.exp(1j * gamma) * sin_beta * (4j * sin_4alpha * sin_beta**2)],
        [12 * np.exp(-2j * gamma) * sin_beta**2 * (7 * cos_2beta + cos_4alpha * (cos_2beta + 3) - 5 - 4j * cos_beta * sin_4alpha), 6 * np.sqrt(2) * np.exp(-1j * gamma) * sin_beta * (-4j * sin_4alpha * sin_beta**2), -561 + 3 * common_term]
    ])
    
    CB = np.array([
        [-63 - 3 * common_term, 6 * np.sqrt(2) * np.exp(1j * gamma) * sin_beta * (-(cos_4alpha - 9) * cos_beta + (cos_4alpha + 7) * np.cos(3 * beta)), 12 * np.exp(2j * gamma) * sin_beta**2 * (5 + cos_4alpha * (cos_2beta + 3))],
        [6 * np.sqrt(2) * np.exp(-1j * gamma) * sin_beta * ((cos_4alpha - 9) * cos_beta - (cos_4alpha + 7) * np.cos(3 * beta)), -194 - 6 * common_term, 6 * np.sqrt(2) * np.exp(1j * gamma) * sin_beta * ((cos_4alpha - 9) * cos_beta - (cos_4alpha + 7) * np.cos(3 * beta))],
        [12 * np.exp(-2j * gamma) * sin_beta**2 * (5 + cos_4alpha * (cos_2beta + 3)), 6 * np.sqrt(2) * np.exp(-1j * gamma) * sin_beta * ((cos_4alpha - 9) * cos_beta - (cos_4alpha + 7) * np.cos(3 * beta)), -383 - 3 * common_term]
    ])
    
    return CA, CB

# UP - correctly set, DOWN to be checked

def L2_up_coeffs(alpha, beta, gamma):
    alpha = np.radians(alpha)
    beta = np.radians(beta)
    gamma = np.radians(gamma)
    """
    Calculate the coefficient matrices for A and B in the L2Up matrix.
    """
    # Pre-compute trigonometric terms for efficiency
    sin_beta = np.sin(beta)
    cos_beta = np.cos(beta)
    cos_2beta = np.cos(2 * beta)
    cos_4beta = np.cos(4 * beta)
    cos_4alpha = np.cos(4 * alpha)
    sin_4alpha = np.sin(4 * alpha)

    # Common term
    common_term = 8 * cos_4alpha * sin_beta**4 + 4 * cos_2beta + 7 * cos_4beta

    # Coefficients for A
    A_matrix = np.array([
        [
            -15 - 3 * common_term,
            6 * np.sqrt(2) * np.exp(1j * gamma) * sin_beta * (4j * sin_4alpha * sin_beta**2 + (cos_4alpha - 9) * cos_beta - (cos_4alpha + 7) * np.cos(3 * beta)),
            -12 * np.exp(2j * gamma) * sin_beta**2 * (7 * cos_2beta + cos_4alpha * (cos_2beta + 3) + 4j * cos_beta * sin_4alpha + 5)
        ],
        [
            6 * np.sqrt(2) * np.exp(-1j * gamma) * sin_beta * (-4j * sin_4alpha * sin_beta**2 + (cos_4alpha - 9) * cos_beta - (cos_4alpha + 7) * np.cos(3 * beta)),
            30 + 6 * common_term,
            6 * np.sqrt(2) * np.exp(1j * gamma) * sin_beta * (-4j * sin_4alpha * sin_beta**2 - (cos_4alpha - 9) * cos_beta + (cos_4alpha + 7) * np.cos(3 * beta))
        ],
        [
            -12 * np.exp(-2j * gamma) * sin_beta**2 * (7 * cos_2beta + cos_4alpha * (cos_2beta + 3) - 4j * cos_beta * sin_4alpha + 5),
            6 * np.sqrt(2) * np.exp(-1j * gamma) * sin_beta * (4j * sin_4alpha * sin_beta**2 - (cos_4alpha - 9) * cos_beta + (cos_4alpha + 7) * np.cos(3 * beta)),
            81 - 3 * common_term
        ]
    ])

    # Coefficients for B
    B_matrix = np.array([
        [
            -1 + 3 * common_term,
            -6 * np.sqrt(2) * np.exp(1j * gamma) * sin_beta * (4j * sin_4alpha * sin_beta**2 + (cos_4alpha - 9) * cos_beta - (cos_4alpha + 7) * np.cos(3 * beta)),
            12 * np.exp(2j * gamma) * sin_beta**2 * (7 * cos_2beta + cos_4alpha * (cos_2beta + 3) + 4j * cos_beta * sin_4alpha + 5)
        ],
        [
            -6 * np.sqrt(2) * np.exp(-1j * gamma) * sin_beta * (-4j * sin_4alpha * sin_beta**2 + (cos_4alpha - 9) * cos_beta - (cos_4alpha + 7) * np.cos(3 * beta)),
            2 - 6 * common_term,
            -6 * np.sqrt(2) * np.exp(1j * gamma) * sin_beta * (-4j * sin_4alpha * sin_beta**2 - (cos_4alpha - 9) * cos_beta + (cos_4alpha + 7) * np.cos(3 * beta))
        ],
        [
            12 * np.exp(-2j * gamma) * sin_beta**2 * (7 * cos_2beta + cos_4alpha * (cos_2beta + 3) - 4j * cos_beta * sin_4alpha + 5),
            -6 * np.sqrt(2) * np.exp(-1j * gamma) * sin_beta * (4j * sin_4alpha * sin_beta**2 - (cos_4alpha - 9) * cos_beta + (cos_4alpha + 7) * np.cos(3 * beta)),
            63 + 3 * common_term
        ]
    ])

    return A_matrix, B_matrix

def L3_up_coeffs(alpha, beta, gamma):
    i = 1j  # Imaginary unit
    alpha = np.radians(alpha)
    beta = np.radians(beta)
    gamma = np.radians(gamma)

    # Define coefficients for A
    sin_beta = np.sin(beta)
    cos_beta = np.cos(beta)
    sin_4alpha = np.sin(4 * alpha)
    cos_4alpha = np.cos(4 * alpha)
    cos_2beta = np.cos(2 * beta)
    cos_4beta = np.cos(4 * beta)
    sin_beta_4 = sin_beta**4
    cos_3beta = np.cos(3 * beta)
    
    # Common term
    common_term = 8 * cos_4alpha * sin_beta_4 + 4 * cos_2beta + 7 * cos_4beta

    # Coefficient matrices
    CA = np.array([
        [561 - 3 * common_term, 
         6 * np.sqrt(2) * np.exp(1j * gamma) * sin_beta * (4j * sin_4alpha * sin_beta**2 + (cos_4alpha - 9)*cos_beta - cos_3beta * (cos_4alpha + 7)), 
         -12 * np.exp(2j * gamma) * sin_beta**2 * (7 * cos_2beta + cos_4alpha * (cos_2beta + 3) + 5 + 4j * cos_beta * sin_4alpha)],
        
        [6 * np.sqrt(2) * np.exp(-1j * gamma) * sin_beta * (-4j * sin_4alpha * sin_beta**2 + (cos_4alpha - 9)*cos_beta - cos_3beta * (cos_4alpha + 7)), 
         318 + 6 * common_term, 
         6 * np.sqrt(2) * np.exp(1j * gamma) * sin_beta * (-4j * sin_4alpha * sin_beta**2 - (cos_4alpha - 9)*cos_beta + cos_3beta * (cos_4alpha + 7))],
        
        [-12 * np.exp(-2j * gamma) * sin_beta**2 * (7 * cos_2beta + cos_4alpha * (cos_2beta + 3) + 5 - 4j * cos_beta * sin_4alpha), 
         6 * np.sqrt(2) * np.exp(-1j * gamma) * sin_beta * (4j * sin_4alpha * sin_beta**2 - (cos_4alpha - 9)*cos_beta + cos_3beta * (cos_4alpha + 7)), 
         81 - 3 * common_term]
    ])
    
    CB = np.array([
        [383 + 3 * common_term, 
         -6 * np.sqrt(2) * np.exp(1j * gamma) * sin_beta * (4j * sin_4alpha * sin_beta**2 + (cos_4alpha - 9)*cos_beta - cos_3beta * (cos_4alpha + 7)), 
         12 * np.exp(2j * gamma) * sin_beta**2 * (7 * cos_2beta + cos_4alpha * (cos_2beta + 3) + 5 + 4j * cos_beta * sin_4alpha)],
        
        [-6 * np.sqrt(2) * np.exp(-1j * gamma) * sin_beta * (-4j * sin_4alpha * sin_beta**2 + (cos_4alpha - 9)*cos_beta - cos_3beta * (cos_4alpha + 7)), 
         194 - 6 * common_term, 
         -6 * np.sqrt(2) * np.exp(1j * gamma) * sin_beta * (-4j * sin_4alpha * sin_beta**2 - (cos_4alpha - 9)*cos_beta + cos_3beta * (cos_4alpha + 7))],
        
        [12 * np.exp(-2j * gamma) * sin_beta**2 * (7 * cos_2beta + cos_4alpha * (cos_2beta + 3) + 5 - 4j * cos_beta * sin_4alpha), 
         -6 * np.sqrt(2) * np.exp(-1j * gamma) * sin_beta * (4j * sin_4alpha * sin_beta**2 - (cos_4alpha - 9)*cos_beta + cos_3beta * (cos_4alpha + 7)), 
         63 + 3 * common_term]
    ])
    
    return CA, CB