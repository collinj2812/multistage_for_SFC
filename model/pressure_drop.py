import numpy as np

def pressure_drop(L_SFC, Q_PM, Q_TM, Q_g):
    rho_l = 1000
    v_s = 0.1  # ???
    d_sfc = 3.18 / 1000
    eta_l = 0.6527 / 1000
    sigma_l = 0.06984
    epsilon_0 = Q_g / (Q_PM+Q_g)
    eta_g = 20 * 10 ** (-6)
    theta = 72
    phi = 0.1  # ???

    # Parameter
    c1 = 1.969
    c2 = -1.102
    c3 = -0.035
    c4 = -0.176
    c5 = -0.0605

    Re = rho_l * v_s * d_sfc / (eta_l)
    Ca = (eta_l * v_s) / sigma_l

    L_s = d_sfc * (c1 * (1 - epsilon_0) ** c2 * epsilon_0 ** c3 * Re ** c4 * Ca ** c5)
    L_g = L_s * (1 - epsilon_0) / epsilon_0
    L_UC = L_s + L_g

    # Parameters
    N_uc = 2 * L_SFC / (L_s + L_g) - 1
    eta_sus = eta_l * (1 + 2.5 * phi)
    Pres_drop = (epsilon_0) * 32 * eta_sus * v_s / (d_sfc ** 2) + (1 - epsilon_0) * 32 * eta_g * v_s / (d_sfc ** 2) + N_uc * (4 * sigma_l) * np.cos(np.radians(theta)) / (d_sfc)



    return Pres_drop