import numpy as np
from scipy.interpolate import PchipInterpolator

#Constant Calculations
param = {
    'rho_PM': 1000,
    'rho_cryst': 1500,
    'delta_H_cryst': 0,
    'c_p_PM': 4186,
    'U_PM_TM': 9.2450442e2,
    'rho_TM': 1000,
    'c_p_TM': 4186,
    'kv': np.pi/6,
    'U_TM_env': 8.2731443e0,
    # 'L': 22,
    'L': 24,
    'd_iPM': 3.18e-3,
    'd_aPM': 4.76e-3,
    'D_iTM': 1.5e-2,
    'D_aTM': 1.9e-2,
    'p_out': 1.01325e5,
    'eta_l': 0.6527 / 1000,    # Viscosity of water [Pa·s]
    'sigma_l': 0.06984,        # Surface tension water
}

param['A_TM_env'] = np.pi*param['D_aTM']*param['L']
param['A_TM_cross'] = np.pi*(param['D_aTM']**2 - param['D_iTM']**2)/4
param['A_PM_cross'] = (np.pi*param['d_iPM']**2)/4

MC_param = {
    'G': 0,
    'B': 0,
    'beta': 0,
    # 'beta_0': 2e4,
    'beta_0': 2e4*0,
    'beta_1': 1,
    'beta_2': 1,
    'Dil': 0,
    'domain': [0.7e-4, 4.3e-4],
    'coordinate': 'L',
    'mu_distribution': 2.5e-4,
    'sigma_distribution': 1e-4,
}

MC_param['n_init'] = lambda x: 1 * np.exp(-((x - MC_param['mu_distribution']) ** 2) / (2 * MC_param['sigma_distribution'] ** 2))

# bounds for inputs
bounds = {
    'mf_PM': [10/60/1000/1000*param['rho_PM'], 50/60/1000/1000*param['rho_PM']],
    'mf_TM': [50/60/1000/1000*param['rho_TM'], 1000/60/1000/1000*param['rho_TM']],
    'Q_g': [10/60/1000/1000, 50/60/1000/1000],
    'w_crystal': [0.001, 0.01],
    # 'mf_PM': [5/60/1000/1000*param['rho_PM'], 20/60/1000/1000*param['rho_PM']],
    # 'mf_TM': [2/60/60, 9/60/60],
    # 'Q_g': [5/60/1000/1000, 20/60/1000/1000],
    # 'T_PM_in': [40+273.15, 50+273.15],
    # 'w_crystal': [0.001, 0.001],
}

controller_param = {
    'dt': 5,
    'n_horizon': 10,
}

default_inputs = {
    'c_in': 0.19,
    'T_PM_in': 273.15 + 50,
    'T_TM_in': 273.15 + 50,
    'mf_PM': 20/60/1000,
    'mf_TM': 600/60/1000,
    'Q_g': 20/60/1000/1000,
    'w_crystal': 0.01,
}


def initialize_init_function(x_points, y_points):
    """
    Initialize a smooth interpolation function that passes through given points
    without oscillations. Uses monotonic cubic interpolation.

    Parameters:
    x_points (array-like): x-coordinates of data points
    y_points (array-like): y-coordinates of data points
    """

    # Convert inputs to numpy arrays
    x_cum = np.array(x_points)
    y_cum = np.array(y_points)

    # compute density from cumulative y
    x = x_cum[1:]
    y = y_cum[1:] - y_cum[:-1]

    # get domain
    first_nonzero = np.nonzero(y)[0][0]
    last_nonzero = len(y) - 1 - np.nonzero(y[::-1])[0][0]

    domain = [x[first_nonzero-1], x[last_nonzero+1]]

    # Check if inputs are valid
    if len(x) != len(y):
        raise ValueError("x_points and y_points must have the same length")
    if len(x) < 2:
        raise ValueError("At least two points are required for interpolation")
    if not np.all(np.diff(x) > 0):
        raise ValueError("x_points must be strictly increasing")

    # Create the PCHIP interpolator
    init_function = PchipInterpolator(x, y)

    # Add bound checking wrapper to prevent extrapolation
    x_min, x_max = x[0], x[-1]
    original_function = init_function

    def bounded_function(x_eval):
        """Wrapper function that handles out-of-bounds inputs"""
        x_eval = np.asarray(x_eval)
        y_eval = np.zeros_like(x_eval, dtype=float)

        # Handle scalar and array inputs
        if np.isscalar(x_eval):
            if x_eval < x_min:
                return y[0]
            elif x_eval > x_max:
                return y[-1]
            else:
                return float(original_function(x_eval))
        else:
            # Vector input
            mask_low = x_eval < x_min
            mask_high = x_eval > x_max
            mask_valid = ~(mask_low | mask_high)

            y_eval[mask_low] = y[0]
            y_eval[mask_high] = y[-1]
            y_eval[mask_valid] = original_function(x_eval[mask_valid])

            return y_eval

    init_function = bounded_function

    # Test the function with the input points to verify interpolation
    test_values = init_function(x)
    max_error = np.max(np.abs(test_values - y))
    if max_error > 1e-10:
        raise RuntimeError(f"Interpolation error too large: {max_error}")

    return init_function, domain


def slug_length(Roh_l, v_s, eta_l, d_sfc, sigma_l, epsilon_0):
    # fitted parameters (credit: A. C. Kufner, M. Rix, K.Wohlgemuth, Modeling of Continuous Slug Flow Cooling Crystallization towards Pharmaceutical Applications, Processes 11 (2023) 2637.)
    c1 = 1.969
    c2 = -1.102
    c3 = -0.035
    c4 = -0.176
    c5 = -0.0605

    Re = Roh_l * v_s * d_sfc / (eta_l)
    Ca = (eta_l * v_s) / sigma_l

    L_s = d_sfc * (c1 * (1 - epsilon_0) ** c2 * epsilon_0 ** c3 * Re ** c4 * Ca ** c5)
    L_g = L_s * (1 - epsilon_0) / epsilon_0
    L_UC = L_s + L_g
    return L_s, L_g, L_UC

if __name__ == '__main__':
    # plot initial distribution over domain to check if it is correct
    import matplotlib.pyplot as plt
    x = np.linspace(MC_param['domain'][0], MC_param['domain'][1], 1001)
    plt.plot(x, MC_param['n_init'](x))
    plt.show()