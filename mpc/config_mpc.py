# define some variables
import numpy as np

state_scaling = np.array([3e2, 3e2, 1.6e-1, 4e-4, 4e-4, 4e-4])
input_scaling = np.array([2e-4, 2e-3, 2e-7, 5e-3])

max_d90 = 6.0e-4
dt_sfc_model = 5
N_discr_sfc_model = 50
n_horizon = 20
dt_controller = 50
dt_databased = 50

first_w_crystal = 0.01
second_w_crystal = 0.001

time_change_w_crystal = 35*50

# initial state and inputs
steps_pre_simulation = 500
T_PM_0 = 310
T_TM_0 = 310
c_0 = 0.18
d10_0 = 1e-5
d50_0 = 2e-5
d90_0 = 3e-5

mf_PM_0 = 2e-5
mf_TM_0 = 2e-4
Q_g_0 = 2e-7
w_crystal_0 = 0.005

# simulation parameters
t_steps = 150
