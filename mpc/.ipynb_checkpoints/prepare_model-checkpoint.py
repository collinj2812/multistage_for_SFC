# prepare model as shown in jupyter notebook
import numpy as np
import sys
sys.path.append('./model')
sys.path.append('./mpc')
import config_mpc
import config
import sfc_model
from tqdm import tqdm

def prepare_model():
    N_discr = 30  # discretization for outer tube
    dt = 5  # time step of simulation
    
    model_param = config.param  # parameters needed for model
    MC_param = config.MC_param  # parameters needed for Monte Carlo simulation part
    
    # generate smooth initial function from measurements for sampling
    x_points = 1e-6*np.array([0, 2, 2.5, 3.2, 4, 5, 6.3, 8, 10.1, 12.7, 16, 20.2, 25.4, 32, 40.3, 50.8, 64, 80.6, 101.6, 128, 161.3, 203.2, 256, 322.5, 406.4, 512, 645.1, 812.7, 1024, 1290.2, 1625.5, 2048, 2580.3])
    # cumulative
    y_points = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.006, 0.02, 0.09, 0.3, 0.8, 0.98, 0.99, 1, 1, 1, 1, 1, 1, 1, 1])
    n_init, domain = config.initialize_init_function(x_points, y_points)

    # overwrite default parameters
    MC_param['n_init'] = n_init
    MC_param['domain'] = domain
    
    # setup model
    model = sfc_model.SFCModel()
    model.setup(model_param['L'], dt, N_discr, model_param, MC_param)

    print('Running pre simulation')
    for _ in tqdm(range(config_mpc.steps_pre_simulation)):
        model.make_step(config_mpc.c_0, config_mpc.T_PM_0, config_mpc.T_TM_0, config_mpc.mf_PM_0, config_mpc.mf_TM_0, config_mpc.Q_g_0, config_mpc.w_crystal_0)

    return model