import numpy as np
import casadi as ca
import do_mpc

import sys
sys.path.append('../')
import config_mpc
import data_based_do_mpc_model_nn

class MPC_NN:
    def __init__(self, sfc_model, data, nn_model):
        self.sfc_model = sfc_model
        self.data = data
        self.nn_model = nn_model

    def setup_mpc(self, n_horizon, t_step, bounds, l):
        self.n_horizon = n_horizon
        self.t_step_controller = t_step
        self.bounds = bounds

        # setup databased model
        self.db_model = data_based_do_mpc_model_nn.data_based_model(self.nn_model.casadi_model, self.data.l, no_states=len(self.data.keys_states), no_inputs=len(self.data.keys_inputs), opt_inputs=self.data.keys_inputs)

        # setup controller
        self.controller = do_mpc.controller.MPC(self.db_model)

        self.controller.settings.n_horizon = n_horizon
        self.controller.settings.t_step = t_step
        self.controller.settings.store_full_solution = True
        self.controller.settings.use_terminal_bounds = True
        self.controller.settings.nlpsol_opts = {'ipopt.max_iter': 2000}
        self.controller.settings.set_linear_solver('MA57')

        # scaling
        for i in range(l):
            self.controller.scaling['_x', f'x_k-{i}'] = config_mpc.state_scaling
        for i in range(1, l):
            self.controller.scaling['_x', f'u_k-{i}'] = config_mpc.input_scaling
        #
        # self.controller.scaling['_u', 'mf_PM'] = config_mpc.input_scaling[0]
        # self.controller.scaling['_u', 'mf_TM'] = config_mpc.input_scaling[1]
        # self.controller.scaling['_u', 'Q_g'] = config_mpc.input_scaling[2]

        # cost function
        stage_cost = self.db_model.aux['maximize_d50'] + self.db_model.aux['maximize_mf_PM'] + self.db_model.aux['minimize_mf_TM']
        terminal_cost = self.db_model.aux['maximize_d50']
        self.controller.set_objective(lterm=stage_cost, mterm=terminal_cost)

        # penalize changes in input
        penalty = 5e6
        self.controller.set_rterm(mf_PM=1e4*penalty, mf_TM=1e3*penalty, Q_g=8e9*penalty)

        # input constraints
        set_inputs_keys = self.data.keys_inputs
        if 'w_crystal' in self.data.keys_inputs:
            set_inputs_keys.remove('w_crystal')
        for key in set_inputs_keys:
            self.controller.bounds['lower', '_u', key], self.controller.bounds['upper', '_u', key] = self.bounds[key]

        # state constraints
        self.controller.set_nl_cons('d90_constraint', expr=self.db_model.aux['d90']-self.db_model.tvp['max_d90'], ub=0, soft_constraint=True, penalty_term_cons=1e7)

        # tvp constraints
        tvp_template = self.controller.get_tvp_template()

        def tvp_fun(t_now):
            tvp_template['_tvp', :, 'max_d90'] = config_mpc.max_d90
            tvp_template['_tvp', :, 'w_crystal'] = config_mpc.first_w_crystal
            if t_now > config_mpc.time_change_w_crystal:
                tvp_template['_tvp', :, 'w_crystal'] = config_mpc.second_w_crystal
            return tvp_template

        self.controller.set_tvp_fun(tvp_fun)
        self.controller.setup()

        self.controller.set_initial_guess()


def initialize_narx(x0, u0, approx_obj):
    # return vector containing l times the initial state and initial input
    if approx_obj.l > 1:
        x_full = np.concatenate(np.array([x0.T for _ in range(approx_obj.l)]))
        u_full = np.concatenate(np.array([u0 for _ in range(approx_obj.l - 1)]))
        return np.concatenate((x_full, u_full))
    else:
        return x0

def mpc_input(sfc_model, data, old_x, w_crystal=None):
    # split old_x into states and inputs
    no_states = len(data.keys_states)
    if w_crystal is None:
        no_inputs = len(data.keys_inputs)
    else:
        no_inputs = len(data.keys_inputs) + 1

    x = old_x[:no_states*data.l]
    u = old_x[no_states*data.l:]

    # delete oldest state and input
    x = x[:-no_states]
    u = u[:-no_inputs]

    # extract states needed for MPC from output from physical model
    # states: data.keys_states
    # inputs: data.keys_inputs
    x_new = [sfc_model.output[key][-1] for key in data.keys_states]
    u_new = [sfc_model.output[key][-1] for key in data.keys_inputs]

    if w_crystal is not None:
        u_new.append(np.array(w_crystal))

    # append new states and inputs
    x = np.concatenate((x_new, x))
    u = np.concatenate((u_new, u))

    return np.concatenate((x, u))

def model_input(inputs, u_mpc, default_inputs, w_crystal=None):
    # construct inputs for physical model from MPC output
    # inputs must be: c_in, T_PM_in, T_TM_in, mf_PM, mf_TM, Q_g, w_crystal
    u_mpc_dict = dict(zip(inputs, u_mpc))

    inputs_list = []
    if 'c_in' in inputs:
        inputs_list.append(u_mpc_dict['c_in'])
    else:
        inputs_list.append(default_inputs['c_in'])

    if 'T_PM_in' in inputs:
        inputs_list.append(u_mpc_dict['T_PM_in'])
    else:
        inputs_list.append(default_inputs['T_PM_in'])

    if 'T_TM_in' in inputs:
        inputs_list.append(u_mpc_dict['T_TM_in'])
    else:
        inputs_list.append(default_inputs['T_TM_in'])

    if 'mf_PM' in inputs:
        inputs_list.append(u_mpc_dict['mf_PM'].squeeze())
    else:
        inputs_list.append(default_inputs['mf_PM'])

    if 'mf_TM' in inputs:
        inputs_list.append(u_mpc_dict['mf_TM'].squeeze())
    else:
        inputs_list.append(default_inputs['mf_TM'])

    if 'Q_g' in inputs:
        inputs_list.append(u_mpc_dict['Q_g'].squeeze())
    else:
        inputs_list.append(default_inputs['Q_g'])

    if w_crystal is None:
        if 'w_crystal' in inputs:
            inputs_list.append(u_mpc_dict['w_crystal'])
        else:
            inputs_list.append(default_inputs['w_crystal'])
    else:
        inputs_list.append(w_crystal)

    return inputs_list