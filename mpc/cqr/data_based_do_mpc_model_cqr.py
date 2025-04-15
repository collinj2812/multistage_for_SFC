import do_mpc
import casadi as ca


def data_based_model(casadi_NN, casadi_NN_lower, casadi_NN_upper, l, no_states, no_inputs, opt_inputs):
    model = do_mpc.model.Model('discrete', 'SX')

    state_list = []
    input_list = []

    for i in range(l):
        state_list.append(model.set_variable('_x', f'x_k-{i}', shape=no_states))

    # check list of inputs, set inputs that are not optimized
    # possible inputs: c_in, T_PM_in, T_TM_in, mf_PM, mf_TM, Q_g, w_crystal
    inputs = []
    if 'c_in' in opt_inputs:
        c_in = model.set_variable(var_type='_u', var_name='c_in')
        inputs.append(c_in)
    if 'T_PM_in' in opt_inputs:
        T_PM_in = model.set_variable(var_type='_u', var_name='T_PM_in')
        inputs.append(T_PM_in)
    if 'T_TM_in' in opt_inputs:
        T_TM_in = model.set_variable(var_type='_u', var_name='T_TM_in')
        inputs.append(T_TM_in)
    if 'mf_PM' in opt_inputs:
        mf_PM = model.set_variable(var_type='_u', var_name='mf_PM')
        inputs.append(mf_PM)
    if 'mf_TM' in opt_inputs:
        mf_TM = model.set_variable(var_type='_u', var_name='mf_TM')
        inputs.append(mf_TM)
    if 'Q_g' in opt_inputs:
        Q_g = model.set_variable(var_type='_u', var_name='Q_g')
        inputs.append(Q_g)
    # if 'w_crystal' in opt_inputs:
    #     w_crystal = model.set_variable(var_type='_u', var_name='w_crystal')
    #     inputs.append(w_crystal)
    #

    # w_crystal should be set and changed during simulation but not optimized
    dummy_tvp_w_crystal = model.set_variable('_tvp', 'w_crystal')

    if 'w_crystal' in opt_inputs:
        # w_crystal = model.set_variable(var_type='_u', var_name='w_crystal')
        inputs.append(dummy_tvp_w_crystal)


    for i in range(l - 1):
        input_list.append(model.set_variable('_x', f'u_k-{i + 1}', shape=no_inputs))


    # uncertainty values
    uncertainty_mean = model.set_variable('_p', 'uncertainty_mean', shape=no_states)
    uncertainty_upper = model.set_variable('_p', 'uncertainty_upper', shape=no_states)
    uncertainty_lower = model.set_variable('_p', 'uncertainty_lower', shape=no_states)

    # set tvp variables
    T_constraint = model.set_variable('_tvp', 'T_constraint')
    set_size = model.set_variable('_tvp', 'max_d90')

    u_k = ca.vertcat(*inputs)

    # define input for NN
    x = ca.vertcat(*state_list)
    u = ca.vertcat(u_k, ca.vertcat(*input_list))

    # input for NN
    input_NN = ca.vertcat(x, u)

    # output from NN
    x_next = casadi_NN(input_NN)

    upper_prediction = casadi_NN_upper(input_NN)
    lower_prediction = casadi_NN_lower(input_NN)

    # expressions to access states
    T_PM = model.set_expression('T_PM', state_list[0][0])
    T_TM = model.set_expression('T_TM', state_list[0][1])
    c = model.set_expression('c', state_list[0][2])
    d10 = model.set_expression('d10', state_list[0][3])
    d50 = model.set_expression('d50', state_list[0][4])
    d90 = model.set_expression('d90', state_list[0][5])


    model.set_expression('maximize_d50', - 1e5 * d50)
    model.set_expression('maximize_mf_PM', - 1e3 * mf_PM)
    model.set_expression('minimize_mf_TM', 1e2 * mf_TM)

    model.set_expression('pred_upper', upper_prediction)
    model.set_expression('pred_lower', lower_prediction)

    # define rhs
    model.set_rhs('x_k-0',
                  uncertainty_mean * x_next + uncertainty_upper * upper_prediction + uncertainty_lower * lower_prediction)

    for i in range(l - 1):
        model.set_rhs(f'x_k-{i + 1}', state_list[i])

    model.set_rhs(f'u_k-{1}', u_k)

    for i in range(l - 2):
        model.set_rhs(f'u_k-{i + 2}', input_list[i])

    model.setup()
    return model
