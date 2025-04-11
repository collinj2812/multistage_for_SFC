import os

import pickle
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

def narx_data_input(states, inputs, l):
    data_matrix = np.hstack((states, inputs))

    # number of time steps
    n_time_steps = data_matrix.shape[0]

    # number of states
    n_states = states.shape[1]

    # construct X and Y
    X = []
    Y = []
    for t in range(n_time_steps - l + 1):
        X_i = []
        U_i = []
        for i in reversed(range(l)):
            X_i.append(states[i + t, :].reshape(1, -1))
            U_i.append(inputs[i + t, :].reshape(1, -1))
        X.append(np.hstack((*X_i, *U_i)))
    X = np.vstack(X)
    return X


def simulation_evaluation(x, u, model, save=False, save_dir='models/figures/', plot_bound=None, states=6, data_test=None,  **kwargs):
    """
    Evaluates a simulation and optionally saves both plots and data.

    Args:
        x: trajectory of states
        u: trajectory of inputs
        model: the trained model
        save: whether to save results
        save_dir: base directory for saving results
        **kwargs: can include 'bll' (std function) or 'cqr' (for quantile regression)

    Returns:
        dict: Dictionary containing all simulation data
    """
    # check if kwargs are provided
    bll_model = False
    cqr_model = False
    model_type = "nn"  # default model type

    if 'bll' in kwargs:
        bll_model = True
        std_function = kwargs['bll']
        model_type = "bll"
    elif 'cqr' in kwargs:
        cqr_model = True
        model_type = "cqr"

    # x is trajectory of states, u is trajectory of inputs
    # model is the trained model
    # construct data matrix using x and u for narx
    l = model.data.l

    # initialize list of states
    states_array = x[:l].copy()

    # list of predictions
    y_pred_list = []

    if bll_model:
        std_list = []
    elif cqr_model:
        y_pred_list_upper = []
        y_pred_list_lower = []

    # list of MSE
    mse_list = []

    y_pred_array_running = states_array.copy()

    # number of time steps
    n_time_steps = x.shape[0]

    for t in range(l, n_time_steps):
        # construct input for current time step
        x_i = narx_data_input(y_pred_array_running[-l:], u[t - l:t], l)
        y_pred = model.casadi_model(x_i)
        if bll_model:
            std_i = std_function(x_i)
            std_list.append(std_i)
        elif cqr_model:
            y_pred_upper = model.casadi_model_upper(x_i)
            y_pred_lower = model.casadi_model_lower(x_i)
            y_pred_list_upper.append(ca.reshape(y_pred_upper, 1, -1))
            y_pred_list_lower.append(ca.reshape(y_pred_lower, 1, -1))

        # append prediction to list
        y_pred_list.append(ca.reshape(y_pred, 1, -1))

        # append true state to list
        states_array = np.vstack((states_array, x[t]))

        # append prediction to y_pred_array
        y_pred_array_running = np.vstack((y_pred_array_running, y_pred.T))

        # calculate MSE
        mse = np.mean((x[t] - y_pred) ** 2)
        mse_list.append(mse)

    # convert lists to numpy arrays
    y_pred_array = np.vstack(y_pred_list)
    mse_array = np.array(mse_list)

    # calculate average MSE and coverage for CQR and BLL for prediction model if data_test is provided
    if data_test is not None:
        if cqr_model:
            prediction = np.array(model.casadi_model(data_test.X.T)).T
            mse_prediction = np.mean((data_test.Y - prediction) ** 2)
            prediction_upper = np.array(model.casadi_model_upper(data_test.X.T)).T
            prediction_lower = np.array(model.casadi_model_lower(data_test.X.T)).T
            coverage = np.mean((data_test.Y >= prediction_lower) & (data_test.Y <= prediction_upper))
            print(f'Coverage: {coverage}')
            print(f'MSE prediction: {mse_prediction}')
        elif bll_model:
            # prediction for test data
            prediction = np.array(model.casadi_model(data_test.X.T)).T
            mse_prediction = np.mean((data_test.Y - prediction) ** 2)
            prediction_upper = prediction + 2 * np.array(std_function(data_test.X.T)).T
            prediction_lower = prediction - 2 * np.array(std_function(data_test.X.T)).T
            coverage = np.mean((data_test.Y >= prediction_lower) & (data_test.Y <= prediction_upper))
            print(f'Coverage: {coverage}')
            print(f'MSE prediction: {mse_prediction}')
        else:
            prediction = np.array(model.casadi_model(data_test.X.T)).T
            mse_prediction = np.mean((data_test.Y - prediction) ** 2)
            print(f'MSE prediction: {mse_prediction}')

    # Create data dictionary to return and save
    data_dict = {
        'x': x,
        'u': u,
        'l': l,
        'y_pred': y_pred_array,
        'mse': mse_array,
        'model_type': model_type,
        'avg_mse': np.mean(mse_array),
        'mse_prediction': mse_prediction,
    }

    if bll_model:
        std = np.hstack(std_list).T
        data_dict['std'] = std
        data_dict['coverage'] = coverage
    elif cqr_model:
        y_pred_array_upper = np.vstack(y_pred_list_upper)
        y_pred_array_lower = np.vstack(y_pred_list_lower)
        data_dict['y_pred_upper'] = y_pred_array_upper
        data_dict['y_pred_lower'] = y_pred_array_lower
        data_dict['coverage'] = coverage


    if plot_bound is None:
        # plot results
        fig, ax = plt.subplots(states, 1, figsize=(6, states*1.2), sharex=True)

        if bll_model:
            for i in range(states):
                ax[i].plot(x[l:, i], label='True')
                ax[i].plot(y_pred_array[:, i], label='Predicted')
                ax[i].fill_between(range(len(y_pred_array)),
                                   y_pred_array[:, i] - 3 * std[:, i],
                                   y_pred_array[:, i] + 3 * std[:, i],
                                   alpha=0.2)
                ax[i].set_ylabel(f'State {i}')
                ax[i].legend()
        elif cqr_model:
            for i in range(states):
                ax[i].plot(x[l:, i], label='True')
                ax[i].plot(y_pred_array[:, i], label='Predicted')
                ax[i].fill_between(range(len(y_pred_array)),
                                   y_pred_array_lower[:, i],
                                   y_pred_array_upper[:, i],
                                   alpha=0.2)
                ax[i].set_ylabel(f'State {i}')
                ax[i].legend()
        else:
            for i in range(states):
                ax[i].plot(x[l:, i], label='True')
                ax[i].plot(y_pred_array[:, i], label='Predicted')
                ax[i].set_ylabel(f'State {i}')
                ax[i].legend()

    else:
        # plot results
        fig, ax = plt.subplots(states, 1, figsize=(6, states*1.2), sharex=True)

        if bll_model:
            for i in range(states):
                ax[i].plot(x[l+plot_bound[0]:plot_bound[1], i], label='True')
                ax[i].plot(y_pred_array[plot_bound[0]:plot_bound[1], i], label='Predicted')
                ax[i].fill_between(range(plot_bound[1]-plot_bound[0]),
                                   y_pred_array[plot_bound[0]:plot_bound[1], i] - 3 * std[plot_bound[0]:plot_bound[1], i],
                                   y_pred_array[plot_bound[0]:plot_bound[1], i] + 3 * std[plot_bound[0]:plot_bound[1], i],
                                   alpha=0.2)
                ax[i].set_ylabel(f'State {i}')
                ax[i].legend()

        elif cqr_model:
            for i in range(states):
                ax[i].plot(x[l+plot_bound[0]:plot_bound[1], i], label='True')
                ax[i].plot(y_pred_array[plot_bound[0]:plot_bound[1], i], label='Predicted')
                ax[i].fill_between(range(plot_bound[1]-plot_bound[0]),
                                   y_pred_array_lower[plot_bound[0]:plot_bound[1], i],
                                   y_pred_array_upper[plot_bound[0]:plot_bound[1], i],
                                   alpha=0.2)
                ax[i].set_ylabel(f'State {i}')
                ax[i].legend()

        else:
            for i in range(states):
                ax[i].plot(x[l+plot_bound[0]:plot_bound[1], i], label='True')
                ax[i].plot(y_pred_array[plot_bound[0]:plot_bound[1], i], label='Predicted')
                ax[i].set_ylabel(f'State {i}')
                ax[i].legend()

    ax[-1].set_xlabel('Time')
    fig.tight_layout()
    # fig.show()

    # print average MSE
    print(f'Average MSE: {np.mean(mse_array)}')

    # Save results if requested
    if save:
        # Create directories based on model type
        model_dir = os.path.join(save_dir, model_type)
        os.makedirs(model_dir, exist_ok=True)

        # Define save paths
        svg_path = os.path.join(model_dir, f'{model_type}_simulation.svg')
        data_path = os.path.join(model_dir, f'{model_type}_simulation_data.pkl')

        # Save plot as SVG
        fig.savefig(svg_path, format='svg', bbox_inches='tight')
        print(f"Plot saved to {svg_path}")

        # Save data as pickle
        with open(data_path, 'wb') as f:
            pickle.dump(data_dict, f)
        print(f"Data saved to {data_path}")

    return data_dict