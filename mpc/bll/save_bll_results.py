import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, ImageMagickWriter
import do_mpc


def format_tensor_or_ndarray(arr):
    """Format tensor or ndarray content into a string representation."""
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    if isinstance(arr, np.ndarray):
        return f"Shape: {arr.shape}\nDtype: {arr.dtype}\nValues: {arr.tolist()}"
    return str(arr)


def format_dict(d, indent=0):
    """Format dictionary content with proper indentation."""
    result = []
    for key, value in d.items():
        if isinstance(value, dict):
            result.append(" " * indent + f"{key}:")
            result.append(format_dict(value, indent + 2))
        else:
            result.append(" " * indent + f"{key}: {value}")
    return "\n".join(result)


def save_object_details(obj, name, base_path="results"):
    """Save object details to a text file with proper formatting."""
    base_path = Path(base_path)
    base_path.mkdir(exist_ok=True)

    filename = f"{name}.txt"
    filepath = base_path / filename

    with open(filepath, 'w') as f:
        f.write(f"=== Object: {name} ===\n")
        f.write(f"Type: {type(obj).__name__}\n\n")

        if isinstance(obj, (torch.Tensor, np.ndarray)):
            f.write(format_tensor_or_ndarray(obj))
        elif isinstance(obj, (list, tuple)):
            f.write("Values:\n")
            for i, item in enumerate(obj):
                f.write(f"[{i}] {str(item)}\n")
        elif isinstance(obj, dict):
            f.write(format_dict(obj))
        elif hasattr(obj, '__dict__'):
            for attr_name, attr_value in obj.__dict__.items():
                f.write(f"\n--- {attr_name} ---\n")
                if isinstance(attr_value, (torch.Tensor, np.ndarray)):
                    f.write(format_tensor_or_ndarray(attr_value))
                elif isinstance(attr_value, dict):
                    f.write(format_dict(attr_value))
                else:
                    f.write(str(attr_value))
        else:
            f.write(str(obj))


def save_mpc_details(mpc_obj, base_path):
    """Save MPC controller details."""
    mpc_dir = base_path / "mpc"
    mpc_dir.mkdir(exist_ok=True)

    if hasattr(mpc_obj, 'bounds'):
        save_object_details(mpc_obj.bounds, "bounds", mpc_dir)
    if hasattr(mpc_obj, 'controller'):
        controller_dir = mpc_dir / "controller"
        controller_dir.mkdir(exist_ok=True)
        controller = mpc_obj.controller

        # Save controller attributes
        controller_attrs = [
            'S', 'aux_struct', 'data', 'flags', 'lam_g_num', 'lam_x_num',
            'model', 'nlp', 'nlp_cons', 'opt_p', 'opt_x', 'settings',
            'slack_cost', 'solver_stats'
        ]
        for attr in controller_attrs:
            if hasattr(controller, attr):
                save_object_details(getattr(controller, attr), attr, controller_dir)


def save_data_class_details(data, base_path):
    """Save all components of the DataClass object."""
    data_dir = base_path / "data"
    data_dir.mkdir(exist_ok=True)

    # Save main arrays
    arrays = [
        'X', 'X_scaled', 'X_train', 'X_val', 'X_train_sc2', 'X_val_sc2',
        'Y', 'Y_scaled', 'Y_train', 'Y_val', 'Y_train_sc2', 'Y_val_sc2',
        'inputs', 'states'
    ]
    for array_name in arrays:
        if hasattr(data, array_name):
            save_object_details(getattr(data, array_name), array_name, data_dir)

    # Save lists and other attributes
    other_attrs = [
        'keys_inputs', 'keys_states', 'databased_pred', 'raw_data',
        'remove_first', 'split_in_three', 'l'
    ]
    for attr in other_attrs:
        if hasattr(data, attr):
            save_object_details(getattr(data, attr), attr, data_dir)

    # Save scalers
    if hasattr(data, 'scaler_X'):
        scaler_dir = data_dir / "scaler_X"
        scaler_dir.mkdir(exist_ok=True)
        scaler_attrs = ['copy', 'mean_', 'scale_', 'var_', 'n_features_in_',
                        'n_samples_seen_', 'with_mean', 'with_std']
        for attr in scaler_attrs:
            if hasattr(data.scaler_X, attr):
                save_object_details(getattr(data.scaler_X, attr), attr, scaler_dir)

    if hasattr(data, 'scaler_Y'):
        scaler_dir = data_dir / "scaler_Y"
        scaler_dir.mkdir(exist_ok=True)
        for attr in scaler_attrs:
            if hasattr(data.scaler_Y, attr):
                save_object_details(getattr(data.scaler_Y, attr), attr, scaler_dir)


def save_results(bll_model=None, mpc_bll=None, data=None, sfc_model=None,
                 param_sfc_model=None, figure=None, mpc_data=None, figure_name="plot", save_dir="results"):
    """
    Main function to save all results and model details.

    Parameters:
    -----------
    bll_model : BLL model object, optional
        The Bayesian Last Layer model
    mpc_bll : MPC BLL object, optional
        The MPC controller with BLL
    data : DataClass object, optional
        The data class object containing all data arrays and scalers
    sfc_model : SFC model object, optional
        The SFC model object
    param_sfc_model : dict, optional
        Parameters for the SFC model
    figure : matplotlib.figure.Figure, optional
        The figure object to be saved
    figure_name : str, default="plot"
        Name of the figure file (without extension)
    save_dir : str, default="results"
        Directory where results will be saved
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(save_dir) / timestamp
    base_dir.mkdir(parents=True, exist_ok=True)

    # Save BLL model details
    if bll_model is not None:
        bll_dir = base_dir / "bll_model"
        bll_dir.mkdir(exist_ok=True)

        # Save main model attributes
        bll_attrs = [
            'X_train', 'X_train_sc2', 'X_val', 'X_val_sc2',
            'Y_train', 'Y_train_sc2', 'Y_val', 'Y_val_sc2',
            'bll_model', 'casadi_model', 'cb_early_stopping',
            'learning_rate', 'metadata_data', 'n_epochs', 'n_neurons',
            'optimizer', 'output_model', 'scaler', 'std_with_noise',
            'std_without_noise'
        ]
        for attr in bll_attrs:
            if hasattr(bll_model, attr):
                save_object_details(getattr(bll_model, attr), attr, bll_dir)

    # Save MPC BLL details
    if mpc_bll is not None:
        mpc_dir = base_dir / "mpc_bll"
        mpc_dir.mkdir(exist_ok=True)
        save_mpc_details(mpc_bll, mpc_dir)

        # Save additional MPC specific attributes
        mpc_attrs = [
            'mpc_graphics', 'mpc_data', 'n_horizon', 'n_steps',
            'db_model', 'controller_param'
        ]
        for attr in mpc_attrs:
            if hasattr(mpc_bll, attr):
                save_object_details(getattr(mpc_bll, attr), attr, mpc_dir)

    # Save Data details
    if data is not None:
        save_data_class_details(data, base_dir)

    # Save SFC model and parameters
    if sfc_model is not None:
        sfc_dir = base_dir / "sfc_model"
        sfc_dir.mkdir(exist_ok=True)
        save_object_details(sfc_model, "sfc_model", sfc_dir)

        sfc_attrs = [
            'MC_param', 'Q_g', 'T_TM', 'mf_PM', 'mf_TM',
            'z_TM_center', 'z_TM_outer', 'set_size',
            'L', 'T_TM_in', 'T_env', 'T_outer', 'dt', 'dz',
            'liquid_slugs', 'output', 'param'
        ]
        for attr in sfc_attrs:
            if hasattr(sfc_model, attr):
                save_object_details(getattr(sfc_model, attr), attr, sfc_dir)

    if param_sfc_model is not None:
        save_object_details(param_sfc_model, "param_sfc_model", base_dir)

    # Save figure
    if figure is not None:
        figures_dir = base_dir / "figures"
        figures_dir.mkdir(exist_ok=True)
        figure_path = figures_dir / f"{figure_name}.png"
        figure.savefig(figure_path, bbox_inches='tight', dpi=300)
        figure_path_pdf = figures_dir / f"{figure_name}.svg"
        figure.savefig(figure_path_pdf, bbox_inches='tight')

    # create gif
    if mpc_data is not None:
        gif_dir = base_dir / "figures"
        gif_dir.mkdir(exist_ok=True)


        def add_line_manual(mpc_graphics, var_type, var_name, column, axis, **pltkwargs):
            if var_type == '_u':
                pltkwargs.update(drawstyle='steps-post')

            mpc_graphics.result_lines[var_type, var_name, column] = axis.plot(mpc_graphics.data['_time'],
                                                                              mpc_graphics.data[var_type, var_name][:,
                                                                              column],
                                                                              **pltkwargs)

            if mpc_graphics.data.dtype == 'MPC' and mpc_graphics.data.meta_data['store_full_solution']:
                y_data = mpc_graphics.data.prediction((var_type, var_name))[column]
                x_data = np.zeros(y_data.shape[0])

                color = mpc_graphics.result_lines[var_type, var_name, column][0].get_color()
                pltkwargs.update(color=color, linestyle='--')
                mpc_graphics.pred_lines[var_type, var_name, column] = axis.plot(x_data, y_data, **pltkwargs)

            mpc_graphics.ax_list.append(axis)

            return mpc_graphics

        # Graphics for mpc
        mpc_graphics = do_mpc.graphics.Graphics(mpc_data)

        fig, ax = plt.subplots(9, sharex=True, figsize=(10, 20))

        mpc_graphics = add_line_manual(mpc_graphics, var_type='_x', var_name='x_k-0', column=0, axis=ax[0])
        mpc_graphics = add_line_manual(mpc_graphics, var_type='_x', var_name='x_k-0', column=1, axis=ax[1])
        mpc_graphics = add_line_manual(mpc_graphics, var_type='_x', var_name='x_k-0', column=2, axis=ax[2])
        mpc_graphics = add_line_manual(mpc_graphics, var_type='_x', var_name='x_k-0', column=3, axis=ax[3])
        mpc_graphics = add_line_manual(mpc_graphics, var_type='_x', var_name='x_k-0', column=4, axis=ax[4])
        mpc_graphics = add_line_manual(mpc_graphics, var_type='_x', var_name='x_k-0', column=5, axis=ax[5])
        # mpc_graphics.add_line(var_type='_tvp', var_name='set_size', axis=ax[4])
        mpc_graphics.add_line(var_type='_u', var_name='mf_PM', axis=ax[6])
        mpc_graphics.add_line(var_type='_u', var_name='mf_TM', axis=ax[7])
        mpc_graphics.add_line(var_type='_u', var_name='Q_g', axis=ax[8])

        # Update properties for all prediction lines:
        for line_i in mpc_graphics.pred_lines.full:
            line_i.set_linewidth(2)

        def update(t_ind):
            print('Writing frame: {}.'.format(t_ind), end='\r')
            mpc_graphics.plot_results(t_ind=t_ind)
            mpc_graphics.plot_predictions(t_ind=t_ind)
            mpc_graphics.reset_axes()
            lines = mpc_graphics.result_lines.full
            return lines

        n_steps = mpc_data['_time'].shape[0]
        # n_steps = 150
        anim = FuncAnimation(fig, update, frames=n_steps, blit=True)

        writer = ImageMagickWriter(fps=2)
        gif_path = gif_dir / f"{figure_name}.gif"
        anim.save(gif_path, writer=writer)
        plt.close()
