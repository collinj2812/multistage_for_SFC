import numpy as np
import torch
from pathlib import Path
from datetime import datetime


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


def save_data_class_details(data, base_path):
    """Save all components of the DataClass object."""
    data_dir = base_path / "data"
    data_dir.mkdir(exist_ok=True)

    # Save main arrays
    arrays = [
        'X', 'X_cal', 'X_scaled', 'X_temp', 'X_train', 'X_val',
        'Y', 'Y_cal', 'Y_scaled', 'Y_temp', 'Y_train', 'Y_val',
        'inputs', 'states'
    ]
    for array_name in arrays:
        if hasattr(data, array_name):
            save_object_details(getattr(data, array_name), array_name, data_dir)

    # Save lists
    lists = ['keys_inputs', 'keys_states', 'databased_pred']
    for list_name in lists:
        if hasattr(data, list_name):
            save_object_details(getattr(data, list_name), list_name, data_dir)

    # Save scalers
    if hasattr(data, 'scaler_X'):
        save_object_details(data.scaler_X, 'scaler_X', data_dir)
    if hasattr(data, 'scaler_Y'):
        save_object_details(data.scaler_Y, 'scaler_Y', data_dir)

    # Save other attributes
    other_attrs = ['l', 'remove_first', 'split_in_three']
    for attr in other_attrs:
        if hasattr(data, attr):
            save_object_details(getattr(data, attr), attr, data_dir)


def save_results(cqr_model=None, mpc_cqr=None, data=None, param_sfc_model=None, sfc_model=None, figure=None,
                 figure_name="plot", save_dir="results"):
    """
    Main function to save all results and model details.

    Parameters:
    -----------
    cqr_model : CQR model object, optional
        The Conformal Quantile Regression model
    mpc_cqr : MPC CQR object, optional
        The MPC controller with CQR
    data : DataClass object, optional
        The data class object containing all data arrays and scalers
    param_sfc_model : dict, optional
        Parameters for the SFC model
    sfc_model : SFC model object, optional
        The SFC model object
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

    # Save CQR model details
    if cqr_model is not None:
        cqr_dir = base_dir / "cqr_model"
        cqr_dir.mkdir(exist_ok=True)
        save_object_details(cqr_model, "cqr_model", cqr_dir)

        # Save CQR specific attributes
        cqr_attrs = [
            'Q_alpha', 'X_cal', 'X_train', 'X_train_PB', 'X_val', 'X_val_PB',
            'Y_cal', 'Y_train', 'Y_train_PB', 'Y_val', 'Y_val_PB',
            'best_model_weights', 'best_model_weights_MSE',
            'train_losses_MSE', 'val_losses_MSE'
        ]
        for attr in cqr_attrs:
            if hasattr(cqr_model, attr):
                save_object_details(getattr(cqr_model, attr), attr, cqr_dir)

    # Save MPC CQR details
    if mpc_cqr is not None:
        mpc_dir = base_dir / "mpc_cqr"
        mpc_dir.mkdir(exist_ok=True)
        save_object_details(mpc_cqr, "mpc_cqr", mpc_dir)

        # Save MPC specific attributes
        if hasattr(mpc_cqr, 'bounds'):
            save_object_details(mpc_cqr.bounds, "bounds", mpc_dir)
        if hasattr(mpc_cqr, 'controller'):
            save_object_details(mpc_cqr.controller, "controller", mpc_dir)

    # Save Data details
    if data is not None:
        save_data_class_details(data, base_dir)

    # Save SFC model parameters
    if param_sfc_model is not None:
        param_dir = base_dir / "param_sfc_model"
        param_dir.mkdir(exist_ok=True)
        save_object_details(param_sfc_model, "param_sfc_model", param_dir)

    # Save SFC model
    if sfc_model is not None:
        sfc_dir = base_dir / "sfc_model"
        sfc_dir.mkdir(exist_ok=True)
        save_object_details(sfc_model, "sfc_model", sfc_dir)

        # Save specific arrays and parameters
        sfc_attrs = ['MC_param', 'Q_g', 'T_TM', 'mf_PM', 'mf_TM',
                     'z_TM_center', 'z_TM_outer', 'set_size']
        for attr in sfc_attrs:
            if hasattr(sfc_model, attr):
                save_object_details(getattr(sfc_model, attr), attr, sfc_dir)

    # Save figure
    if figure is not None:
        figures_dir = base_dir / "figures"
        figures_dir.mkdir(exist_ok=True)
        figure_path = figures_dir / f"{figure_name}.png"
        figure.savefig(figure_path, bbox_inches='tight', dpi=300)

        # Also save in PDF format for vector graphics
        figure_path_pdf = figures_dir / f"{figure_name}.svg"
        figure.savefig(figure_path_pdf, bbox_inches='tight')
