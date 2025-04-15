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


def save_scaler_details(scaler, name, base_path):
    """Save details of a StandardScaler object."""
    scaler_dir = base_path / name
    scaler_dir.mkdir(exist_ok=True)

    attributes = [
        'copy', 'mean_', 'n_features_in_', 'n_samples_seen_',
        'scale_', 'var_', 'with_mean', 'with_std'
    ]

    for attr in attributes:
        if hasattr(scaler, attr):
            save_object_details(getattr(scaler, attr), attr, scaler_dir)


def save_data_class_details(data, base_path):
    """Save all components of the DataClass object."""
    data_dir = base_path / "data"
    data_dir.mkdir(exist_ok=True)

    # Save main arrays
    arrays = [
        'X', 'X_scaled', 'X_train', 'X_val',
        'Y', 'Y_scaled', 'Y_train', 'Y_val',
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

    # Save raw_data DataFrame
    if hasattr(data, 'raw_data'):
        save_object_details(data.raw_data, 'raw_data', data_dir)

    # Save scalers
    if hasattr(data, 'scaler_X'):
        save_scaler_details(data.scaler_X, 'scaler_X', data_dir)
    if hasattr(data, 'scaler_Y'):
        save_scaler_details(data.scaler_Y, 'scaler_Y', data_dir)

    # Save other attributes
    other_attrs = ['l', 'remove_first', 'split_in_three']
    for attr in other_attrs:
        if hasattr(data, attr):
            save_object_details(getattr(data, attr), attr, data_dir)


def save_results(mpc_nn=None, nn_model=None, sfc_model=None, data=None, figure=None, figure_name="plot",
                 save_dir="results"):
    """
    Main function to save all results and model details.

    Parameters:
    -----------
    mpc_nn : MPC_NN object, optional
        The MPC neural network object
    nn_model : Neural Network model object, optional
        The trained neural network model
    sfc_model : SFC model object, optional
        The SFC model object
    data : DataClass object, optional
        The data class object containing all data arrays and scalers
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

    # Save MPC_NN details
    if mpc_nn is not None:
        mpc_dir = base_dir / "mpc_nn"
        mpc_dir.mkdir(exist_ok=True)
        save_object_details(mpc_nn, "mpc_nn", mpc_dir)
        if hasattr(mpc_nn, 'controller'):
            save_object_details(mpc_nn.controller, "controller", mpc_dir)
        if hasattr(mpc_nn, 'bounds'):
            save_object_details(mpc_nn.bounds, "bounds", mpc_dir)

    # Save Neural Network model details
    if nn_model is not None:
        nn_dir = base_dir / "nn_model"
        nn_dir.mkdir(exist_ok=True)
        save_object_details(nn_model, "nn_model", nn_dir)

        for attr in ['X_train', 'Y_train', 'X_val', 'Y_val',
                     'best_model_weights', 'train_losses', 'val_losses']:
            if hasattr(nn_model, attr):
                save_object_details(getattr(nn_model, attr), attr, nn_dir)

    # Save SFC model details
    if sfc_model is not None:
        sfc_dir = base_dir / "sfc_model"
        sfc_dir.mkdir(exist_ok=True)
        save_object_details(sfc_model, "sfc_model", sfc_dir)

        if hasattr(sfc_model, 'MC_param'):
            save_object_details(sfc_model.MC_param, "MC_param", sfc_dir)

        for attr in ['T_TM', 'Q_g', 'mf_PM', 'mf_TM', 'z_TM_center', 'z_TM_outer']:
            if hasattr(sfc_model, attr):
                save_object_details(getattr(sfc_model, attr), attr, sfc_dir)

    # Save DataClass details
    if data is not None:
        save_data_class_details(data, base_dir)

    # Save figure
    if figure is not None:
        figures_dir = base_dir / "figures"
        figures_dir.mkdir(exist_ok=True)
        figure_path = figures_dir / f"{figure_name}.png"
        figure.savefig(figure_path, bbox_inches='tight', dpi=300)

        # Also save in PDF format for vector graphics
        figure_path_pdf = figures_dir / f"{figure_name}.svg"
        figure.savefig(figure_path_pdf, bbox_inches='tight')
