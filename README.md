# Multi-stage Model Predictive Control for Slug Flow Crystallizers

This repository contains the implementation of the methods and results presented in the paper:

> **Multi-stage model predictive control for slug flow crystallizers using uncertainty-aware surrogate models**  
> Collin R. Johnson, Stijn de Vries, Kerstin Wohlgemuth and Sergio Lucia

The training data used for the results in the paper can be found here:
https://osf.io/a37bm/files/osfstorage#

## Overview

This project demonstrates a novel approach for controlling slug flow crystallizers using multi-stage model predictive control (MPC) with uncertainty-aware surrogate models. The repository includes:

1. A novel first-principles model for simulating slug flow crystallizers
2. Training and validation of data-based surrogate models
3. Implementation of MPC using these surrogate models

## Repository Structure

```
├── data_based_models/         # Trained surrogate models
│   ├── BLL_model.pkl          # Bayesian Last Layer model
│   ├── CQR_model.pkl          # Conformalized Quantile Regression model
│   └── NN_model.pkl           # Standard Neural Network model
├── helper_functions/          # Utility functions
│   ├── data_class.py          # Data handling classes
│   ├── save.py                # Functions for saving model results
│   └── simulation_evaluation.py # Functions for evaluating simulations
├── model/                     # First-principles model implementation
│   ├── config.py              # Model configuration
│   ├── pressure_drop.py       # Pressure drop calculations
│   └── sfc_model.py           # Slug Flow Crystallizer model
├── mpc/                       # Model Predictive Control implementation
│   ├── bll/                   # BLL model for MPC
│   │   ├── data_based_do_mpc_model_bll.py    # BLL model integration in do_mpc
│   │   ├── mpc_bll_class.py                  # BLL-specific MPC implementation
│   │   └── save_bll_results.py               # Saving results from BLL MPC
│   ├── cqr/                   # CQR model for MPC
│   │   ├── data_based_do_mpc_model_cqr.py    # CQR model integration in do_mpc
│   │   ├── mpc_cqr_class.py                  # CQR-specific MPC implementation
│   │   └── save_cqr_results.py               # Saving results from CQR MPC
│   ├── nn/                    # NN model for MPC
│   │   ├── data_based_do_mpc_model_nn.py     # NN model integration in do_mpc
│   │   ├── mpc_nn_class.py                   # NN-specific MPC implementation
│   │   └── save_nn_results.py                # Saving results from NN MPC
│   ├── config_mpc.py          # MPC configuration
│   └── prepare_model.py       # Prepare models for MPC
├── training/                  # Model training scripts
│   ├── train_BLL.py           # BLL model training
│   ├── train_CQR.py           # CQR model training
│   └── train_standard_NN.py   # Standard NN model training
├── training_data/             # Data for model training
│   ├── test.xlsx              # Test dataset
│   └── training.xlsx          # Training dataset
├── bll/                       # Bayesian Last Layer (BLL) implementation from https://github.com/4flixt/2023_paper_bll_lml
├── LICENSE.txt                # License file
├── README.md                  # This README file
├── run_mpc.ipynb              # Jupyter notebook for running MPC
├── simulate_model.ipynb       # Jupyter notebook for model simulation
└── train_models.ipynb         # Jupyter notebook for training models
```

## Key Components

### Jupyter Notebooks

This repository provides three main Jupyter notebooks to reproduce the results from the paper:

1. **`simulate_model.ipynb`**: Demonstrates how to simulate the novel first-principles model for slug flow crystallizers
2. **`train_models.ipynb`**: Shows the training process for data-based surrogate models (Standard Neural Network, Conformalized Quantile Regression, and Bayesian Last Layer)
3. **`run_mpc.ipynb`**: Illustrates how to use the trained surrogate models as internal models for Model Predictive Control

### Models

Three types of surrogate models are implemented:

1. **Standard Neural Network (NN)**: A deterministic model that provides point predictions
2. **Conformalized Quantile Regression (CQR)**: Provides uncertainty quantification through conformalized quantile regression
3. **Bayesian Last Layer (BLL)**: Provides uncertainty quantification by providing a Gaussian distributed output

## Usage

### Prerequisites

This project requires Python with the following packages:
- numpy
- pandas
- do-mpc
- torch
- keras
- matplotlib
- jupyter
- scikit-learn
- pickle
- tqdm


## License

See the LICENSE.txt file for license rights and limitations.
