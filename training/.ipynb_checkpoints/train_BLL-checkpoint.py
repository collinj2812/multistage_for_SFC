import sys
import casadi as ca

sys.path.append('./bll')
import bayesianlastlayer as bll
import tools

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

import save
from data_class import DataClass
from simulation_evaluation import simulation_evaluation

import pickle


def bll_to_casadi_function(model, data, activation: str = "sigmoid"):
    # check if any scaler has zeros in var (if so, it will cause division by zero) --> set to 1
    if 0 in data.scaler_X.var_:
        data.scaler_X.var_[data.scaler_X.var_ == 0] = 1
    if 0 in data.scaler_Y.var_:
        data.scaler_Y.var_[data.scaler_Y.var_ == 0] = 1
    if 0 in model.bll_model.scaler.scaler_x.var_:
        model.bll_model.scaler.scaler_x.var_[model.bll_model.scaler.scaler_x.var_ == 0] = 1
    if 0 in model.bll_model.scaler.scaler_y.var_:
        model.bll_model.scaler.scaler_y.var_[model.bll_model.scaler.scaler_y.var_ == 0] = 1

    # Automatically extract input size from the first Linear layer
    layers = model.bll_model.joint_model.layers

    # Identify and filter out only the Linear layers (ignoring any final activation function)
    linear_layers = [layer for layer in model.bll_model.joint_model.layers if isinstance(layer, keras.layers.Dense)]

    # Input size is taken from the first Linear layer
    input_size = data.X.shape[1]

    # CasADi symbolic input
    x = ca.SX.sym('x', input_size)

    # Set up CasADi symbolic variables
    input_casadi = x

    # Scale input
    # for standard scaler
    input_casadi = (input_casadi - data.scaler_X.mean_) / data.scaler_X.scale_

    # implementation of bll requires second scaler
    input_casadi = (input_casadi - model.bll_model.scaler.scaler_x.mean_) / model.bll_model.scaler.scaler_x.scale_

    # Iterate through the Linear layers of the PyTorch model
    for i, layer in enumerate(linear_layers):
        # Extract weights and bias from Keras Dense layer
        weight = layer.get_weights()[0]
        bias = layer.get_weights()[1]

        # Perform the linear transformation: y = Wx + b
        input_casadi = ca.mtimes(weight.T, input_casadi) + bias

        # Apply activation function unless it's the last Linear layer
        if i < len(linear_layers) - 1:
            if activation == "relu":
                input_casadi = ca.fmax(input_casadi, 0)  # ReLU activation
            elif activation == "sigmoid":
                input_casadi = 1 / (1 + ca.exp(-input_casadi))  # Sigmoid activation
            elif activation == "tanh":
                input_casadi = ca.tanh(input_casadi)
            elif activation == "gelu":
                input_casadi = 0.5 * input_casadi * (
                            1 + ca.tanh(ca.sqrt(2 / ca.pi) * (input_casadi + 0.044715 * input_casadi ** 3)))
            else:
                raise ValueError("Unsupported activation function. Use 'sigmoid', 'relu', 'gelu', or 'tanh'.")

        # in second to last loop:
        if i == len(linear_layers) - 2:
            phi_tilde = input_casadi

    # bll requires second scaler
    input_casadi = input_casadi * model.bll_model.scaler.scaler_y.scale_ + model.bll_model.scaler.scaler_y.mean_

    # Scale output
    # for standard scaler
    input_casadi = input_casadi * data.scaler_Y.scale_ + data.scaler_Y.mean_

    # Create the CasADi function
    model_casadi_function = ca.Function('model', [x], [input_casadi])

    # calculation of covariance matrix
    phi = ca.vertcat(phi_tilde, ca.SX(1)).T

    sigma_e2 = np.exp(2 * model.bll_model.log_sigma_e.numpy())

    cov_0 = phi @ model.bll_model.Sigma_p_bar @ phi.T

    # initialize casadi matrices
    cov_without_noise = ca.SX.zeros(sigma_e2.shape[0], sigma_e2.shape[0])
    cov_with_noise = ca.SX.zeros(sigma_e2.shape[0], sigma_e2.shape[0])

    for i, sigma_e2_i in enumerate(sigma_e2):
        cov_i_scaled = sigma_e2_i * cov_0
        cov_with_noise_i_scaled = cov_i_scaled + sigma_e2_i

        # unscale
        cov_with_noise_i = cov_with_noise_i_scaled * model.bll_model.scaler.scaler_y.scale_[i] ** 2
        cov_without_noise_i = cov_i_scaled * model.bll_model.scaler.scaler_y.scale_[i] ** 2

        # unscale
        cov_with_noise_i = cov_with_noise_i * data.scaler_Y.scale_[i] ** 2
        cov_without_noise_i = cov_without_noise_i * data.scaler_Y.scale_[i] ** 2

        cov_without_noise[i, i] = ca.fmax(cov_without_noise_i, 0)
        cov_with_noise[i, i] = ca.fmax(cov_with_noise_i, 0)

    std_without_noise = ca.sqrt(ca.diag(cov_without_noise))
    std_with_noise = ca.sqrt(ca.diag(cov_with_noise))

    # create casadi functions
    std_without_noise_function = ca.Function('std_without_noise', [x], [std_without_noise])
    std_with_noise_function = ca.Function('std_with_noise', [x], [std_with_noise])

    return model_casadi_function, std_with_noise_function, std_without_noise_function


class Approximate:
    def __init__(self, data: DataClass, metadata_data):
        # load data
        self.std_without_noise = None
        self.std_with_noise = None
        self.casadi_model = None
        self.Y_val_sc2 = None
        self.X_val_sc2 = None
        self.Y_train_sc2 = None
        self.X_train_sc2 = None
        self.scaler = None
        self.optimizer = None
        self.bll_model = None
        self.output_model = None
        self.joint_model = None
        self.learning_rate = None
        self.n_epochs = None
        self.n_neurons = None
        self.data = data
        self.metadata_data = metadata_data

        self.X_train = data.X_train
        self.Y_train = data.Y_train

        self.X_val = data.X_val
        self.Y_val = data.Y_val

    def setup(self, n_neurons, n_epochs=1000, learning_rate=0.001):
        # training parameters
        self.n_neurons = n_neurons
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate

        # according to BLL implementation another scaler is necessary
        self.scaler = tools.Scaler(self.X_train, self.Y_train)
        self.X_train_sc2, self.Y_train_sc2 = self.scaler.scale(self.X_train, self.Y_train)
        self.X_val_sc2, self.Y_val_sc2 = self.scaler.scale(self.X_val, self.Y_val)

        # create joint_model and output_model
        model_input = keras.Input(shape=(self.X_train.shape[1],))

        # Hidden units
        architecture = [
            (keras.layers.Dense, {'units': self.n_neurons, 'activation': 'gelu', 'name': '01_dense'}),
            # (keras.layers.Dense, {'units': self.n_neurons, 'activation': 'sigmoid', 'name': '02_dense'}),
            (keras.layers.Dense, {'name': 'output', 'units': self.Y_train.shape[1]})
        ]

        # Get layers and outputs:
        model_layers, model_outputs = tools.DNN_from_architecture(model_input, architecture)
        self.joint_model = keras.Model(model_input, [model_outputs[-2], model_outputs[-1]])
        self.output_model = keras.Model(model_input, model_outputs[-1])

        # compile model
        self.bll_model = bll.BayesianLastLayer(self.joint_model, self.scaler)
        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.bll_model.setup_training(self.optimizer)

        self.cb_early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=1e-4,
            patience=10000,
            verbose=True,
            mode="auto",
            baseline=None,
            restore_best_weights=True,
        )

    def train(self):
        self.bll_model.fit(self.X_train_sc2.astype(np.float32), self.Y_train_sc2.astype(np.float32), verbose=True,
                           val=[self.X_val_sc2.astype(np.float32), self.Y_val_sc2.astype(np.float32)],
                           epochs=self.n_epochs)

        # plot training history
        self.bll_model.log_alpha = 1000
        history = self.bll_model.training_history
        epochs = range(1, len(history['loss']) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(history['epochs'], history['loss'], 'b-', label='Training Loss')
        plt.plot(history['epochs'], history['val_loss'], 'r-', label='Validation Loss')
        plt.title('Model Loss During Training')
        plt.xlabel('Epochs')
        plt.ylabel('Loss [LML]')
        plt.legend()
        plt.grid(True)
        # plt.show()

    def get_casadi_model(self):
        self.casadi_model, self.std_with_noise, self.std_without_noise = bll_to_casadi_function(self, self.data,
                                                                                                activation='gelu')

    def predict(self):
        pass

    def save(self, save_name):
        # pickle model
        with open(save_name, 'wb') as file:
            pickle.dump(self, file)


if __name__ == '__main__':
    # save model
    save_model = False
    # load model
    load_model = True

    if load_model:
        load_name = '../models/bll_model_sfc_model_largerbounds2.pkl'
        # load_name = '../varying dataset size analysis/simulation_results/n_steps_9500/simulation_2_models/bll_model.pkl'
        with open(load_name, 'rb') as f:
            model = pickle.load(f)

        # load paramters
        keys_states = model.data.keys_states
        keys_inputs = model.data.keys_inputs

        l = model.data.l
    else:
        # load data
        load_name = '../sfc_model_largerbounds.xlsx'
        data_dict, metadata = save.read_excel(load_name)

        # setup data: choose features and target (states and inputs), scale data, NARX, split data
        keys_states = ['T_PM', 'T_TM', 'c', 'd10', 'd50', 'd90']
        keys_inputs = ['mf_PM', 'mf_TM', 'Q_g', 'w_crystal']

        l = 4

        data = DataClass(data_dict, keys_states, keys_inputs, l)

        # setup approximate
        model = Approximate(data, metadata)
        model.setup(n_neurons=30, n_epochs=2700, learning_rate=0.004)
        model.train()
        model.get_casadi_model()
        if save_model:
            model.save('../models/bll_model_sfc_model_largerbounds2.pkl')

    # bll model dict
    bll_dict = {
        'bll': model.std_with_noise
    }

    # load test data
    load_name = '../sfc_model_largerbounds_test.xlsx'
    data_test_dict, meatdata_test = save.read_excel(load_name)

    data_test = DataClass(data_test_dict, keys_states, keys_inputs, l)

    # plot_bound = [400,1400]
    plot_bound = None

    simulation_evaluation(data_test.states, data_test.inputs, model, save=True, plot_bound=plot_bound,
                          data_test=data_test, **bll_dict)

    sys.exit()
