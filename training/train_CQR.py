import numpy as np
import matplotlib.pyplot as plt
import sys

import save
from data_class import DataClass
from simulation_evaluation import simulation_evaluation

import torch
import torch.nn as nn

import torch
import torch.nn as nn
import casadi as ca

import pickle


def CQR_to_casadi_function(model: torch.nn.Module, data, activation: str = "sigmoid"):
    # first casadi function is standard model, second is lower bound, third is upper bound

    # standard model:
    standard_model = model.torch_model_MSE

    # Automatically extract input size from the first Linear layer
    layers = list(standard_model.children())

    # Identify and filter out only the Linear layers (ignoring any final activation function)
    linear_layers = [layer for layer in layers if isinstance(layer, torch.nn.Linear)]

    if len(linear_layers) == 0:
        raise ValueError("The model does not contain any Linear layers. Cannot create CasADi function.")

    # Input size is taken from the first Linear layer
    input_size = data.X.shape[1]

    # CasADi symbolic input
    x = ca.SX.sym('x', input_size)

    # Set up CasADi symbolic variables
    input_casadi = x

    # Scale input
    # for standard scaler
    input_casadi = (input_casadi - data.scaler_X.mean_) / data.scaler_X.scale_

    # Iterate through the Linear layers of the PyTorch model
    for i, layer in enumerate(linear_layers):
        # Extract the weights and bias from the PyTorch linear layer
        weight = layer.weight.detach().numpy()  # .detach().cpu().numpy()
        bias = layer.bias.detach().numpy()  # .detach().cpu().numpy()

        # Perform the linear transformation: y = Wx + b
        input_casadi = ca.mtimes(weight, input_casadi) + bias

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

    # get intermediate output
    intermediate_function = ca.Function('model', [x], [input_casadi])

    # Scale output
    # for standard scaler
    input_casadi = input_casadi * data.scaler_Y.scale_ + data.scaler_Y.mean_

    # Create the CasADi function
    model_casadi_function_MSE = ca.Function('model', [x], [input_casadi])

    # lower bound model:
    lower_model = model.models_PB[str(model.quantiles[0])]['torch_model']

    # Automatically extract input size from the first Linear layer
    layers = list(lower_model.children())

    # Identify and filter out only the Linear layers (ignoring any final activation function)
    linear_layers = [layer for layer in layers if isinstance(layer, torch.nn.Linear)]

    if len(linear_layers) == 0:
        raise ValueError("The model does not contain any Linear layers. Cannot create CasADi function.")

    # CasADi symbolic input
    x = ca.SX.sym('x', input_size)

    # Set up CasADi symbolic variables
    input_casadi = x

    # Scale input
    # for standard scaler
    input_casadi = (input_casadi - data.scaler_X.mean_) / data.scaler_X.scale_

    # Iterate through the Linear layers of the PyTorch model
    for i, layer in enumerate(linear_layers):
        # Extract the weights and bias from the PyTorch linear layer
        weight = layer.weight.detach().numpy()  # .detach().cpu().numpy()
        bias = layer.bias.detach().numpy()  # .detach().cpu().numpy()

        # Perform the linear transformation: y = Wx + b
        input_casadi = ca.mtimes(weight, input_casadi) + bias

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

    # add intermediate output
    input_casadi = input_casadi + intermediate_function(x)

    # conformalization step
    input_casadi = input_casadi - model.Q_alpha

    # Scale output
    # for standard scaler
    input_casadi = input_casadi * data.scaler_Y.scale_ + data.scaler_Y.mean_

    # Create the CasADi function
    model_casadi_function_lower = ca.Function('model', [x], [input_casadi])

    # upper bound model:
    upper_model = model.models_PB[str(model.quantiles[1])]['torch_model']

    # Automatically extract input size from the first Linear layer
    layers = list(upper_model.children())

    # Identify and filter out only the Linear layers (ignoring any final activation function)
    linear_layers = [layer for layer in layers if isinstance(layer, torch.nn.Linear)]

    if len(linear_layers) == 0:
        raise ValueError("The model does not contain any Linear layers. Cannot create CasADi function.")

    # CasADi symbolic input
    x = ca.SX.sym('x', input_size)

    # Set up CasADi symbolic variables
    input_casadi = x

    # Scale input
    # for standard scaler
    input_casadi = (input_casadi - data.scaler_X.mean_) / data.scaler_X.scale_

    # Iterate through the Linear layers of the PyTorch model
    for i, layer in enumerate(linear_layers):
        # Extract the weights and bias from the PyTorch linear layer
        weight = layer.weight.detach().numpy()  # .detach().cpu().numpy()
        bias = layer.bias.detach().numpy()  # .detach().cpu().numpy()

        # Perform the linear transformation: y = Wx + b
        input_casadi = ca.mtimes(weight, input_casadi) + bias

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

    # add intermediate output
    input_casadi = input_casadi + intermediate_function(x)

    # conformalization step
    input_casadi = input_casadi + model.Q_alpha

    # Scale output
    # for standard scaler
    input_casadi = input_casadi * data.scaler_Y.scale_ + data.scaler_Y.mean_

    # Create the CasADi function
    model_casadi_function_upper = ca.Function('model', [x], [input_casadi])

    return model_casadi_function_MSE, model_casadi_function_lower, model_casadi_function_upper


class PinballLoss(nn.Module):
    """
    Calculates the quantile loss function.

    Attributes
    ----------
    self.pred : torch.tensor
        Predictions.
    self.target : torch.tensor
        Target to predict.
    self.quantiles : torch.tensor
    """

    def __init__(self, quantiles):
        super(PinballLoss, self).__init__()
        self.pred = None
        self.targets = None
        self.quantiles = quantiles

    def forward(self, pred, target):
        """
        Computes the loss for the given prediction.
        """
        error = target - pred
        upper = self.quantiles * error
        lower = (self.quantiles - 1) * error

        losses = torch.max(lower, upper)
        loss = torch.mean(torch.sum(losses, dim=1))
        return loss


class TorchModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=40):
        super(TorchModel, self).__init__()

        # define layers
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        self.sigmoid = torch.nn.Sigmoid()
        self.gelu = torch.nn.GELU()

    def forward(self, x):
        x = self.gelu(self.fc1(x))
        x = self.fc2(x)
        return x


class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class Approximate:
    # same structure as for NN and BLL
    def __init__(self, data, quantiles, alpha, metadata_data):
        self.Q_alpha = None
        self.val_loader_MSE = None
        self.train_loader_MSE = None
        self.learning_rate = None
        self.n_epochs = None
        self.batch_size = None
        self.n_neurons = None
        self.quantiles = quantiles
        self.alpha = alpha
        self.data = data
        self.metadata_data = metadata_data

        # load data from DataClass object and convert to torch tensors
        self.X_train = torch.tensor(data.X_train, dtype=torch.float32)
        self.Y_train = torch.tensor(data.Y_train, dtype=torch.float32)

        self.X_val = torch.tensor(data.X_val, dtype=torch.float32)
        self.Y_val = torch.tensor(data.Y_val, dtype=torch.float32)

        self.X_cal = torch.tensor(data.X_cal, dtype=torch.float32)
        self.Y_cal = torch.tensor(data.Y_cal, dtype=torch.float32)

    def setup(self, n_neurons_mean, n_neurons_quantiles, batch_size=32, n_epochs_mean=1000, n_epochs_quantiles=1000,
              learning_rate=0.001, weight_decay_MSE=0.0, weight_decay_quantiles=0.0, patience_MSE=10,
              patience_quantiles=10, lr_scheduler_factor_MSE=0.5, lr_scheduler_factor_quantiles=0.5):
        # training parameters
        self.n_neurons_mean = n_neurons_mean
        self.n_neurons_quantiles = n_neurons_quantiles
        self.batch_size = batch_size
        self.n_epochs_mean = n_epochs_mean
        self.n_epochs_quantiles = n_epochs_quantiles
        self.learning_rate = learning_rate
        self.weight_decay_MSE = weight_decay_MSE
        self.weight_decay_quantiles = weight_decay_quantiles
        self.patience_MSE = patience_MSE
        self.patience_quantiles = patience_quantiles
        self.lr_scheduler_factor_MSE = lr_scheduler_factor_MSE
        self.lr_scheduler_factor_quantiles = lr_scheduler_factor_quantiles

        # create dataset
        train_dataset = TimeSeriesDataset(self.X_train, self.Y_train)
        val_dataset = TimeSeriesDataset(self.X_val, self.Y_val)

        # data loader
        self.train_loader_MSE = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader_MSE = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

    def train(self):
        # first train with MSELoss, then compute difference between predictions and targets, then train twice with PinballLoss

        # MSE training:
        self.torch_model_MSE = TorchModel(self.X_train.shape[1], self.Y_train.shape[1], self.n_neurons_mean)
        # self.criterion = PinballLoss(torch.tensor([0.5,0.9]))
        self.criterion_MSE = torch.nn.L1Loss()
        self.optimizer_MSE = torch.optim.Adam(self.torch_model_MSE.parameters(), lr=self.learning_rate,
                                              weight_decay=self.weight_decay_MSE)
        self.scheduler_MSE = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_MSE, mode='min',
                                                                        factor=self.lr_scheduler_factor_MSE,
                                                                        patience=self.patience_MSE, verbose=True)
        # lists for losses
        self.train_losses_MSE = []
        self.val_losses_MSE = []

        # store best model weights
        best_loss_MSE = np.inf
        self.best_model_weights_MSE = self.torch_model_MSE.state_dict()

        # training loop
        for epoch in range(self.n_epochs_mean):
            self.torch_model_MSE.train()
            train_loss = 0.0
            for inputs, targets in self.train_loader_MSE:
                # forward pass
                outputs = self.torch_model_MSE(inputs)
                loss = self.criterion_MSE(outputs, targets)

                # backward pass and optimization
                self.optimizer_MSE.zero_grad()
                loss.backward()
                self.optimizer_MSE.step()

                train_loss += loss.item()

            # average train loss
            train_loss /= len(self.train_loader_MSE)
            self.train_losses_MSE.append(train_loss)

            # validation loss
            self.torch_model_MSE.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in self.val_loader_MSE:
                    outputs = self.torch_model_MSE(inputs)
                    loss = self.criterion_MSE(outputs, targets)
                    val_loss += loss.item()

            # average validation loss
            val_loss /= len(self.val_loader_MSE)
            self.val_losses_MSE.append(val_loss)

            # learning rate scheduler
            self.scheduler_MSE.step(val_loss)

            # store best model weights
            if val_loss < best_loss_MSE:
                best_loss_MSE = val_loss
                self.best_model_weights = self.torch_model_MSE.state_dict()

            # print losses
            if epoch % 10 == 0:
                current_lr = self.optimizer_MSE.param_groups[0]['lr']
                print(
                    f'Epoch {epoch + 1}/{self.n_epochs_mean}, Train Loss: {train_loss}, Val Loss: {val_loss}, LR: {current_lr}')

        # load best model weights
        self.torch_model_MSE.load_state_dict(self.best_model_weights)

        # print losses of best model
        print(f'Best model: Train Loss: {train_loss}, Val Loss: {val_loss}')

        # plot losses
        plt.figure()
        plt.title('MSE Loss')
        plt.plot(self.train_losses_MSE, label='Train Loss')
        plt.plot(self.val_losses_MSE, label='Val Loss')
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        # plt.show()

        # create dataset for PinballLoss training
        self.X_train_PB = self.X_train
        self.Y_train_PB = self.Y_train - self.torch_model_MSE(self.X_train).detach()

        self.X_val_PB = self.X_val
        self.Y_val_PB = self.Y_val - self.torch_model_MSE(self.X_val).detach()

        train_dataset_PB = TimeSeriesDataset(self.X_train_PB, self.Y_train_PB)
        val_dataset_PB = TimeSeriesDataset(self.X_val_PB, self.Y_val_PB)

        # data loader
        self.train_loader_PB = torch.utils.data.DataLoader(train_dataset_PB, batch_size=self.batch_size, shuffle=True)
        self.val_loader_PB = torch.utils.data.DataLoader(val_dataset_PB, batch_size=self.batch_size, shuffle=False)

        # PinballLoss training:
        # output is dictionary with quantile as highest key and all model objects as values
        self.models_PB = {}
        for quantile_i in self.quantiles:
            torch_model_PB = TorchModel(self.X_train_PB.shape[1], self.Y_train_PB.shape[1], self.n_neurons_quantiles)
            criterion_PB = PinballLoss(torch.tensor([quantile_i]))
            optimizer_PB = torch.optim.Adam(torch_model_PB.parameters(), lr=self.learning_rate,
                                            weight_decay=self.weight_decay_quantiles)
            scheduler_PB = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_PB, mode='min',
                                                                      factor=self.lr_scheduler_factor_quantiles,
                                                                      patience=self.patience_quantiles, verbose=True)
            # lists for losses
            train_losses_PB = []
            val_losses_PB = []

            # store best model weights
            best_loss_PB = np.inf
            best_model_weights_PB = torch_model_PB.state_dict()

            # training loop
            for epoch in range(self.n_epochs_quantiles):
                torch_model_PB.train()
                train_loss = 0.0
                for inputs, targets in self.train_loader_PB:
                    # backward pass and optimization
                    optimizer_PB.zero_grad()

                    # forward pass
                    outputs = torch_model_PB(inputs)
                    loss = criterion_PB(outputs, targets)

                    loss.backward()
                    optimizer_PB.step()

                    train_loss += loss.item()

                # average train loss
                train_loss /= len(self.train_loader_PB)
                train_losses_PB.append(train_loss)

                # validation loss
                torch_model_PB.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for inputs, targets in self.val_loader_PB:
                        outputs = torch_model_PB(inputs)
                        loss = criterion_PB(outputs, targets)
                        val_loss += loss.item()

                # average validation loss
                val_loss /= len(self.val_loader_PB)
                val_losses_PB.append(val_loss)

                # learning rate scheduler
                scheduler_PB.step(val_loss)

                # store best model weights
                if val_loss < best_loss_PB:
                    best_loss_PB = val_loss
                    best_model_weights_PB = torch_model_PB.state_dict()

                # print losses
                if epoch % 10 == 0:
                    current_lr = optimizer_PB.param_groups[0]['lr']
                    print(
                        f'Epoch {epoch + 1}/{self.n_epochs_quantiles}, Train Loss: {train_loss}, Val Loss: {val_loss}, LR: {current_lr}')

            # load best model weights
            torch_model_PB.load_state_dict(best_model_weights_PB)

            # print losses of best model
            print(f'Best model: Train Loss: {train_loss}, Val Loss: {val_loss}')

            # plot losses
            plt.figure()
            plt.title(f'Pinball Loss {quantile_i}')
            plt.plot(train_losses_PB, label='Train Loss')
            plt.plot(val_losses_PB, label='Val Loss')
            plt.yscale('log')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            # plt.show()

            # construct sub-dictionary for quantile, save everything that was written in self for MSE model
            dict_i = {
                'torch_model': torch_model_PB,
                'criterion': criterion_PB,
                'optimizer': optimizer_PB,
                'train_losses': train_losses_PB,
                'val_losses': val_losses_PB,
                'best_model_weights': best_model_weights_PB
            }

            self.models_PB[str(quantile_i)] = dict_i

        # conformalization step
        # go through calibration data and calculate residuals
        E_list = []
        for x, y in zip(self.X_cal, self.Y_cal):
            E_i = ca.fmax(
                self.torch_model_MSE(x).detach().numpy() + self.models_PB[str(self.quantiles[0])]['torch_model'](
                    x).detach().numpy() - y.numpy(), y.numpy() - (
                        self.torch_model_MSE(x).detach().numpy() + self.models_PB[str(self.quantiles[1])][
                    'torch_model'](x).detach().numpy()))
            E_list.append(E_i)
        E_array = np.array(E_list)

        self.Q_alpha = np.quantile(E_array, (1 - self.alpha) * (1 + 1 / len(E_array)), axis=0)

    def predict(self, X):
        y_pred = []
        self.torch_model_MSE.eval()
        with torch.no_grad():
            y_pred.append(self.torch_model_MSE(X).numpy())

        # predict quantiles
        for quantile_i in self.quantiles:
            self.models_PB[str(quantile_i)]['torch_model'].eval()
            with torch.no_grad():
                y_pred_i = self.models_PB[str(quantile_i)]['torch_model'](X).numpy()
            y_pred.append(y_pred_i)

        return y_pred

    def get_casadi_model(self):
        self.casadi_model, self.casadi_model_lower, self.casadi_model_upper = CQR_to_casadi_function(self, self.data,
                                                                                                     activation='gelu')

    def save(self, save_name):
        # pickle model
        with open(save_name, 'wb') as f:
            pickle.dump(self, f)


if __name__ == '__main__':
    # save model
    save_model = False
    # load model
    load_model = True
    if load_model:
        load_name = '../models/cqr_model_sfc_model_largerbounds.pkl'
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

        data = DataClass(data_dict, keys_states, keys_inputs, l, split_in_three=True)

        alpha = 0.05  # miscoverage rate
        quantiles = [alpha / 2, 1 - alpha / 2]
        print(f'Train CQR model with alpha = {alpha}, quantiles = {quantiles}')

        # approximate
        model = Approximate(data, quantiles, alpha, metadata)
        model.setup(n_neurons_mean=30, n_neurons_quantiles=10, batch_size=32, n_epochs_mean=200,
                    n_epochs_quantiles=200, learning_rate=0.001, weight_decay_MSE=2e-5, weight_decay_quantiles=2e-5,
                    patience_MSE=10, patience_quantiles=10, lr_scheduler_factor_MSE=0.5,
                    lr_scheduler_factor_quantiles=0.5)
        model.train()
        model.get_casadi_model()
        if save_model:
            model.save('../models/cqr_model_sfc_model_largerbounds.pkl')

    # cqr model dict
    cqr_dict = {
        'cqr': None,
    }

    # load test data
    load_name = '../sfc_model_largerbounds_test.xlsx'
    data_test_dict, metadata_test = save.read_excel(load_name)

    data_test = DataClass(data_test_dict, keys_states, keys_inputs, l)

    # plot_bound = [200, 1800]
    plot_bound = None
    simulation_evaluation(data_test.states, data_test.inputs, model, save=True, plot_bound=plot_bound, data_test=data_test, **cqr_dict)

    sys.exit()
