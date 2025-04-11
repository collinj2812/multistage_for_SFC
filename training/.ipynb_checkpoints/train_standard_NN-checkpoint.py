import sys
import matplotlib.pyplot as plt
import casadi as ca

import numpy as np
import torch
import torch.utils.data

sys.path.append('./helper functions')

import save
from data_class import DataClass
from simulation_evaluation import simulation_evaluation

import pickle

def torch_to_casadi_function(model: torch.nn.Module, data, activation: str = "sigmoid"):
    """
    Converts a PyTorch model to a CasADi function for evaluation.

    Args:
        model (torch.nn.Module): The PyTorch model to convert.
        data: DataClass object containing data for scaling.
        activation (str): The activation function to use ("sigmoid" or "relu"). Default is "sigmoid".

    Returns:
        casadi.Function: A CasADi function that evaluates the PyTorch model.
    """

    # Automatically extract input size from the first Linear layer
    layers = list(model.children())

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
        weight = layer.weight.detach().numpy()#.detach().cpu().numpy()
        bias = layer.bias.detach().numpy()#.detach().cpu().numpy()

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
                input_casadi = 0.5 * input_casadi * (1 + ca.tanh(ca.sqrt(2 / ca.pi) * (input_casadi + 0.044715 * input_casadi ** 3)))
            else:
                raise ValueError("Unsupported activation function. Use 'sigmoid', 'relu', 'gelu', or 'tanh'.")

    # Scale output
    # for standard scaler
    input_casadi = input_casadi * data.scaler_Y.scale_ + data.scaler_Y.mean_

    # Create the CasADi function
    model_casadi_function = ca.Function('model', [x], [input_casadi])

    return model_casadi_function

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
    def __init__(self, data: DataClass, metadata_data):
        self.casadi_model = None
        self.best_model_weights = None
        self.val_losses = None
        self.train_losses = None
        self.optimizer = None
        self.criterion = None
        self.torch_model = None
        self.val_loader = None
        self.train_loader = None
        self.learning_rate = None
        self.n_epochs = None
        self.batch_size = None
        self.n_neurons = None
        self.data = data
        self.metadata_data = metadata_data

        # load data from DataClass object and convert to torch tensors
        self.X_train = torch.tensor(data.X_train, dtype=torch.float32)
        self.Y_train = torch.tensor(data.Y_train, dtype=torch.float32)

        self.X_val = torch.tensor(data.X_val, dtype=torch.float32)
        self.Y_val = torch.tensor(data.Y_val, dtype=torch.float32)

    def setup(self, n_neurons, batch_size = 32, n_epochs = 1000, learning_rate = 0.001, weight_decay = 0.0, patience = 10, lr_scheduler_factor = 0.5):
        # training parameters
        self.n_neurons = n_neurons
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.lr_scheduler_factor = lr_scheduler_factor

        # create dataset
        train_dataset = TimeSeriesDataset(self.X_train, self.Y_train)
        val_dataset = TimeSeriesDataset(self.X_val, self.Y_val)

        # data loader
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # model
        self.torch_model = TorchModel(self.X_train.shape[1], self.Y_train.shape[1], self.n_neurons)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.torch_model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=self.lr_scheduler_factor, patience=self.patience, verbose=True)


    def train(self):
        # lists for losses
        self.train_losses = []
        self.val_losses = []

        # store best model weights
        best_loss = np.inf
        self.best_model_weights = self.torch_model.state_dict()

        # training loop
        for epoch in range(self.n_epochs):
            self.torch_model.train()
            train_loss = 0.0
            for inputs, targets in self.train_loader:
                # forward pass
                outputs = self.torch_model(inputs)
                loss = self.criterion(outputs, targets)

                # backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            # average train loss
            train_loss /= len(self.train_loader)
            self.train_losses.append(train_loss)

            # validation loss
            self.torch_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in self.val_loader:
                    outputs = self.torch_model(inputs)
                    loss = self.criterion(outputs, targets)
                    val_loss += loss.item()

            # average validation loss
            val_loss /= len(self.val_loader)
            self.val_losses.append(val_loss)

            # learning rate scheduler
            self.scheduler.step(val_loss)

            # store best model weights
            if val_loss < best_loss:
                best_loss = val_loss
                self.best_model_weights = self.torch_model.state_dict()

            # print losses
            if epoch % 10 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f'Epoch {epoch+1}/{self.n_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}, LR: {current_lr}')

        # load best model weights
        self.torch_model.load_state_dict(self.best_model_weights)

        # print losses of best model
        print(f'Best model: Train Loss: {train_loss}, Val Loss: {val_loss}')

        # plot losses
        plt.figure()
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        # plt.show()

    def get_casadi_model(self):
        self.casadi_model = torch_to_casadi_function(self.torch_model, self.data, activation='gelu')


    def predict(self, X):
        self.torch_model.eval()
        with torch.no_grad():
            return self.torch_model(X).numpy()

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
        load_name = '../models/NN_model_sfc_model_largerbounds.pkl'
        with open(load_name, 'rb') as f:
            model = pickle.load(f)

        # load paramters
        keys_states = model.data.keys_states
        keys_inputs = model.data.keys_inputs

        l = model.data.l

    else:
        application = False

        # setup data: choose features and target (states and inputs), scale data, NARX, split data
        if application:
            # load data
            load_name = '../sfc_model_application_test.xlsx'
            data_dict, metadata = save.read_excel(load_name)
            keys_states = ['T_PM', 'T_TM']
            keys_inputs = ['mf_TM', 'T_PM_in']
            l = 6
        else:
            # load data
            load_name = '../sfc_model_largerbounds.xlsx'
            data_dict, metadata = save.read_excel(load_name)
            keys_states = ['T_PM', 'T_TM', 'c', 'd10', 'd50', 'd90']
            keys_inputs = ['mf_PM', 'mf_TM', 'Q_g', 'w_crystal']
            l = 4


        data = DataClass(data_dict, keys_states, keys_inputs,l)

        # approximate
        model = Approximate(data, metadata)
        # model.setup(n_neurons=20, n_epochs=500, learning_rate=0.0001)
        model.setup(n_neurons=30, n_epochs=2000, learning_rate=0.01, weight_decay=1e-4, patience=50, lr_scheduler_factor=0.5)
        model.train()
        model.get_casadi_model()
        if save_model:
            if application:
                model.save('../models/NN_model_sfc_model_application.pkl')
            else:
                model.save('../models/NN_model_sfc_model_largerbounds.pkl')

    # load test data
    load_name = '../sfc_model_largerbounds_test.xlsx'
    data_test_dict, metadata_test = save.read_excel(load_name)

    data_test = DataClass(data_test_dict, keys_states, keys_inputs, l)

    # plot_bound = [400,1400]
    plot_bound = None
    simulation_evaluation(data_test.states, data_test.inputs, model, states=len(keys_states), save=True, plot_bound=plot_bound, data_test=data_test)

    sys.exit()
