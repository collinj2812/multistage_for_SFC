import numpy as np
import sklearn.preprocessing
import sklearn.model_selection

class DataClass:
    def __init__(self, data, keys_states, keys_inputs, l, split_in_three=False, remove_first=10):
        self.raw_data = data
        self.keys_states = keys_states
        self.keys_inputs = keys_inputs
        self.l = l  # lag parameter for NARX
        self.split_in_three = split_in_three
        self.remove_first = remove_first

        # extract states and inputs
        self.extract_states_inputs()

        # NARX
        self.narx()

        # scale data
        self.scale_data()

        # split data
        self.split_data()

    def extract_states_inputs(self):
        self.states = np.array([self.raw_data[key] for key in self.keys_states]).T
        self.inputs = np.array([self.raw_data[key] for key in self.keys_inputs]).T

        self.states = self.states[self.remove_first:]
        self.inputs = self.inputs[self.remove_first:]



    def narx(self):
        # construct data matrix
        data_matrix = np.hstack((self.states, self.inputs))

        # number of time steps
        n_time_steps = data_matrix.shape[0]

        # number of states
        n_states = data_matrix.shape[1]

        # construct X and Y
        X = []
        Y = []

        for t in range(n_time_steps - self.l):
            X_i = []
            U_i = []
            for i in reversed(range(self.l)):
                X_i.append(self.states[t + i].reshape(1,-1))
                U_i.append(self.inputs[t + i].reshape(1,-1))
            X.append(np.hstack((*X_i, *U_i)))
            Y.append(self.states[t + self.l].reshape(1,-1))

        self.X = np.vstack(X)
        self.Y = np.vstack(Y)

    def scale_data(self):
        # scale self.X and self.Y
        self.scaler_X = sklearn.preprocessing.StandardScaler()
        self.scaler_Y = sklearn.preprocessing.StandardScaler()

        self.X_scaled = self.scaler_X.fit_transform(self.X)
        self.Y_scaled = self.scaler_Y.fit_transform(self.Y)

    def split_data(self):
        # split data into training and validation
        if self.split_in_three:
            # apply sklearn split twice
            self.X_train, self.X_temp, self.Y_train, self.Y_temp = sklearn.model_selection.train_test_split(self.X_scaled, self.Y_scaled, test_size=0.4, random_state=42)
            self.X_val, self.X_cal, self.Y_val, self.Y_cal = sklearn.model_selection.train_test_split(self.X_temp, self.Y_temp, test_size=0.75, random_state=42)
        else:
            self.X_train, self.X_val, self.Y_train, self.Y_val = sklearn.model_selection.train_test_split(self.X_scaled, self.Y_scaled, test_size=0.1, random_state=42)
