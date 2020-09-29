"""Helper module to process simulation data."""

import numpy as np
import torch
from tqdm import tqdm

# make torch results reproducible and use double precision
torch.set_default_tensor_type(torch.DoubleTensor)
torch.manual_seed(42)
np.random.seed(42)


def training_loop(model, path, x_train, y_train, epochs, l_rate):
    """Optimize the weights of a given MLP.

    Parameters
    ----------
    model - SimpleMLP : model to optimize
    path - String : path to save best model weights
    x_train - array-like : feature vector of dimension [n_samples, n_features]
    y_train - array-like : label vector of dimension [n_samples, n_labels]
    epochs - Integer : number of epochs to train
    l_rate - Float : learning rate

    Returns
    -------
    history - List : loss developments over epochs
    model - SimpleMLP : opimized model in evaluation mode

    """
    x_tensor = torch.from_numpy(x_train.astype(np.float64))
    if len(x_train.shape) == 1:
        x_tensor = x_tensor.unsqueeze(-1)
    y_tensor = torch.from_numpy(y_train.astype(np.float64))
    if len(y_train.shape) == 1:
        y_tensor = y_tensor.unsqueeze(-1)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=l_rate)

    best_loss = 1.0E5
    train_loss = []

    for e in tqdm(range(1, epochs+1)):
        optimizer.zero_grad()
        output = model.forward(x_tensor)
        loss = criterion(output, y_tensor)
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), path)
    return model.eval(), np.asarray(train_loss)


class SimpleMLP(torch.nn.Module):
    """Implements a standard MLP with otional batch normalization.
    """

    def __init__(self, **kwargs):
        """Create a SimpleMLP object derived from torch.nn.Module.

        Parameters
        ----------
        n_inputs - Integer : number of features/inputs
        n_outputs - Integer : number of output values
        n_layers - Integer : number of hidden layers
        n_neurons - Integer : number of neurons per hidden layer
        activation - Function : nonlinearity/activation function
        batch_norm - Boolean : use batch normalization instead of bias if True

        Members
        -------
        layers - List : list with network layers and activations

        """
        super().__init__()
        self.n_inputs = kwargs.get("n_inputs", 1)
        self.n_outputs = kwargs.get("n_outputs", 1)
        self.n_layers = kwargs.get("n_layers", 1)
        self.n_neurons = kwargs.get("n_neurons", 10)
        self.activation = kwargs.get("activation", torch.sigmoid)
        self.batch_norm = kwargs.get("batch_norm", True)
        self.layers = torch.nn.ModuleList()

        if self.batch_norm:
            # input layer to first hidden layer
            self.layers.append(torch.nn.Linear(
                self.n_inputs, self.n_neurons*2, bias=False))
            self.layers.append(torch.nn.BatchNorm1d(self.n_neurons*2))
            # add more hidden layers if specified
            if self.n_layers > 2:
                for hidden in range(self.n_layers-2):
                    self.layers.append(torch.nn.Linear(
                        self.n_neurons*2, self.n_neurons*2, bias=False))
                    self.layers.append(torch.nn.BatchNorm1d(self.n_neurons*2))
            self.layers.append(torch.nn.Linear(
                self.n_neurons*2, self.n_neurons, bias=False))
            self.layers.append(torch.nn.BatchNorm1d(self.n_neurons))
        else:
            # input layer to first hidden layer
            self.layers.append(torch.nn.Linear(self.n_inputs, self.n_neurons))
            # add more hidden layers if specified
            if self.n_layers > 1:
                for hidden in range(self.n_layers-1):
                    self.layers.append(torch.nn.Linear(
                        self.n_neurons, self.n_neurons))
        # last hidden layer to output layer
        self.layers.append(torch.nn.Linear(self.n_neurons, self.n_outputs))
        # print("Created model with {} weights.".format(self.model_parameters()))

    def forward(self, x):
        """Compute forward pass through model.

        Parameters
        ----------
        x - array-like : feature vector with dimension [n_samples, n_inputs]

        Returns
        -------
        output - array-like : model output with dimension [n_samples, n_outputs]

        """
        if self.batch_norm:
            for i_layer in range(len(self.layers)-1):
                if isinstance(self.layers[i_layer], torch.nn.Linear):
                    x = self.layers[i_layer](x)
                else:
                    x = self.activation(self.layers[i_layer](x))
        else:
            for i_layer in range(len(self.layers)-1):
                x = self.activation(self.layers[i_layer](x))
        return self.layers[-1](x)

    def model_parameters(self):
        """Compute total number of trainable model parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
