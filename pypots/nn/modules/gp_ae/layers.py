"""

"""

# Created by Jun Wang <jwangfx@connect.ust.hk> and Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def rbf_kernel(T, length_scale):
    xs = torch.arange(T).float()
    xs_in = torch.unsqueeze(xs, 0)
    xs_out = torch.unsqueeze(xs, 1)
    distance_matrix = (xs_in - xs_out) ** 2
    distance_matrix_scaled = distance_matrix / length_scale**2
    kernel_matrix = torch.exp(-distance_matrix_scaled)
    return kernel_matrix


def diffusion_kernel(T, length_scale):
    assert length_scale < 0.5, "length_scale has to be smaller than 0.5 for the kernel matrix to be diagonally dominant"
    sigmas = torch.ones(T, T) * length_scale
    sigmas_tridiag = torch.diagonal(sigmas, offset=0, dim1=-2, dim2=-1)
    sigmas_tridiag += torch.diagonal(sigmas, offset=1, dim1=-2, dim2=-1)
    sigmas_tridiag += torch.diagonal(sigmas, offset=-1, dim1=-2, dim2=-1)
    kernel_matrix = sigmas_tridiag + torch.eye(T) * (1.0 - length_scale)
    return kernel_matrix


def matern_kernel(T, length_scale):
    xs = torch.arange(T).float()
    xs_in = torch.unsqueeze(xs, 0)
    xs_out = torch.unsqueeze(xs, 1)
    distance_matrix = torch.abs(xs_in - xs_out)
    distance_matrix_scaled = distance_matrix / torch.sqrt(length_scale).type(torch.float32)
    kernel_matrix = torch.exp(-distance_matrix_scaled)
    return kernel_matrix


def cauchy_kernel(T, sigma, length_scale):
    xs = torch.arange(T).float()
    xs_in = torch.unsqueeze(xs, 0)
    xs_out = torch.unsqueeze(xs, 1)
    distance_matrix = (xs_in - xs_out) ** 2
    distance_matrix_scaled = distance_matrix / length_scale**2
    kernel_matrix = sigma / (distance_matrix_scaled + 1.0)

    alpha = 0.001
    eye = torch.eye(kernel_matrix.shape[-1])
    return kernel_matrix + alpha * eye


def make_nn(input_size, output_size, hidden_sizes):
    """This function used to creates fully connected neural network.

    Parameters
    ----------
    input_size : int,
        the dimension of input embeddings

    output_size : int,
        the dimension of out embeddings

    hidden_sizes : tuple,
        the tuple of hidden layer sizes, and the tuple length sets the number of hidden layers

    Returns
    -------
    output: tensor
        the processing embeddings
    """
    layers = []
    for i in range(len(hidden_sizes)):
        if i == 0:
            layers.append(nn.Linear(in_features=input_size, out_features=hidden_sizes[i]))
        else:
            layers.append(nn.Linear(in_features=hidden_sizes[i - 1], out_features=hidden_sizes[i]))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(in_features=hidden_sizes[-1], out_features=output_size))
    return nn.Sequential(*layers)


class CustomConv1d(torch.nn.Conv1d):
    def __init(self, in_channels, out_channels, kernel_size, padding):
        super().__init__(in_channels, out_channels, kernel_size, padding)

    def forward(self, x):
        if len(x.shape) > 2:
            shape = list(np.arange(len(x.shape)))
            new_shape = [0, shape[-1]] + shape[1:-1]
            out = super().forward(x.permute(*new_shape))
            shape = list(np.arange(len(out.shape)))
            new_shape = [0, shape[-1]] + shape[1:-1]
            if self.kernel_size[0] % 2 == 0:
                out = F.pad(out, (0, -1), "constant", 0)
            return out.permute(new_shape)

        return super().forward(x)


def make_cnn(input_size, output_size, hidden_sizes, kernel_size=3):
    """This function used to construct neural network consisting of
       one 1d-convolutional layer that utilizes temporal dependencies,
       fully connected network

    Parameters
    ----------
    input_size : int,
        the dimension of input embeddings

    output_size : int,
        the dimension of out embeddings

    hidden_sizes : tuple,
        the tuple of hidden layer sizes, and the tuple length sets the number of hidden layers,

    kernel_size : int
        kernel size for convolutional layer

    Returns
    -------
    output: tensor
        the processing embeddings
    """
    padding = kernel_size // 2

    cnn_layer = CustomConv1d(input_size, hidden_sizes[0], kernel_size=kernel_size, padding=padding)
    layers = [cnn_layer]

    for i, h in zip(hidden_sizes, hidden_sizes[1:]):
        layers.extend([nn.Linear(i, h), nn.ReLU()])
    if isinstance(output_size, tuple):
        net = nn.Sequential(*layers)
        return [net] + [nn.Linear(hidden_sizes[-1], o) for o in output_size]

    layers.append(nn.Linear(hidden_sizes[-1], output_size))
    return nn.Sequential(*layers)


class SimpleVAEEncoder(nn.Module):
    def __init__(self, input_size, z_size, hidden_sizes=(128, 128)):
        """Simple VAE Encoder that encodes each time step independently.

        Parameters:
        ----------
        input_size : int
            The feature dimension of the input.
        z_size : int
            The dimension of the latent space.
        hidden_sizes : tuple
            A tuple of hidden layer sizes.
        """
        super(SimpleVAEEncoder, self).__init__()
        self.input_size = input_size
        self.z_size = z_size

        # Define the network layers
        layers = []
        in_size = input_size
        for h_size in hidden_sizes:
            layers.append(nn.Linear(in_size, h_size))
            layers.append(nn.ReLU())
            in_size = h_size
        self.fc_layers = nn.Sequential(*layers)

        # Output layers for mean and log variance
        self.fc_mu = nn.Linear(in_size, z_size)
        self.fc_logvar = nn.Linear(in_size, z_size)

    def forward(self, x):
        """
        Forward pass through the encoder.

        Parameters:
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, time_steps, input_size).

        Returns:
        -------
        mu : torch.Tensor
            Mean of the latent distribution, shape (batch_size, time_steps, z_size).
        logvar : torch.Tensor
            Log variance of the latent distribution, shape (batch_size, time_steps, z_size).
        """
        batch_size, time_steps, _ = x.size()

        # Reshape to process each time step independently
        x = x.view(-1, self.input_size)  # Shape: (batch_size * time_steps, input_size)

        # Forward pass through the network
        h = self.fc_layers(x)  # Shape: (batch_size * time_steps, last_hidden_size)

        # Compute mean and log variance
        mu = self.fc_mu(h)      # Shape: (batch_size * time_steps, z_size)
        logvar = self.fc_logvar(h)  # Shape: (batch_size * time_steps, z_size)

        # Reshape back to (batch_size, time_steps, z_size)
        mu = mu.view(batch_size, time_steps, self.z_size)
        logvar = logvar.view(batch_size, time_steps, self.z_size)

        return mu, logvar


class SimpleVAEDecoder(nn.Module):
    def __init__(self, z_size, output_size, hidden_sizes=(128, 128)):
        """Simple VAE Decoder that decodes each time step independently.

        Parameters:
        ----------
        z_size : int
            The dimension of the latent space.
        output_size : int
            The feature dimension of the output.
        hidden_sizes : tuple
            A tuple of hidden layer sizes.
        """
        super(SimpleVAEDecoder, self).__init__()
        self.z_size = z_size
        self.output_size = output_size

        # Define the network layers
        layers = []
        in_size = z_size
        for h_size in hidden_sizes:
            layers.append(nn.Linear(in_size, h_size))
            layers.append(nn.ReLU())
            in_size = h_size
        self.fc_layers = nn.Sequential(*layers)

        # Output layer for reconstruction
        self.fc_output = nn.Linear(in_size, output_size)

    def forward(self, z):
        """
        Forward pass through the decoder.

        Parameters:
        ----------
        z : torch.Tensor
            Latent tensor of shape (batch_size, time_steps, z_size).

        Returns:
        -------
        recon_x : torch.Tensor
            Reconstructed input tensor, shape (batch_size, time_steps, output_size).
        """
        batch_size, time_steps, _ = z.size()

        # Reshape to process each time step independently
        z = z.view(-1, self.z_size)  # Shape: (batch_size * time_steps, z_size)

        # Forward pass through the network
        h = self.fc_layers(z)  # Shape: (batch_size * time_steps, last_hidden_size)

        # Compute reconstruction
        recon_x = self.fc_output(h)  # Shape: (batch_size * time_steps, output_size)

        # Reshape back to (batch_size, time_steps, output_size)
        recon_x = recon_x.view(batch_size, time_steps, self.output_size)

        return recon_x
