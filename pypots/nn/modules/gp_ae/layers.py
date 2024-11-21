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

def make_fc(input_size, output_size, hidden_sizes):
    """
    Make a fully connected network that takes all the time steps independently
    """
    layers = []

    # First fully connected layer
    layers.extend([nn.Linear(input_size, hidden_sizes[0]), nn.ReLU()])

    # Add additional hidden layers
    for i, h in zip(hidden_sizes[:-1], hidden_sizes[1:]):
        #layers.extend([nn.Linear(i, h), nn.BatchNorm1d(h), nn.ReLU()]) # add batchnorm
        layers.extend([nn.Linear(i, h), nn.LeakyReLU()]) # add batchnorm


    # Handle output layers
    if isinstance(output_size, tuple):
        net = nn.Sequential(*layers)
        return [net] + [nn.Linear(hidden_sizes[-1], o) for o in output_size]

    layers.append(nn.Linear(hidden_sizes[-1], output_size))
    net = nn.Sequential(*layers)

    # Initialize weights using Xavier initialization
    for m in net.modules():
        if isinstance(m, nn.Linear):
            torch.init.xavier_uniform_(m.weight)

    return net
class GpvaeEncoder(nn.Module):
    def __init__(self, input_size, z_size, hidden_sizes=(128, 128, 128)):
        super().__init__()
        self.z_size = int(z_size)
        self.input_size = input_size
        self.net, self.mu_layer, self.logvar_layer = make_fc(
            input_size*2, (z_size, z_size), hidden_sizes
        )




    def forward(self, x, mask = None):
        batch_size, time_length, input_size = x.size()
        #print(f"Input shape: {x.size()}")  # Debug

        # Reshape input to [batch_size * time_length, input_size]
        x_reshaped = torch.clone(x.view(batch_size * time_length, input_size))
        mask = (x_reshaped!=0)
        x_reshaped[~mask] = torch.rand(size = x_reshaped[~mask].shape)
        x_concat = torch.cat((x_reshaped,mask), dim = 1)
        #print(f"Reshaped input: {x_reshaped.size()}")  # Debug

        # Pass through the fully connected network
        mapped = self.net(x_concat)

        # Compute mean and log variance
        mu = self.mu_layer(mapped)
        logvar = self.logvar_layer(mapped)
        #print(f"Mu shape: {mu.size()}")       # Should be [batch_size * time_length, z_size]
        #print(f"Logvar shape: {logvar.size()}")  # Should be [batch_size * time_length, z_size]

        # Compute standard deviation
        std = torch.exp(0.5 * logvar)
        #print(f"Std shape before reshape: {std.size()}")  # Should be [batch_size * time_length, z_size]

        # Reshape tensors back to [batch_size, time_length, z_size]
        mu = mu.view(batch_size, time_length, self.z_size)
        std = std.view(batch_size, time_length, self.z_size)
        #print(f"Mu shape after reshape: {mu.size()}")     # [batch_size, time_length, z_size]
        #print(f"Std shape after reshape: {std.size()}")   # [batch_size, time_length, z_size]

        # Create Normal distribution
        z_dist = torch.distributions.Normal(loc=mu, scale=std)
        # Make it an Independent distribution over the last dimension (latent dimension)
        z_dist = torch.distributions.Independent(z_dist, 1)

        assert not torch.isnan(mu).any(), print(mu)
        assert not torch.isnan(std).any(), print(std)

        return z_dist

class GpvaeEncoder_cnn(nn.Module):
    def __init__(self, input_size, z_size, hidden_sizes=(128, 128), window_size=24):
        """This module is an encoder with 1d-convolutional network and multivariate Normal posterior used by GP-VAE with
        proposed banded covariance matrix

        Parameters
        ----------
        input_size : int,
            the feature dimension of the input

        z_size : int,
            the feature dimension of the output latent embedding

        hidden_sizes : tuple,
            the tuple of the hidden layer sizes, and the tuple length sets the number of hidden layers

        window_size : int
            the kernel size for the Conv1D layer
        """
        super().__init__()
        self.z_size = int(z_size)
        self.input_size = input_size
        self.net, self.mu_layer, self.logvar_layer = make_cnn(
            input_size, (z_size, z_size), hidden_sizes, window_size
        )

    def forward(self, x):
        mapped = self.net(x)
        batch_size = mapped.size(0)
        time_length = mapped.size(1)

        num_dim = len(mapped.shape)
        mu = self.mu_layer(mapped)
        logvar = self.logvar_layer(mapped)

        #print(f"Mu shape: {mu.size()}")       # Should be [batch_size * time_length, z_size]
        #print(f"Logvar shape: {logvar.size()}")  # Should be [batch_size * time_length, z_size]

        # Compute standard deviation
        std = torch.exp(0.5 * logvar)
        #print(f"Std shape before reshape: {std.size()}")  # Should be [batch_size * time_length, z_size]

        # Reshape tensors back to [batch_size, time_length, z_size]
        mu = mu.view(batch_size, time_length, self.z_size)
        std = std.view(batch_size, time_length, self.z_size)
        #print(f"Mu shape after reshape: {mu.size()}")     # [batch_size, time_length, z_size]
        #print(f"Std shape after reshape: {std.size()}")   # [batch_size, time_length, z_size]

        # Create Normal distribution
        z_dist = torch.distributions.Normal(loc=mu, scale=std)
        # Make it an Independent distribution over the last dimension (latent dimension)
        z_dist = torch.distributions.Independent(z_dist, 1)

        assert not torch.isnan(mu).any(), print(mu)
        assert not torch.isnan(std).any(), print(std)

        return z_dist


class GpvaeDecoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=(256, 256)):
        """This module is a decoder with Gaussian output distribution.

        Parameters
        ----------
        output_size : int,
            the feature dimension of the output

        hidden_sizes: tuple
            the tuple of hidden layer sizes, and the tuple length sets the number of hidden layers.
        """
        super().__init__()
        self.net = make_nn(input_size, output_size, hidden_sizes)

    def forward(self, x):
        mu = self.net(x) 
        var = torch.ones_like(mu).type(torch.float)*.3
        return torch.distributions.Normal(mu, var)

    ##





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
        mask = (x!=0)
        x = torch.cat((x,mask))
        print(x.shape)

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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

class GaussianProcess(nn.Module):
    def __init__(self, time_length, latent_dim, kernel='rbf', quantile=0.5):
        """
        GaussianProcess class to estimate GP kernel parameters on each latent dimension.

        Parameters:
        ----------
        time_length : int
            The length of the time series.
        latent_dim : int
            The number of latent dimensions.
        kernel : str
            The type of kernel to use ('rbf', 'diffusion', 'matern', 'cauchy').
        quantile : float
            The quantile threshold for variance to select observations.
        """
        super().__init__() #GaussianProcess, self
        self.time_length = time_length
        self.latent_dim = latent_dim
        self.kernel_type = kernel
        self.quantile = quantile

        # Initialize kernel parameters for each latent dimension
        # We'll learn length_scale and sigma for each latent dimension
        self.length_scales = nn.Parameter(torch.ones(latent_dim) * 1.0)
        self.sigmas = nn.Parameter(torch.ones(latent_dim) * 1.0)

    def forward(self, z, variances):
        """
        Estimate the GP kernel parameters on each latent dimension.

        Parameters:
        ----------
        z : torch.Tensor
            Latent variables of shape (batch_size, time_steps, latent_dim)
        variances : torch.Tensor
            Variances of the latent variables of shape (batch_size, time_steps, latent_dim)

        Returns:
        -------
        prior : torch.distributions.MultivariateNormal
            The prior distribution with estimated kernel parameters.
        """
        batch_size, time_steps, latent_dim = z.size()
        device = z.device

        # Initialize list to collect kernel matrices for each latent dimension
        kernel_matrices = []

        # For each latent dimension
        for d in range(self.latent_dim):
            # Get the variances and latent values for this dimension
            var_d = variances[:, :, d]  # Shape: (batch_size, time_steps)
            z_d = z[:, :, d]            # Shape: (batch_size, time_steps)

            # Flatten batch and time dimensions
            var_d_flat = var_d.reshape(-1)
            z_d_flat = z_d.reshape(-1)

            # Compute the threshold based on quantile
            threshold = torch.quantile(var_d_flat, self.quantile)

            # Select indices where variance is below the threshold
            indices = (var_d_flat <= threshold).nonzero(as_tuple=True)[0]

            # Select corresponding time indices
            time_indices = torch.arange(time_steps, device=device).repeat(batch_size)

            # Select observations with low variance
            z_d_selected = z_d_flat[indices]
            time_selected = time_indices[indices]

            # If there are not enough points, skip fitting
            if len(z_d_selected) < 2:
                # Use default kernel parameters
                length_scale = torch.exp(self.length_scales[d])
                sigma = torch.exp(self.sigmas[d])
            else:
                # Estimate length_scale based on selected data
                time_diffs = torch.abs(time_selected.unsqueeze(0) - time_selected.unsqueeze(1))
                covariances = (z_d_selected.unsqueeze(0) - z_d_selected.unsqueeze(1)) ** 2

                # Avoid division by zero
                time_diffs = time_diffs + 1e-6

                # Estimate length_scale (simplified for demonstration)
                length_scale_est = torch.sqrt((covariances / time_diffs).mean())
                length_scale = length_scale_est.detach()

                # Update the parameter (optional)
                self.length_scales.data[d] = torch.log(length_scale + 1e-6)

                # Estimate sigma
                sigma_est = z_d_selected.var()
                sigma = sigma_est.detach()

                # Update the parameter (optional)
                self.sigmas.data[d] = torch.log(sigma + 1e-6)

            # Build the kernel matrix using the estimated parameters
            if self.kernel_type == 'rbf':
                kernel_matrix = rbf_kernel(self.time_length, length_scale)
            elif self.kernel_type == 'diffusion':
                kernel_matrix = diffusion_kernel(self.time_length, length_scale)
            elif self.kernel_type == 'matern':
                kernel_matrix = matern_kernel(self.time_length, length_scale)
            elif self.kernel_type == 'cauchy':
                kernel_matrix = cauchy_kernel(self.time_length, sigma, length_scale)
            else:
                raise ValueError(f"Unsupported kernel type: {self.kernel_type}")

            # Append the kernel matrix
            kernel_matrices.append(kernel_matrix.to(device))

        # Stack kernel matrices for all latent dimensions
        kernel_matrices = torch.stack(kernel_matrices, dim=0)  # Shape: (latent_dim, time_steps, time_steps)

        # Expand to match batch size
        kernel_matrices = kernel_matrices.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # (batch_size, latent_dim, T, T)

        # Build the prior distribution
        prior_loc = torch.zeros(batch_size, self.latent_dim, self.time_length, device=device)
        prior = MultivariateNormal(prior_loc, covariance_matrix=kernel_matrices)

        return prior


##### FROM ASHMAN

from .layers_ashman import LinearNN, NNHeteroGaussian
from torch.nn import functional as F
from torch.distributions import Normal

class FactorNet(nn.Module):
    """FactorNet from Sparse Gaussian Process Variational Autoencoders.

    :param in_dim (int): dimension of the input variable.
    :param out_dim (int): dimension of the output variable.
    :param h_dims (list, optional): dimensions of hidden layers.
    :param min_sigma (float, optional): sets the minimum output sigma.
    :param initi_sigma (float, optional): sets the initial output sigma.
    :param nonlinearity (function, optional): non-linearity to apply in
    between layers.
    """
    def __init__(self, input_size, z_size, hidden_sizes=(128, 128, 128), min_sigma=1e-3,
                 init_sigma=None, nonlinearity=F.relu):
        super().__init__()

        in_dim = input_size
        out_dim = z_size
        h_dims = hidden_sizes

        self.out_dim = out_dim

        # Rescale sigmas for multiple outputs.
        if init_sigma is not None:
            init_sigma = init_sigma * in_dim ** 0.5

        min_sigma = min_sigma * in_dim ** 0.5

        # A network for each dimension.
        self.networks = nn.ModuleList()
        for _ in range(in_dim):
            self.networks.append(NNHeteroGaussian(
                1, out_dim, h_dims, min_sigma, init_sigma,
                nonlinearity=nonlinearity))

    def forward(self, z, mask=None):
        """Returns parameters of a diagonal Gaussian distribution."""
        np_1 = torch.zeros(z.shape[0], z.shape[1], self.out_dim)
        np_2 = torch.zeros_like(np_1)

        # Pass through individual networks.
        #print(z.transpose(0, 2).shape)
        for dim, z_dim in enumerate(z.permute(2, 0, 1)):
            print(z_dim.shape)
            if mask is not None:
                batch_idx, time_idx = torch.where(mask[:, :, dim])[0]
                z_in = z_dim[batch_idx, time_idx]
                z_in_shape = z_in.shape
                z_in = z_in.flatten().unsqueeze(1)

                print(z_in.shape)

                # Don't pass through if no inputs.
                if len(z_in) != 0:
                    pz = self.networks[dim](z_in)
                    mu, sigma = pz.mean, pz.stddev
                    print(mu.shape)
                    mu = mu.reshape(z_in_shape)
                    sigma = sigma.reshape(z_in_shape)
                    print(mu.shape, z_dim.shape, np_1.shape)
                    np_1[idx, dim, :] = mu / sigma ** 2
                    np_2[idx, dim, :] = - 1. / (2. * sigma ** 2)
            else:
                z_in = z_dim.unsqueeze(1)
                pz = self.networks[dim](z_in)
                mu, sigma = pz.mean, pz.stddev
                np_1[:, dim, :] = mu / sigma ** 2
                np_2[:, dim, :] = -1. / (2. * sigma ** 2)

        # Sum natural parameters.
        np_1 = torch.sum(np_1, 1)
        np_2 = torch.sum(np_2, 1)
        sigma = (- 1. / (2. * np_2)) ** 0.5
        mu = np_1 * sigma ** 2.

        pz = Normal(mu, sigma)

        return pz

    def forward(self, z, mask=None):
        """Returns parameters of a diagonal Gaussian distribution."""
       
        # reshape z
        z_shape = z.shape
        z = z.reshape(-1, z.shape[2])       
       
        np_1 = torch.zeros(z.shape[0], z.shape[1], self.out_dim)
        np_2 = torch.zeros_like(np_1)

        # Pass through individual networks.
        for dim, z_dim in enumerate(z.transpose(0, 1)):
            if mask is not None:
                idx = torch.where(mask[:, dim])[0]
                z_in = z_dim[idx].unsqueeze(1)

                # Don't pass through if no inputs.
                if len(z_in) != 0:
                    pz = self.networks[dim](z_in)
                    mu, sigma = pz.mean, pz.stddev
                    np_1[idx, dim, :] = mu / sigma ** 2
                    np_2[idx, dim, :] = - 1. / (2. * sigma ** 2)
            else:
                z_in = z_dim.unsqueeze(1)
                pz = self.networks[dim](z_in)
                mu, sigma = pz.mean, pz.stddev
                np_1[:, dim, :] = mu / sigma ** 2
                np_2[:, dim, :] = -1. / (2. * sigma ** 2)

            assert not torch.isnan(sigma).any(), print('sigma', sigma)

        # Sum natural parameters.
        epsilon = 1e-1
        np_1 = torch.sum(np_1, 1)
        np_2 = torch.sum(np_2, 1) + epsilon
        sigma = torch.clip(- 1. / (2. * np_2),min = 0.01) ** 0.5
        mu = np_1 * sigma ** 2.

        assert not (np_2==0).any(), print('np is 0', np_2)
        assert not torch.isnan(np_2).any(), print('np', np_2)

        assert not torch.isnan(sigma).any(), print('sigma is nan', sigma, np_2)
        assert not torch.isnan(np_2).any(), print('np', np_2)

        mu = mu.reshape((z_shape[0], z_shape[1], -1))
        sigma = sigma.reshape((z_shape[0], z_shape[1], -1))

        pz = Normal(mu, sigma)

        return pz


class IndexNet(nn.Module):
    """IndexNet from Sparse Gaussian Process Variational Autoencoders.

    :param in_dim (int): dimension of the input variable.
    :param out_dim (int): dimension of the output variable.
    :param inter_dim (int): dimension of intermediate representation.
    :param h_dims (list, optional): dimension of the encoding function.
    :param rho_dims (list, optional): dimension of the shared function.
    :param min_sigma (float, optional): sets the minimum output sigma.
    :param init_sigma (float, optional): sets the initial output sigma.
    :param nonlinearity (function, optional): non-linearity to apply in
    between layers.
    """
    def __init__(self, input_size, z_size, hidden_sizes=(128, 128, 128),
                 min_sigma=1e-3, init_sigma=None,
                 nonlinearity=F.relu):
        super().__init__()

        self.out_dim = z_size
        self.inter_dim = hidden_sizes[0]

        # A network for each dimension.
        self.networks = nn.ModuleList()
        for _ in range(3):
            self.networks.append(LinearNN(1, self.inter_dim, hidden_sizes, nonlinearity))

        # Takes the aggregation of the outputs from self.networks.
        self.rho = nn.ModuleList()
        for _ in range(3):
            self.rho.append(LinearNN(1, self.inter_dim, hidden_sizes, nonlinearity))

    def forward(self, x, mask=None):
        """Returns parameters of a diagonal Gaussian distribution."""
        out = torch.zeros(x.shape[0], x.shape[1], self.inter_dim)

        # Pass through individual networks.
        for dim, z_dim in enumerate(x.transpose(0, 1)):
            if mask is not None:
                idx = torch.where(mask[:, dim])[0]
                z_in = z_dim[idx].unsqueeze(1)

                # Don't pass through if no inputs.
                if len(z_in) != 0:
                    z_out = self.networks[dim](z_in)
                    out[idx, dim, :] = z_out

            else:
                z_in = z_dim.unsqueeze(1)
                z_out = self.networks[dim](z_in)
                out[:, dim, :] = z_out

        # Aggregation layer.
        out = torch.sum(out, 1)

        # Pass through shared network.
        pz = self.rho(out)

        return pz