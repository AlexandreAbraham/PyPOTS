import torch
import torch.nn as nn
from torch.distributions import Normal, Independent
from .layers import (
    SimpleVAEEncoder,
    SimpleVAEDecoder,
    GpvaeEncoder, #modified versinos
    GpvaeDecoder, #modified versions
    GaussianProcess
)

# for mcar imputation
from pygrinder import mcar, fill_and_get_mask_torch

# for plotting
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

class BackboneGP_VAE(nn.Module):
    """Modified GPVAE model with prior variance proportional to missing values.

    Parameters
    ----------
    input_dim : int,
        the feature dimension of the input

    time_length : int,
        the length of each time series

    latent_dim : int,
        the feature dimension of the latent embedding

    encoder_sizes : tuple,
        the tuple of the network size in encoder

    decoder_sizes : tuple,
        the tuple of the network size in decoder

    beta : float,
        the weight of the KL divergence

    M : int,
        the number of Monte Carlo samples for ELBO estimation

    K : int,
        the number of importance weights for IWAE model

    kernel : str,
        the Gaussian Process kernel ["cauchy", "diffusion", "rbf", "matern"]

    sigma : float,
        the scale parameter for a kernel function

    length_scale : float,
        the length scale parameter for a kernel function

    kernel_scales : int,
        the number of different length scales over latent space dimensions
    """ 

    def __init__(
        self,
        input_dim,
        time_length,
        latent_dim,
        encoder_sizes=(64, 64),
        decoder_sizes=(64, 64),
        beta=1,
        M=1,
        K=1,
        kernel=None,  # No kernel needed for simple VAE
        sigma=1.0,
        length_scale=7.0,
        kernel_scales=1,
        window_size=24,
    ):
        super().__init__()
        # Remove kernel-related initializations
        self.kernel = kernel
        self.sigma = sigma
        self.length_scale = length_scale
        self.kernel_scales = kernel_scales

        self.input_dim = input_dim
        self.time_length = time_length
        self.latent_dim = latent_dim
        self.beta = beta
        self.encoder = GpvaeEncoder(input_dim, latent_dim, encoder_sizes)
        self.decoder = GpvaeDecoder(latent_dim, input_dim, encoder_sizes)
        self.M = M
        self.K = K

        self.gp = GaussianProcess(
            time_length=time_length,
            latent_dim=latent_dim,
            kernel='rbf',  # or any other kernel you prefer
            quantile=0.5   # Adjust quantile as needed
        )

        print('Model dimensions is: ')
        print(self.encoder)
        print(self.decoder)

    # for tracking
        self.loss_history = {
            'elbo': [],
            'nll_recon': [],
            'nll_imputation': [],
            'kl': [],
            'temporal_loss': [],
            'dependence_loss': [],
            'sigma': []
        }

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    @staticmethod
    def kl_divergence(q_dist, p_dist):
        return torch.distributions.kl_divergence(q_dist, p_dist)

    def plot_latent_series_and_reconstruction(self, z, px_z, qz_x, X_ori, X, latent_dim, time_steps, nll, kl, tl, folder='latent_plots'):
        """
        Plots the mean and variance of all latent time series and the original vs reconstructed data.

        Parameters
        ----------
        z : torch.Tensor
            The latent variable tensor with shape [batch_size, time_steps, latent_dim].
        px_z : torch.distributions.Normal
            The distribution for the reconstructed data.
        qz_x : torch.distributions.Distribution
            The posterior distribution q(z|x).
        X_ori : torch.Tensor
            The original data before any missingness is added.
        X : torch.Tensor
            The data used for reconstruction (with missing values).
        latent_dim : int
            The dimension of the latent space.
        time_steps : int
            The number of time steps in the sequence.
        nll : torch.Tensor
            The negative log-likelihood.
        kl : torch.Tensor
            The KL divergence.
        tl : torch.Tensor
            The temporal loss.
        folder : str
            The folder where the plot will be saved.
        """

        # Convert to CPU numpy arrays for plotting
        z_mean = z.mean(dim=0).detach().cpu().numpy()  # Shape: [time_steps, latent_dim]
        z_mean = z[0].detach().cpu().numpy()
        z_var = qz_x.variance[0].detach().cpu().numpy()

        X_ori_np = torch.clone(X_ori).detach().cpu().numpy()
        X_np = X.detach().cpu().numpy()

        # Sample 10 times from the posterior to get 10 reconstructions
        reconstructions = []
        for i in range(10):
            z_sample = qz_x.rsample()  # Sample from the posterior
            px_z_sample = self.decode(z_sample)  # Reconstruct the data
            X_recon_sample = px_z_sample.mean.detach().cpu().numpy()  # Get the mean of the reconstruction
            reconstructions.append(X_recon_sample)

        X_ori_np[X_ori_np == 0] = np.nan

        # Create directory if it doesn't exist
        if not os.path.exists(folder):
            os.makedirs(folder)

        # Create a unique filename based on the current timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = os.path.join(folder, f'latent_series_and_reconstruction_{timestamp}.png')

        # Plot all latent dimensions' mean and variance
        plt.figure(figsize=(15, 12))

        losses = f'kl = {kl.mean().item()} - nll = {nll.mean().item()} - temporal {tl.item()}'
        plt.subplot(4, 1, 1)
        for dim in range(latent_dim):
            plt.plot(range(time_steps), z_mean[:, dim], label=f'Latent dim {dim} Mean')
            plt.fill_between(range(time_steps), z_mean[:, dim] - z_var[:, dim], z_mean[:, dim] + z_var[:, dim], alpha=0.2)

        plt.title('Latent Time Series (Mean and Variance) ' + losses)
        plt.xlabel('Time Steps')
        plt.ylabel('Latent Values')

        # Plot scales for prior and posterior
        plt.subplot(4, 1, 2)
        nb_missing_vals = (X_np[0] == 0).sum(axis=1)
        missing_ratio = (X_np[0] != 0).mean(axis=1)
        prior_scale = (1 - missing_ratio) ** 0.5
        plt.semilogy(prior_scale, label='Prior scale')
        plt.semilogy(np.linalg.norm(z_var, axis=1), 'r:', label='Posterior scale')
        plt.semilogy(z_var, alpha = .5)
        plt.legend()

        # Plot original vs reconstructed data for all 10 reconstructions
        plt.subplot(4, 1, 3)
        for i, X_recon_sample in enumerate(reconstructions):
            plt.plot(range(time_steps), X_recon_sample[0, :, :], alpha=0.6)
        
        plt.gca().set_prop_cycle(None)
        plt.plot(range(time_steps), X_np[0, :, :], 'o')
        plt.gca().set_prop_cycle(None)
        plt.plot(range(time_steps), X_ori_np[0, :, :], '+', label='Original Data')
        plt.title('Original vs Reconstructed Data (10 Samples)')
        plt.xlabel('Time Steps')
        plt.ylabel('Feature Values')
        #plt.legend()

        # Save the plot
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

    # Modified forward function
    def forward(self, X, missing_mask):

        batch_size, time_steps, _ = X.size()
        # Merge prepare_data and simulate_missing_data
        X_ori, missing_mask_ori, X, missing_mask = self.prepare_and_simulate(X, missing_mask)
        # get indicating mask of where data has been artificially removed
        indicating_mask = (missing_mask_ori.float() - missing_mask.float()).to(torch.bool)

        # Encode input to get approximate posterior q(z|x) and sample from it
        qz_x = self.encode(X)
        z = qz_x.rsample()

        # Decode to get likelihood p(x|z)
        px_z = self.decode(z)

        # Negative log-likelihood
        nll_recon = self.compute_nll(px_z, X, missing_mask) #reconstruction error
        nll_imputation = self.compute_nll(px_z, X_ori, indicating_mask, keep_best = True) #imputation error
        # sample missing vals of X_ori from p(X^m; X^o)
        X_sampled = self.sample_missing_vals(X_ori, missing_mask_ori)
        nll_sampling = self.compute_nll(px_z, X_sampled, ~missing_mask_ori) #? I think the mask is right ?
        alpha = .1 #self.temperature * .5
        nll = nll_recon * (1 - alpha) + nll_imputation * alpha/2 + nll_sampling * alpha/2

        ## Compute KL divergence between q(z|x) and p(z) : Here the imposed prior is only on the variance

        # Compute the number of missing values per sample per time step
        missing_ratio = missing_mask.float().mean(dim=2)  # Shape: (batch_size * K * M, time_steps)
        prior_scale = (1 - missing_ratio) # this will be changed to something smarter later on

        # The prior on the variance forces the  mean variance to be proportional to the amount of missing
        # data for the observed point
        kl = ( qz_x.variance - prior_scale.unsqueeze(2)).pow(2)
        kl = kl.sum(1).mean()

        ## Compute a loss based on a Gaussian process prior between 1 point and the next
        # basically z_{t+1} ~N(z_t, sigma^2)
        sigma = qz_x.variance.mean().pow(.5) * 5 # this will also be changed to something smarter
        temporal_loss = (z[:,1:] - z[:,:-1]).pow(2).sum(axis=-1).mean() / sigma**2

        # get dependence loss between variance and missingness patterns
        sigma = np.exp(np.random.uniform(-8,5))
        dependence_loss, sigma = HSIC_loss(qz_x.variance[::(self.K * self.M)], missing_mask[::(self.K * self.M)])

        # get final elbo
        elbo = -nll - self.beta * kl - temporal_loss #- dependence_loss * 10000
        elbo = elbo.mean()

        self.loss_history['elbo'].append(elbo.item())
        self.loss_history['nll_recon'].append(nll_recon.mean().item())
        self.loss_history['nll_imputation'].append(nll_imputation.mean().item())
        self.loss_history['kl'].append(kl.mean().item())
        self.loss_history['temporal_loss'].append(temporal_loss.mean().item())
        self.loss_history['dependence_loss'].append(dependence_loss.mean().item())
        self.loss_history['sigma'].append(sigma)


        if len(self.loss_history['elbo']) % 100 == 0:
            self.plot_losses()

        # Validation and optional plotting
        self.validate_elbo(elbo, nll_recon, nll_imputation, kl, z, qz_x, X_ori, X, time_steps, px_z, temporal_loss)

        return -elbo

    def reconstruction_error(self, px_z, X, X_ori, missing_mask, missing_mask_ori, indicating_mask):

        # Reconstruction error
        nll_recon = self.compute_nll(px_z, X, missing_mask) #reconstruction error
        nll_imputation = self.compute_nll(px_z, X_ori, indicating_mask, keep_best = True) #imputation error
        # sample missing vals of X_ori from p(X^m; X^o)
        X_sampled = self.sample_missing_vals(X_ori, missing_mask_ori)
        nll_sampling = self.compute_nll(px_z, X_sampled, ~missing_mask_ori) #? I think the mask is right ?
        alpha = .1 #self.temperature * .5
        nll = nll_recon * (1 - alpha) + nll_imputation * alpha/2 + nll_sampling * alpha/2

        return nll

    def sample_missing_vals(self, X, missing_mask):
        """
        Sample along a law that approximates p(x^m|x^o)
        """

        # Reshape X and missing_mask to 2D tensors
        batch_size, seq_len, feature_size = X.shape
        X = X.detach().reshape(-1, feature_size)
        missing_mask = missing_mask.detach().reshape(-1, feature_size)

        X_recon = X.clone()
        num_samples = X.shape[0]

        for j in range(feature_size):
            # Indices of samples where feature j is missing
            missing_indices = torch.where(missing_mask[:, j])[0]
            # Indices of samples where feature j is observed
            observed_indices = torch.where(~missing_mask[:, j])[0]

            if len(observed_indices) > 0:

                # Extract observed values for feature j
                observed_values = X[observed_indices, j]

                # Compute distances between missing samples and observed samples
                xi_missing = X[missing_indices]  # Shape: [num_missing, feature_size]
                xi_observed = X[observed_indices]  # Shape: [num_observed, feature_size]

                # Compute distance matrix using the custom function
                distances = self.compute_distance_matrix(xi_missing, xi_observed)  # Shape: [num_missing, num_observed]

                # Compute softmax probabilities over negative distances
                probabilities = torch.softmax(-distances, dim=1)  # Along observed samples

                # Handle infinite distances (no common features)
                valid_mask = ~torch.isinf(distances)
                probabilities = probabilities * valid_mask.float()
                probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)

                # Sample indices for each missing sample
                sampled_indices = torch.multinomial(probabilities, num_samples=1).squeeze(1)  # Shape: [num_missing]

                # Get the observed values corresponding to the sampled indices
                sampled_values = observed_values[sampled_indices]  # Shape: [num_missing]

                # Assign the sampled values to the missing positions
                X_recon[missing_indices, j] = sampled_values

        # Reshape X_recon back to the original shape
        X_recon = X_recon.reshape(batch_size, seq_len, feature_size)
        return X_recon

    def compute_distance_matrix(self, xi_missing, xi_observed):
        """
        Computes the distance matrix between xi_missing and xi_observed,
        handling missing values by considering only common observed features
        and adding expected differences for missing features.

        Args:
            xi_missing (Tensor): Tensor of shape [num_missing, feature_size].
            xi_observed (Tensor): Tensor of shape [num_observed, feature_size].

        Returns:
            distances (Tensor): Tensor of shape [num_missing, num_observed].
        """
        num_missing, feature_size = xi_missing.shape
        num_observed = xi_observed.shape[0]

        # Create masks indicating observed features (non-zero)
        mask_missing = xi_missing != 0  # Shape: [num_missing, feature_size]
        mask_observed = xi_observed != 0  # Shape: [num_observed, feature_size]

        # Expand dimensions to align samples for broadcasting
        xi_missing_exp = xi_missing.unsqueeze(1)  # Shape: [num_missing, 1, feature_size]
        xi_observed_exp = xi_observed.unsqueeze(0)  # Shape: [1, num_observed, feature_size]

        # Expand masks
        mask_missing_exp = mask_missing.unsqueeze(1)  # Shape: [num_missing, 1, feature_size]
        mask_observed_exp = mask_observed.unsqueeze(0)  # Shape: [1, num_observed, feature_size]

        # Compute common observed features mask
        common_observed_mask = mask_missing_exp & mask_observed_exp  # Shape: [num_missing, num_observed, feature_size]

        # Count number of common observed features for each pair
        num_common_features = common_observed_mask.sum(dim=2)  # Shape: [num_missing, num_observed]

        # Handle cases with no common features
        no_common_features = num_common_features == 0

        # Compute squared differences where both features are observed
        diff = xi_missing_exp - xi_observed_exp  # Shape: [num_missing, num_observed, feature_size]
        squared_diff = diff ** 2

        # Set squared differences to zero where features are not commonly observed
        squared_diff = squared_diff * common_observed_mask.float()

        # Sum squared differences over features
        sum_squared_diff = squared_diff.sum(dim=2)  # Shape: [num_missing, num_observed]

        # Compute expected squared difference for missing features (2.0 per missing feature)
        num_missing_features = feature_size - num_common_features
        expected_squared_diff = 2.0 * num_missing_features  # Shape: [num_missing, num_observed]

        # Total squared difference includes observed and expected differences
        total_squared_diff = sum_squared_diff + expected_squared_diff  # Shape: [num_missing, num_observed]

        # Assign a large distance where there are no common features
        distances = total_squared_diff.masked_fill(no_common_features, float('inf'))

        return distances

    def prepare_and_simulate(self, X, missing_mask):
        """Prepare data by repeating and simulate missing data."""
        X_ori = X.repeat(self.K * self.M, 1, 1) #augment by K x M if we want to sample multiple times in the latent space (if not K = M = 1)
        missing_mask_ori = missing_mask.repeat(self.K * self.M, 1, 1).type(torch.bool) 
        X = mcar(X_ori, p=.5) #missing completely at random, missingness proba = 0.3
        X, missing_mask = fill_and_get_mask_torch(X) 
        #missing_mask = (X != 0).type(torch.bool) # just to be sure this is what the missing mask returns
        return X_ori, missing_mask_ori, X, missing_mask.type(torch.bool)

    def compute_nll(self, px_z, X, mask, keep_best = False):
        """
        Compute the negative log-likelihood. If we are dealing with imputation we set keep_best is true, and we only get 
        the loss on the closest element within each of the K x M samples for each observation 
        -> I need to explicit the math for this but this helps to keep a diversity within the guesses but still
        drives the imputation towards the right answer
        """
        nll = -px_z.log_prob(X)
        nll = torch.where(torch.isfinite(nll), nll, torch.zeros_like(nll))
        if mask is not None:
            nll = torch.where(mask, nll, torch.zeros_like(nll))
        if keep_best:
            a,b,c = nll.shape
            nll = nll.reshape(self.K * self.M, -1, b, c) #reshape so first axis containts K x M samples for a same observation
            nll = torch.min(nll, axis = 0)[0]
            return nll.sum()
        else:
            return nll.sum(dim=(1, 2))

    def validate_elbo(self, elbo, nll_recon, nll_imputation, kl, z, qz_x, X_ori, X, time_steps, px_z, tl):
        """Perform assertions, debugging, and optional plotting."""
        assert not (elbo.abs() > 1e6).any(), print('elbo too big', nll_recon.mean().item(), nll_imputation.mean().item(), kl.mean().item())
        assert not (elbo > 50), print('elbo negative', elbo.item(), nll_recon.mean().item(), nll_imputation.mean().item(), kl.mean().item())

        if torch.rand(1) < 0.01: # plot randomly with proba 1/1000 je sais c'est degueu mais j'aime bien
            self.plot_latent_series_and_reconstruction(z, px_z, qz_x, X_ori.detach(), X.detach(), self.latent_dim, time_steps, -elbo, kl, tl)

    def plot_losses(self):
        iterations = range(1, len(self.loss_history['elbo']) + 1)
        plt.figure(figsize=(10, 6))
        plt.semilogy(iterations, self.loss_history['elbo'], label='ELBO')
        plt.semilogy(iterations, self.loss_history['nll_recon'], label='NLL Recon')
        plt.semilogy(iterations, self.loss_history['nll_imputation'], label='NLL Imputation')
        plt.semilogy(iterations, self.loss_history['kl'], label='KL Divergence')
        plt.semilogy(iterations, self.loss_history['temporal_loss'], label='Temporal Loss')
        plt.semilogy(iterations, self.loss_history['dependence_loss'], label='Dependence Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss (log scale)')
        plt.title('Loss Components over Iterations')
        plt.legend()
        plt.grid(True)

        plot_path = os.path.join('latent_plots/losses.png')

        # Save the plot
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

def HSIC_loss(A, B, sigma = 1.0):
    """
    Loss that computes the covariance between A and B
    """

    # Flatten the tensors to compute covariance across all elements
    A = A.reshape(-1, A.shape[-1])
    B = B.reshape(-1, B.shape[-1]).float()  # Convert to float for computation

    n = A.size(0)  # Number of samples
    device = A.device

    # Compute Gram matrices
    K = gaussian_kernel(A, sigma)
    L = gaussian_kernel(B, sigma)

    # Center the Gram matrices
    H = centering_matrix(n, device)
    Kc = H @ K @ H
    Lc = H @ L @ H

    # Compute HSIC
    hsic = (1 / (n - 1) ** 2) * torch.trace(Kc @ Lc)

    return hsic, sigma

def gaussian_kernel(X, sigma):
    """
    Computes the Gaussian (RBF) kernel matrix for tensor X.
    """
    pairwise_distances = torch.cdist(X, X) ** 2  # Shape: [n, n]
    K = torch.exp(-pairwise_distances / (2 * sigma ** 2))
    return K

def centering_matrix(n, device):
    """
    Creates a centering matrix of size n x n.
    """
    I = torch.eye(n).to(device)
    ones = torch.ones(n, n).to(device) / n
    H = I - ones
    return H
