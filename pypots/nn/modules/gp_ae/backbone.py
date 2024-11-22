import torch
import torch.nn as nn
from torch.distributions import Normal, Independent
from .layers import (
    SimpleVAEEncoder,
    SimpleVAEDecoder,
    GpvaeEncoder,  # modified versions
    GpvaeDecoder,  # modified versions
    GaussianProcess,
    FactorNet
)

# for mcar imputation
from pygrinder import mcar, fill_and_get_mask_torch

# for plotting
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import time

class BackboneGP_VAE(nn.Module):
    """Modified GPVAE model with prior variance proportional to missing values.

    Parameters
    ----------
    input_dim : int
        The feature dimension of the input.

    time_length : int
        The length of each time series.

    latent_dim : int
        The feature dimension of the latent embedding.

    encoder_sizes : tuple
        The tuple of the network size in encoder.

    decoder_sizes : tuple
        The tuple of the network size in decoder.

    beta : float
        The weight of the KL divergence.

    M : int
        The number of Monte Carlo samples for ELBO estimation.

    K : int
        The number of importance weights for IWAE model.

    kernel : str
        The Gaussian Process kernel ["cauchy", "diffusion", "rbf", "matern"].

    sigma : float
        The scale parameter for a kernel function.

    length_scale : float
        The length scale parameter for a kernel function.

    kernel_scales : int
        The number of different length scales over latent space dimensions.
    """

    def __init__(
        self,
        input_dim,
        time_length,
        latent_dim,
        encoder_sizes=(128, 64),
        decoder_sizes=(64, 128),
        beta=1,
        M=1,
        K=1,
        kernel=None,  # No kernel needed for simple VAE
        sigma=1.0,
        length_scale=7.0,
        kernel_scales=1,
        window_size=24,
        device=None,  # Added device parameter
    ):
        super().__init__()
        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Remove kernel-related initializations
        self.kernel = kernel
        self.sigma = sigma
        self.length_scale = length_scale
        self.kernel_scales = kernel_scales

        self.input_dim = input_dim
        self.time_length = time_length
        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = 0.01

        # Ensure that encoder and decoder are on the correct device
        self.encoder = GpvaeEncoder(input_dim, latent_dim, encoder_sizes, device=self.device).to(self.device)
        # self.encoder = FactorNet(input_dim, latent_dim, encoder_sizes).to(self.device)
        self.decoder = GpvaeDecoder(latent_dim, input_dim, decoder_sizes).to(self.device)
        self.M = M
        self.K = K

        self.gp = GaussianProcess(
            time_length=time_length,
            latent_dim=latent_dim,
            kernel='rbf',  # or any other kernel you prefer
            quantile=0.5  # Adjust quantile as needed
        ).to(self.device)

        print('Model dimensions is: ')
        print(self.encoder)
        print(self.decoder)

        self.p = 0.5

        self.sampling = False
        self.train_decoder = True

        # For tracking
        self.loss_history = {
            'elbo': [],
            'nll_recon': [],
            'nll_imputation': [],
            'nll_sampling': [],
            'kl': [],
            'temporal_loss': [],
            'dependence_loss': [],
            'sigma': []
        }

    def encode(self, x, missing_mask=None):
        return self.encoder(x, missing_mask)

    def decode(self, z):
        return self.decoder(z)

    @staticmethod
    def kl_divergence(q_dist, p_dist):
        return torch.distributions.kl_divergence(q_dist, p_dist)

    def temporal_loss(self, z, X, missing_mask, batch_size, eps=1e-3):
        temporal_loss = 0

        for i in range(batch_size):
            mask_diff = (missing_mask[i::batch_size, 1:] * missing_mask[i::batch_size, :-1])
            mask_diff_sum = mask_diff.sum(2)
            mask_diff_sum[mask_diff_sum == 0] = 1
            X_diff = ((X[i::batch_size, 1:] - X[i::batch_size, :-1]) * mask_diff).pow(2).sum(2) / mask_diff_sum
            z_diff = (z[i::batch_size, 1:] - z[i::batch_size, :-1]).pow(2).sum(2)
            temporal_loss += z_diff / (X_diff + eps)

            assert not torch.isnan(temporal_loss).any(), print(temporal_loss[temporal_loss != temporal_loss], (z_diff / (X_diff + eps))[temporal_loss != temporal_loss])
            assert not torch.isnan(temporal_loss).any(), print(z_diff[temporal_loss != temporal_loss], z_diff[temporal_loss != temporal_loss])

        return temporal_loss.mean()

    def forward(self, X, missing_mask):
        batch_size, time_steps, _ = X.size()
        # Merge prepare_data and simulate_missing_data
        X_ori, missing_mask_ori, X, missing_mask = self.prepare_and_simulate(X, missing_mask)

        # Compute masks
        missing_mask = (X != 0)
        missing_mask_ori = (X_ori != 0)
        # Get indicating mask of where data has been artificially removed
        indicating_mask = (missing_mask_ori.float() - missing_mask.float()).to(torch.bool)

        # Encode input to get approximate posterior q(z|x) and sample from it
        qz_x = self.encode(X, missing_mask)
        z = qz_x.rsample()

        # Decode to get likelihood p(x|z)
        px_z = self.decode(z)

        # Compute reconstruction error
        nll, nll_recon, nll_imputation, nll_sampling = self.reconstruction_error(
            qz_x, px_z, X, X_ori, missing_mask, missing_mask_ori, indicating_mask
        )

        # Compute KL divergence between q(z|x) and p(z): Here the imposed prior is only on the variance
        kl = qz_x.mean.pow(2).sum(2).pow(0.5).mean()
        kl = kl.sum()

        # Compute a loss based on a Gaussian process prior between 1 point and the next
        temporal_loss = self.temporal_loss(z, X, missing_mask, batch_size)

        # Get dependence loss between variance and missingness patterns
        sigma = np.exp(np.random.uniform(-8, 5))
        dependence_loss, sigma = HSIC_loss(qz_x.variance[::(self.K * self.M)], missing_mask[::(self.K * self.M)])

        # Get final ELBO
        elbo = -nll - self.beta * kl - temporal_loss * self.gamma  # - dependence_loss * 10000
        elbo = elbo.mean()

        self.loss_history['elbo'].append(-1 * elbo.item())
        self.loss_history['nll_recon'].append(nll_recon.mean().item())
        self.loss_history['nll_imputation'].append(nll_imputation.mean().item())
        self.loss_history['nll_sampling'].append(nll_sampling.mean().item())
        self.loss_history['kl'].append(kl.mean().item())
        self.loss_history['temporal_loss'].append(temporal_loss.mean().item())
        self.loss_history['dependence_loss'].append(dependence_loss.mean().item())
        self.loss_history['sigma'].append(sigma)

        if len(self.loss_history['elbo']) % 10 == 0:
            self.plot_losses()

        # Validation and optional plotting
        self.validate_elbo(elbo, nll_recon, nll_imputation, kl, z, qz_x, X_ori, X, time_steps, px_z, temporal_loss)

        return -elbo

    def latent_imputation_error(self, qz_x, X_ori, missing_mask, missing_mask_ori, for_plotting=False):
        """
        Loss that forces qz_x to contain X_ori
        """
        qz_x_ori = self.encode(X_ori)

        loss = (qz_x.mean.detach().cpu() - qz_x_ori.mean.detach().cpu()).pow(2) / (qz_x.variance / qz_x_ori.variance.detach().cpu())

        mask_imputed_coords = (missing_mask != missing_mask_ori).sum(axis=2) != 0

        if not for_plotting:
            loss = loss[mask_imputed_coords]
        else:
            print('mean of mask of imputed coords :', mask_imputed_coords.float().mean())

        return loss * len(mask_imputed_coords.flatten()) * 100  # Normalization trick

    def latent_sampling_error(self, qz_x, X_sampled):
        """
        This error forces the AE to learn representation of missing values
        X_sampled (num_samples, batch_size, n_observations, n_dimensions)
        """
        loss = 0
        # For each sample, compute the log prob of its embedding mean belonging to qz_x
        for x_sampled in X_sampled:
            qz_x_sampled = self.encode(x_sampled)
            loss -= qz_x_sampled.log_prob(qz_x.rsample().detach().cpu())

        return 100 * loss / len(qz_x.mean.flatten())

    def sampling_error(self, qz_x, X_ori, missing_mask_ori):
        """
        Forces the model to learn representations of missing values.
        """
        num_samples = 10
        X_sampled, mask_sampled = self.sample_selected_missing_vals(X_ori, missing_mask_ori, num_samples=num_samples)
        X_sampled = torch.permute(X_sampled, (3, 0, 1, 2))
        loss = self.latent_sampling_error(qz_x, X_sampled)

        return loss.mean()

    def reconstruction_error(self, qz_x, px_z, X, X_ori, missing_mask, missing_mask_ori, indicating_mask):
        # Negative log-likelihood
        nll_recon = self.compute_nll(px_z, X_ori, missing_mask_ori, keep_best=False)

        nll_imputation = self.latent_imputation_error(qz_x, X_ori, missing_mask, missing_mask_ori)

        if self.sampling:
            nll_sampling = self.sampling_error(qz_x, X_ori, missing_mask_ori)
        else:
            nll_sampling = torch.tensor(1.0, device=self.device)  # Ensure tensor is on the correct device

        alpha = self.p
        nll = nll_recon * (1 - alpha) + nll_imputation * alpha / 2 + nll_sampling * alpha / 2

        return nll, nll_recon, nll_imputation, nll_sampling

    def sample_selected_missing_vals(self, X, missing_mask, num_samples=1):
        """
        Impute missing values by selecting a specific unobserved feature for each point with
        observed values, and compute distances only with points that have the selected feature
        observed and share other observed features.
        """
        batch_size, seq_len, feature_size = X.shape
        X_flat = X.reshape(-1, feature_size)
        missing_mask_flat = missing_mask.reshape(-1, feature_size)
        X_recon = X_flat.clone().unsqueeze(2).repeat(1, 1, num_samples)
        # Ensure mask is on the correct device
        mask = torch.zeros(X_flat.shape, device=self.device)

        selected_indices = np.random.choice(np.arange(X_flat.shape[0]), size=100, replace=False)

        for i in selected_indices:
            observed_features = missing_mask_flat[i]

            # Check if there are any unobserved features
            unobserved_features = torch.where(~missing_mask_flat[i])[0]
            if len(unobserved_features) == 0:
                continue  # Skip if no unobserved features

            # Randomly select one unobserved feature j for this point
            feature_j = unobserved_features[torch.randint(len(unobserved_features), (1,))].item()

            # Find indices of points that have feature j observed and other features in common
            compatible_indices = torch.where(
                (missing_mask_flat[:, feature_j]) & (observed_features & missing_mask_flat).any(dim=1)
            )[0]

            if len(compatible_indices) > 0:
                # Get observed values for feature j from compatible points
                observed_values_j = X_flat[compatible_indices, feature_j]

                # Select features common between xi and compatible points
                xi = X_flat[i]
                compatible_points = X_flat[compatible_indices]
                distances = self.compute_selected_distance(xi, compatible_points, observed_features)

                # Sample from the closest compatible points based on distances
                temperature = 0.1
                probabilities = torch.softmax(-distances * temperature, dim=0)
                p = probabilities.detach().cpu().numpy().flatten()

                sampled_idx = torch.multinomial(probabilities, num_samples=num_samples).squeeze()
                X_recon[i, feature_j] = observed_values_j[sampled_idx]
                mask[i, feature_j] = 1

        return X_recon.reshape(batch_size, seq_len, feature_size, num_samples), mask.reshape(batch_size, seq_len, feature_size).bool()

    def compute_selected_distance(self, xi, compatible_points, observed_features):
        """
        Compute distances only for observed features in xi that are also observed in compatible_points.
        """
        common_features_mask = observed_features.unsqueeze(0) & (compatible_points != 0)
        diff = (xi - compatible_points) ** 2 * common_features_mask.float()
        distances = diff.sum(dim=1)

        # When no common features, replace missing distance by a reasonable quantile of the distance
        distances += ((1 - common_features_mask.float())).sum(dim=1) * 0.5

        return distances

    def compute_nll(self, px_z, X, mask, keep_best=False):
        """
        Compute the negative log-likelihood.
        """
        nll = (X - px_z.mean).pow(2)
        nll = torch.where(torch.isfinite(nll), nll, torch.zeros_like(nll))
        nll = torch.where(nll >= 0, nll, torch.zeros_like(nll))
        if mask is not None:
            nll = torch.where(mask, nll, torch.zeros_like(nll))
        if keep_best:
            a, b, c = nll.shape
            nll = nll.reshape(self.K * self.M, -1, b, c)  # Reshape so first axis contains K x M samples for a same observation
            nll = torch.min(nll, axis=0)[0]
            return nll.sum()
        else:
            # Use mask.numel() instead of creating a tensor from mask.shape
            scale_factor = mask.numel() / mask.sum()
            return (nll.sum(dim=(1, 2)) * scale_factor).mean()

    # Helpers and plotters
    def prepare_and_simulate(self, X, missing_mask):
        """Prepare data by repeating and simulate missing data."""
        X_ori = torch.clone(X.repeat(self.K * self.M, 1, 1))
        missing_mask_ori = missing_mask.repeat(self.K * self.M, 1, 1).type(torch.bool)
        X = mcar(X_ori, p=self.p)
        X, missing_mask = fill_and_get_mask_torch(X)
        return X_ori, missing_mask_ori, X, missing_mask.type(torch.bool)

    def validate_elbo(self, elbo, nll_recon, nll_imputation, kl, z, qz_x, X_ori, X, time_steps, px_z, tl):
        """Perform assertions, debugging, and optional plotting."""
        assert not (elbo.abs() > 1e6).any(), print('elbo too big', nll_recon.mean().item(), nll_imputation.mean().item(), kl.mean().item())
        assert not (elbo > 50), print('elbo negative', elbo.item(), nll_recon.mean().item(), nll_imputation.mean().item(), kl.mean().item())

        if len(self.loss_history['elbo']) % 100 == 0:
            self.plot_latent_series_and_reconstruction(z, px_z, qz_x, X_ori.detach().cpu(), X.detach().cpu(), self.latent_dim, time_steps, -elbo, kl, tl)

    def plot_losses(self):
        # Calculate the step size n to ensure we have a maximum of 500 points plotted
        max_points = 500
        num_points = len(self.loss_history['elbo'])
        n = max(1, num_points // max_points)  # Ensure n is at least 1

        # Create the iterations range with step size n
        iterations = range(1, num_points + 1)[::n]

        plt.figure(figsize=(10, 6))

        # Plot each loss component with downsampling
        plt.semilogy(iterations, self.loss_history['elbo'][::n], label='ELBO')
        plt.semilogy(iterations, self.loss_history['nll_recon'][::n], label='NLL Recon')
        plt.semilogy(iterations, self.loss_history['nll_imputation'][::n], label='NLL Imputation')
        plt.semilogy(iterations, self.loss_history['nll_sampling'][::n], label='NLL Sampling')
        plt.semilogy(iterations, self.loss_history['kl'][::n], label='KL Divergence')
        plt.semilogy(iterations, self.loss_history['temporal_loss'][::n], label='Temporal Loss')
        # plt.semilogy(iterations, self.loss_history['dependence_loss'][::n], label='Dependence Loss')

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

    def plot_latent_series_and_reconstruction(self, z, px_z, qz_x, X_ori, X, latent_dim, time_steps, nll, kl, tl, folder='latent_plots'):
        """
        Plots the mean and variance of all latent time series and the original vs reconstructed data.
        """
        # Convert to CPU numpy arrays for plotting
        z_mean = qz_x.mean[0].detach().cpu().numpy()
        z_var = qz_x.variance[0].detach().cpu().numpy()

        X_ori_np = torch.clone(X_ori).detach().cpu().numpy()
        X_np = X.detach().cpu().numpy()

        z_mean_ori = self.encode(X_ori).mean.detach().cpu()

        # Sample 10 times from the posterior to get 10 reconstructions
        reconstructions = []
        for i in range(10):
            z_sample = qz_x.rsample()  # Sample from the posterior
            px_z_sample = self.decode(z_sample)  # Reconstruct the data
            X_recon_sample = px_z_sample.mean.detach().cpu().numpy()  # Get the mean of the reconstruction
            # Set first half of non-reconstructed values to NaN
            num_missing_vals = np.sum(X_ori_np == 0)
            X_recon_sample[X_ori_np == 0][:num_missing_vals // 2] == np.nan
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

        plt.subplot(4, 1, 4)
        mask, mask_ori = (X != 0), (X_ori != 0)
        plt.plot(self.latent_imputation_error(qz_x, X_ori, mask, mask_ori, for_plotting=True)[0].detach().cpu())
        plt.title('Log probability of original z belonging to the corrupted Gaussian')

        losses = f'kl = {kl.mean().item()} - nll = {nll.mean().item()} - temporal {tl.item()}'
        plt.subplot(4, 1, 1)
        for dim in range(latent_dim):
            plt.plot(range(time_steps), z_mean[:, dim], label=f'Latent dim {dim} Mean')
            plt.fill_between(range(time_steps), z_mean[:, dim] - z_var[:, dim] ** 0.5, z_mean[:, dim] + z_var[:, dim] ** 0.5, alpha=0.2)
            plt.scatter(range(time_steps), z_mean_ori[0][:, dim], label=f'Latent dim {dim} Mean')

        plt.title('Latent Time Series (Mean and Variance) ' + losses)
        plt.xlabel('Time Steps')
        plt.ylabel('Latent Values')

        # Plot scales for prior and posterior
        plt.subplot(4, 1, 2)
        nb_missing_vals = (X_np[0] == 0).sum(axis=1)
        missing_ratio = (X_np[0] != 0).mean(axis=1)
        prior_scale = (1 - missing_ratio) ** 0.5
        plt.semilogy(z_var, alpha=0.5)
        plt.semilogy((z_mean - z_mean_ori[0].numpy()) ** 2, alpha=0.5)
        plt.semilogy(prior_scale, label='Prior scale')
        plt.semilogy(np.linalg.norm(z_var, axis=1), 'r:', label='Posterior scale')
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

        # Save the plot
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

def HSIC_loss(A, B, sigma=1.0):
    """
    Loss that computes the covariance between A and B.
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
    I = torch.eye(n, device=device)
    ones = torch.ones(n, n, device=device) / n
    H = I - ones
    return H

def compute_kl_divergence(qz_x, prior_variance=0.1):
    """
    Computes the KL divergence between the approximate posterior q(z|x)
    and the prior p(z) with a specified variance.
    """
    mu_q = qz_x.mean  # Mean of q(z|x)
    sigma_q2 = qz_x.variance  # Variance of q(z|x)

    # Prior parameters
    mu_p = torch.zeros_like(mu_q)  # Prior mean is zero
    sigma_p2 = prior_variance  # Prior variance

    # Compute the KL divergence components
    eps = 1e-8
    sigma_q2 = sigma_q2 + eps

    term1 = sigma_q2 / sigma_p2  # (σ_q^2) / (σ_p^2)
    term2 = (mu_q - mu_p).pow(2) / sigma_p2  # (μ_q - μ_p)^2 / (σ_p^2)
    term3 = -1  # -1
    term4 = torch.log(sigma_p2 / sigma_q2)  # ln(σ_p^2 / σ_q^2)

    # Sum over the latent dimensions
    kl_div = 0.5 * torch.sum(term1 + term2 + term3 + term4, dim=1)  # [batch_size]

    # Return the mean KL divergence over the batch
    return kl_div.mean()
