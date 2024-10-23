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
        alpha = .1 #self.temperature * .5
        nll = nll_recon * (1 - alpha) + nll_imputation * alpha

        ## Compute KL divergence between q(z|x) and p(z) : Here the imposed prior is only on the variance

        # Compute the number of missing values per sample per time step
        missing_ratio = missing_mask.float().mean(dim=2)  # Shape: (batch_size * K * M, time_steps)
        prior_scale = (1 - missing_ratio) # this will be changed to something smarter later on

        # The prior on the variance forces the  mean variance to be proportional to the amount of missing
        # data for the observed point
        kl = ( torch.mean(qz_x.variance, dim = 2) - prior_scale).pow(2)
        kl = kl.sum(1).mean()

        ## Compute a loss based on a Gaussian process prior between 1 point and the next
        # basically z_{t+1} ~N(z_t, sigma^2)
        sigma = qz_x.variance.mean().pow(.5) * 5 # this will also be changed to something smarter
        temporal_loss = (z[:,1:] - z[:,:-1]).pow(2).sum(axis=-1).mean() / sigma**2

        # get final elbo
        elbo = -nll - self.beta * kl - temporal_loss
        elbo = elbo.mean()

        # Validation and optional plotting
        self.validate_elbo(elbo, nll_recon, nll_imputation, kl, z, qz_x, X_ori, X, time_steps, px_z, temporal_loss)

        return -elbo

    def prepare_and_simulate(self, X, missing_mask):
        """Prepare data by repeating and simulate missing data."""
        X_ori = X.repeat(self.K * self.M, 1, 1) #augment by K x M if we want to sample multiple times in the latent space (if not K = M = 1)
        missing_mask_ori = missing_mask.repeat(self.K * self.M, 1, 1).type(torch.bool) 
        X = mcar(X_ori, p=.3) #missing completely at random, missingness proba = 0.3
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

        if torch.rand(1) < 0.001: # plot randomly with proba 1/1000 je sais c'est degueu mais j'aime bien
            self.plot_latent_series_and_reconstruction(z, px_z, qz_x, X_ori.detach(), X.detach(), self.latent_dim, time_steps, -elbo, kl, tl)