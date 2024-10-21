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
        plt.plot(prior_scale, label='Prior scale')
        plt.plot(np.linalg.norm(z_var, axis=1), label='Posterior scale')
        plt.legend()

        # Plot original vs reconstructed data for all 10 reconstructions
        plt.subplot(4, 1, 3)
        for i, X_recon_sample in enumerate(reconstructions):
            plt.plot(range(time_steps), X_recon_sample[0, :, :], label=f'Reconstructed {i+1}', alpha=0.6)
        
        plt.gca().set_prop_cycle(None)
        plt.plot(range(time_steps), X_np[0, :, :], 'o', label='Data with missing Data')
        plt.gca().set_prop_cycle(None)
        plt.plot(range(time_steps), X_ori_np[0, :, :], '+', label='Original Data')
        plt.title('Original vs Reconstructed Data (10 Samples)')
        plt.xlabel('Time Steps')
        plt.ylabel('Feature Values')
        plt.legend()

        # Save the plot
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

    # Modified forward function
    def forward(self, X, missing_mask):
        batch_size, time_steps, _ = X.size()
        X_ori = X.repeat(self.K * self.M, 1, 1)
        missing_mask_ori = missing_mask.repeat(self.K * self.M, 1, 1).type(torch.bool)

        X = mcar(X_ori, p=.3)
        X, missing_mask = fill_and_get_mask_torch(X)
        missing_mask = (X != 0).type(torch.bool)

        indicating_mask = (missing_mask_ori.float() - missing_mask.float()).to(torch.bool)

        # Encode input to get approximate posterior q(z|x)
        qz_x = self.encode(X)
        z = qz_x.rsample()

        # Decode to get likelihood p(x|z)
        px_z = self.decode(z)

        # Negative log-likelihood (reconstruction loss)
        nll_recon = -px_z.log_prob(X)
        nll_recon = torch.where(torch.isfinite(nll_recon), nll_recon, torch.zeros_like(nll_recon))
        if missing_mask is not None:
            nll_recon = torch.where(missing_mask, nll_recon, torch.zeros_like(nll_recon))
        nll_recon = nll_recon.sum(dim=(1, 2))

        # Negative log-likelihood (reconstruction loss) for imputation
        nll_imputation = -px_z.log_prob(X_ori)
        nll_imputation = torch.where(torch.isfinite(nll_imputation), nll_imputation, torch.zeros_like(nll_imputation))
        if missing_mask is not None:
            nll_imputation = torch.where(indicating_mask, nll_imputation, torch.zeros_like(nll_imputation))
        nll_imputation = nll_imputation.sum(dim=(1, 2))

        # get final nll
        #alpha = self.temperature * .5
        alpha = .1
        nll = nll_recon * (1 - alpha) + nll_imputation * alpha

        # Compute the number of missing values per sample per time step
        missing_ratio = missing_mask.float().mean(dim=2)  # Shape: (batch_size * K * M, time_steps)

        # Avoid zero variance by adding a small epsilon
        epsilon = 1e-3
        prior_scale = (1 - missing_ratio).pow(0.5).unsqueeze(-1) + epsilon  # Shape: (batch_size * K * M, time_steps, 1)
        if prior_scale.min() < .4:
            print(prior_scale.min(), prior_scale.max())
        prior_scale = prior_scale.expand(-1, -1, self.latent_dim)  # Expand to latent_dim
        #prior_loc = torch.zeros_like(prior_scale)  # Zero mean
        prior_loc = qz_x.mean.detach() #no prior on the mean

        # Create prior distribution with adjusted variance
        prior = Independent(Normal(prior_loc, prior_scale), 1)  # Independent over latent dimensions

        #print(prior.variance)
        #print(qz_x.variance)

        # Compute KL divergence between q(z|x) and p(z)
        
        kl = torch.abs( torch.norm(qz_x.variance, dim = 2) - prior_loc[:,:,0])
        kl = kl.sum(1).mean()

        elbo = -nll - self.beta * kl
        elbo = elbo.mean()

        ## add temporal loss
        temporal_loss = (z[:,1:] - z[:,:-1]).abs().mean()
        #elbo = elbo - .01*temporal_loss

        if elbo > 50:
            print((-nll - self.beta * kl).mean().item())
            print(nll.mean().item(),kl.mean().item())
            print(self.temperature, alpha)


        assert not (elbo.abs() > 1e6).any(), print( 'elbo too big ', nll_recon.mean().item(), nll_imputation.mean().item(), kl.mean().item(), temporal_loss.item(), qz_x.variance.mean())
        assert not (elbo > 50), print( 'elbo negative ', elbo.item(), nll_recon.mean().item(), nll_imputation.mean().item(), kl.mean().item(), temporal_loss.item(), qz_x.variance.mean())
        assert not (torch.isnan(elbo).any()), print( 'elbo is nan ', elbo.item(), nll_recon.mean().item(), nll_imputation.mean().item(), kl.mean().item(), temporal_loss.item(), qz_x.variance.mean())


        # Occasionally plot the latent series mean and variance and reconstruction
        if torch.rand(1) < 0.001:
            #print('losses', nll.mean().item(), kl.mean().item(), temporal_loss.item())
            #print(nll_recon.mean().item(), nll_imputation.mean().item())
            self.plot_latent_series_and_reconstruction(z, px_z, qz_x, X_ori.detach(), X.detach(), self.latent_dim, time_steps, nll, kl, temporal_loss)

        return -elbo