import torch
import torch.nn as nn
from torch.distributions import Normal, Independent
from .layers import (
    SimpleVAEEncoder,
    SimpleVAEDecoder,
)

class BackboneGP_VAE(nn.Module):
    """Modified GPVAE model with prior variance proportional to missing values.

    Parameters
    ----------
    [Same as before, but kernel-related parameters can be ignored or set to None]
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
        self.encoder = SimpleVAEEncoder(input_dim, latent_dim, encoder_sizes, window_size)
        self.decoder = SimpleVAEDecoder(latent_dim, input_dim, decoder_sizes)
        self.M = M
        self.K = K

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    @staticmethod
    def kl_divergence(q_dist, p_dist):
        return torch.distributions.kl_divergence(q_dist, p_dist)

    def forward(self, X, missing_mask):
        batch_size, time_steps, _ = X.size()
        X = X.repeat(self.K * self.M, 1, 1)
        missing_mask = missing_mask.repeat(self.K * self.M, 1, 1).type(torch.bool)

        # Encode input to get approximate posterior q(z|x)
        qz_x = self.encode(X)
        z = qz_x.rsample()

        # Decode to get likelihood p(x|z)
        px_z = self.decode(z)

        # Negative log-likelihood (reconstruction loss)
        nll = -px_z.log_prob(X)
        nll = torch.where(torch.isfinite(nll), nll, torch.zeros_like(nll))
        if missing_mask is not None:
            # Mask out missing data
            nll = torch.where(missing_mask.any(dim=2), nll, torch.zeros_like(nll))
        nll = nll.sum(dim=1)  # Sum over time steps

        # Compute the number of missing values per sample per time step
        missing_count = missing_mask.sum(dim=2).float()  # Shape: (batch_size * K * M, time_steps)

        # Avoid zero variance by adding a small epsilon
        epsilon = 1e-6
        prior_scale = missing_count.unsqueeze(-1) + epsilon  # Shape: (batch_size * K * M, time_steps, 1)
        prior_scale = prior_scale.expand(-1, -1, self.latent_dim)  # Expand to latent_dim
        prior_loc = torch.zeros_like(prior_scale)  # Zero mean

        # Create prior distribution with adjusted variance
        prior = Independent(Normal(prior_loc, prior_scale), 1)  # Independent over latent dimensions

        # Compute KL divergence between q(z|x) and p(z)
        kl = self.kl_divergence(qz_x, prior)
        kl = torch.where(torch.isfinite(kl), kl, torch.zeros_like(kl))
        kl = kl.sum(dim=1)  # Sum over time steps

        # ELBO Loss
        elbo = -nll - self.beta * kl
        elbo = elbo.mean()

        return -elbo
