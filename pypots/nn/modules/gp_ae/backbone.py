import torch
import torch.nn as nn
from torch.distributions import Normal, Independent
from .layers import (
    SimpleVAEEncoder,
    SimpleVAEDecoder,
    GpvaeEncoder, #modified versinos
    GpvaeDecoder, #modified versions
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
        self.gamma = .01
        self.encoder = GpvaeEncoder(input_dim, latent_dim, encoder_sizes)
        #self.encoder = FactorNet(input_dim, latent_dim, encoder_sizes)
        self.decoder = GpvaeDecoder(latent_dim, input_dim, decoder_sizes)
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

        self.p = .5

        self.sampling = False
        self.train_decoder = True

    # for tracking
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

    def encode(self, x, missing_mask = None):
        return self.encoder(x, missing_mask)

    def decode(self, z):
        return self.decoder(z)

    @staticmethod
    def kl_divergence(q_dist, p_dist):
        return torch.distributions.kl_divergence(q_dist, p_dist)

    def temporal_loss(self, z, X, missing_mask, batch_size, eps = 1e-3):

        temporal_loss = 0

        for i in range(batch_size):

            mask_diff = (missing_mask[i::batch_size,1:] * missing_mask[i::batch_size,:-1])
            mask_diff_sum = mask_diff.sum(2)
            mask_diff_sum[mask_diff_sum==0] = 1
            X_diff = ((X[i::batch_size,1:] - X[i::batch_size,:-1])*mask_diff).pow(2).sum(2) / mask_diff_sum
            z_diff = (z[i::batch_size,1:] - z[i::batch_size,:-1]).pow(2).sum(2)
            temporal_loss += z_diff / (X_diff + eps)

            #print(X[missing_mask], 'oo')

            #print(X_diff[mask_diff], X[1:][mask_diff])
            #print(X[mask_diff], )

            #print(temporal_loss)
            #print(X_diff)
            if torch.isnan(temporal_loss).any():
                print(mask_diff.sum(2))
                print(temporal_loss[temporal_loss!=temporal_loss], (z_diff / (X_diff + eps))[temporal_loss!=temporal_loss])
                print(z_diff[temporal_loss!=temporal_loss], X_diff[temporal_loss!=temporal_loss])

            assert not torch.isnan(temporal_loss).any(), print(temporal_loss[temporal_loss!=temporal_loss], (z_diff / (X_diff + eps))[temporal_loss!=temporal_loss])

            assert not torch.isnan(temporal_loss).any(), print(z_diff[temporal_loss!=temporal_loss], z_diff[temporal_loss!=temporal_loss])

        #print(temporal_loss.mean())

        return temporal_loss.mean()            

    # Modified forward function
    def forward(self, X, missing_mask):

        #t = time.time()

        batch_size, time_steps, _ = X.size()
        # Merge prepare_data and simulate_missing_data
        X_ori, missing_mask_ori, X, missing_mask = self.prepare_and_simulate(X, missing_mask)
        
        # compute masks
        missing_mask = (X!=0)
        missing_mask_ori = (X_ori!=0)
        # get indicating mask of where data has been artificially removed
        indicating_mask = (missing_mask_ori.float() - missing_mask.float()).to(torch.bool)

        # Encode input to get approximate posterior q(z|x) and sample from it
        qz_x = self.encode(X, missing_mask)
        z = qz_x.rsample()

        # Decode to get likelihood p(x|z)
        px_z = self.decode(z)

        # compute reconstruction error
        nll, nll_recon, nll_imputation, nll_sampling = self.reconstruction_error(qz_x, px_z, X, X_ori, missing_mask, missing_mask_ori, indicating_mask)

        ## Compute KL divergence between q(z|x) and p(z) : Here the imposed prior is only on the variance

        kl = qz_x.mean.pow(2).sum(2).pow(.5).mean()
        kl = kl.sum()

        #kl += -torch.log(qz_x.variance.abs().mean())/50

        ## Compute a loss based on a Gaussian process prior between 1 point and the next
        temporal_loss = self.temporal_loss(z, X, missing_mask, batch_size)

        # get dependence loss between variance and missingness patterns
        sigma = np.exp(np.random.uniform(-8,5))
        dependence_loss, sigma = HSIC_loss(qz_x.variance[::(self.K * self.M)], missing_mask[::(self.K * self.M)])

        # get final elbo
        elbo = -nll - self.beta * kl - temporal_loss * self.gamma  #- dependence_loss * 10000
        elbo = elbo.mean()

        #print('time end', time.time() - t)
        #t = time.time()

        self.loss_history['elbo'].append(-1*elbo.item())
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

        #qz_x_normalized = torch.distributions.Normal(loc = qz_x.mean, scale = qz_x.stddev / qz_x_ori.stddev.detach())

        #qz_x.variance = qz_x.variance / qz_x_ori.variance.detach() ##normalize by other variance !!

        loss = (qz_x.mean.detach() - qz_x_ori.mean.detach()).pow(2) / (qz_x.variance / qz_x_ori.variance.detach())

        #loss = -qz_x_normalized.log_prob(qz_x_ori.mean.detach())
        mask_imputed_coords = (missing_mask != missing_mask_ori).sum(axis=2)!=0
        #loss = torch.clamp(loss, min = -10)
        
        if not for_plotting:
            loss = loss[mask_imputed_coords]
        else:
            print('mean of mask of imputed coords :', mask_imputed_coords.float().mean())
        #loss = -qz_x.log_prob(qz_x_ori.mean.detach())

        ## ponderer la loss par le nombre de valeurs manquantes

        #nb_missing_vals = missing_mask.sum(axis=2).unsqueeze(2)
        #print(nb_missing_vals.shape, missing_mask.shape, loss.shape)
        #loss = loss * nb_missing_vals / nb_missing_vals
        #loss = loss * len(loss)
        #print(loss.mean(), loss.shape)


        #loss = loss * loss.abs() # make square

        #print(loss.mean(), qz_x.variance.mean())

        return loss * len(mask_imputed_coords.flatten()) * 100 #* len(loss) * .001 #/100 #normalization trick just to compensate for la ponderation par le nb de vals manquantes

    def latent_sampling_error(self, qz_x, X_sampled):
        """
        This error forces the AE to learn representation of missing values
        X_sampled (num_samples, batch_size, n_observations, n_dimensions)
        """

        Z_sampled = []
        loss = 0
        # For each sample, compute the log prob of its embedding mean belonging to qz_x
        for x_sampled in X_sampled:
            qz_x_sampled = self.encode(x_sampled)
            #z_sampled = qz_x_sampled.rsample().detach()
            #print(z_sampled, qz_x.mean)
            #loss -= qz_x.log_prob(z_sampled) 

            loss -= qz_x_sampled.log_prob(qz_x.rsample().detach()) #.mean.detach()

            #print(loss, qz_x.mean.flatten().shape)

            #print(loss.mean())

        return 100 * loss/len(qz_x.mean.flatten())


    def sampling_error(self, qz_x, X_ori, missing_mask_ori):
        """
        Forces the 
        """

        # Reconstruct 
        num_samples = 10
        X_sampled, mask_sampled = self.sample_selected_missing_vals(X_ori, missing_mask_ori, num_samples = num_samples)
        # permute dimensions to have the first dimension being the number of samples
        X_sampled = torch.permute(X_sampled, (3, 0, 1, 2)) 
        loss = self.latent_sampling_error(qz_x, X_sampled)

        return loss.mean() #sum(axis = 1)


    def reconstruction_error(self, qz_x, px_z, X, X_ori, missing_mask, missing_mask_ori, indicating_mask):

        # Negative log-likelihood
        nll_recon = self.compute_nll(px_z, X_ori, missing_mask_ori, keep_best = False)

        nll_imputation = self.latent_imputation_error(qz_x, X_ori, missing_mask, missing_mask_ori)

        if self.sampling and False:
            num_samples = 1
            X_sampled, mask_sampled = self.sample_selected_missing_vals(X_ori, missing_mask_ori, num_samples = num_samples)
            nll_sampling = 0
            for i in range(num_samples):
                nll_sampling += self.compute_nll(px_z, X_sampled[:,:,:,i], mask_sampled) 
        if self.sampling:
            nll_sampling = self.sampling_error(qz_x, X_ori, missing_mask_ori)
        else:
            nll_sampling = torch.tensor(1).float()

        #if not self.train_decoder:
        #    self.p = self.temperature * .5
        #self.p = .3

        alpha = self.p
        nll = nll_recon * (1 - alpha) + nll_imputation * alpha/2 + nll_sampling * alpha/2

        return nll, nll_recon, nll_imputation, nll_sampling

    def sample_selected_missing_vals(self, X, missing_mask, num_samples=1):
        """
        Impute missing values by selecting a specific unobserved feature for each point with
        observed values, and compute distances only with points that have the selected feature
        observed and share other observed features.

        Args:
            X (Tensor): Input tensor of shape [batch_size, seq_len, feature_size]
            missing_mask (Tensor): Mask indicating missing values in X
            num_samples (int): Number of similar observed points to sample from for imputation
        
        Returns:
            Tensor: Imputed tensor X with selected missing values filled
        """

        batch_size, seq_len, feature_size = X.shape
        X_flat = X.reshape(-1, feature_size)
        missing_mask_flat = missing_mask.reshape(-1, feature_size)
        X_recon = X_flat.clone().unsqueeze(2).repeat(1,1,num_samples)
        mask = torch.zeros(X_flat.shape)


        selected_indices = np.random.choice(np.arange(X_flat.shape[0]), size = 100, replace = False)

        #s = time.time()

        for i in selected_indices:
            #print(time.time() - s)
            #s = time.time()
            observed_features = missing_mask_flat[i]

            #print(missing_mask_flat[:2])
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

                #print(observed_values_j)
                #print(X_flat[~compatible_indices, feature_j])

                # Select features common between xi and compatible points
                xi = X_flat[i]
                compatible_points = X_flat[compatible_indices]
                distances = self.compute_selected_distance(xi, compatible_points, observed_features)
                
                # Sample from the closest compatible points based on distances
                temperature = .1
                probabilities = torch.softmax(-distances*temperature, dim=0)
                p = probabilities.detach().numpy().flatten()

                if False:

                    sorted = np.argsort(probabilities)
                    samples = np.random.choice(np.arange(len(p))[sorted], size = 100000, p = p)
                    plt.hist(samples, bins = len(p))

                    plt.show()
                    plt.plot(probabilities[sorted])
                    plt.show()
                    #sjkomzkz

                    sampled_idxs = np.random.choice(np.arange(len(p)), size = 100000, p = p)
                    plt.hist(X_recon[sampled_idxs][:,feature_j].detach())
                    plt.show()

                    sjkls
                
                sampled_idx = torch.multinomial(probabilities, num_samples=num_samples).squeeze()
                X_recon[i, feature_j] = observed_values_j[sampled_idx]
                mask[i, feature_j] = 1

                #plt.hist(X_recon[i, feature_j].detach())
                #plt.show()


        return X_recon.reshape(batch_size, seq_len, feature_size, num_samples), mask.reshape(batch_size, seq_len, feature_size).bool()

    def compute_selected_distance(self, xi, compatible_points, observed_features):
        """
        Compute distances only for observed features in xi that are also observed in compatible_points.
        """
        common_features_mask = observed_features.unsqueeze(0) & (compatible_points != 0)
        diff = (xi - compatible_points) ** 2 * common_features_mask.float()
        distances = diff.sum(dim=1)
        n_feats = common_features_mask.shape[1]

        # when no common features, replace missing distance by a reasonable quantile of the distance
        compute_true_quantiles = False
        if compute_true_quantiles:
            quantiles = torch.ones(1,n_feats)
            for j in range(n_feats):
                vals = diff[:,j][common_features_mask[:,j]]
                q = torch.quantile(input = vals, q = .6, interpolation = 'higher')
                #plt.plot(np.sort(vals.detach().numpy()))
                #plt.axhline(q)
                #plt.show()
                quantiles[0,j] = q
            distances += ((1 - common_features_mask.float()) * quantiles).sum(dim=1)
        else:
            #quantiles = torch.ones(1,n_feats) * .5
            distances += ((1 - common_features_mask.float())).sum(dim=1) * .5

        plot = False
        if plot:

            plt.imshow(common_features_mask.detach()[:50].T)
            plt.show()
            plt.imshow((xi - compatible_points)[:50].detach().T ** 2)
            plt.show()
            print(common_features_mask)
            plt.imshow((common_features_mask.float() + diff).detach()[:50].T)
            plt.show()

            #fig, ax = plt.subplots(2, sharex = True)
            plt.imshow((diff + ((1 - common_features_mask.float()) * quantiles)).detach()[:50].T)
            #distances = distances.masked_fill((distances == 0), 1e8)  # Assign large distances for no common features
            plt.show()
            plt.plot(distances.detach()[:50])
            plt.plot(common_features_mask.float().sum(axis=1).detach()[:50],'o')
            plt.show()

            sorting = np.argsort(distances.detach().numpy())
            plt.plot(distances[sorting], label = 'sorted distances')
            plt.plot(common_features_mask.float().sum(axis=1).detach()[sorting],'o', label = 'number of common labels')
            plt.legend()
            plt.show()

        return distances

    def compute_nll(self, px_z, X, mask, keep_best = False):
        """
        Compute the negative log-likelihood. If we are dealing with imputation we set keep_best is true, and we only get 
        the loss on the closest element within each of the K x M samples for each observation 
        -> I need to explicit the math for this but this helps to keep a diversity within the guesses but still
        drives the imputation towards the right answer
        """
        #nll = -px_z.log_prob(X)
        nll = (X - px_z.mean).pow(2)
        nll = torch.where(torch.isfinite(nll), nll, torch.zeros_like(nll))
        nll = torch.where(nll >= 0, nll, torch.zeros_like(nll))
        if mask is not None:
            nll = torch.where(mask, nll, torch.zeros_like(nll))
        if keep_best: # only keep ebst approx
            #print('keep best')
            #print(torch.where(torch.isfinite(nll), nll, torch.zeros_like(nll)))
            a,b,c = nll.shape
            nll = nll.reshape(self.K * self.M, -1, b, c) #reshape so first axis containts K x M samples for a same observation
            nll = torch.min(nll, axis = 0)[0]
            #print('nll')
            #print(nll)
            return nll.sum()
        else:
            return (torch.tensor(mask.shape).prod() * nll.sum(dim=(1, 2)) / mask.sum()).mean()

    def compute_nll_(self, px_z, X, mask, keep_best = False):

        mse = (X - px_z.mean).pow(2)
        return mse[mask].float().mean()

# helpers and plotters

    def prepare_and_simulate(self, X, missing_mask):
        """Prepare data by repeating and simulate missing data."""
        X_ori = torch.clone(X.repeat(self.K * self.M, 1, 1)) #augment by K x M if we want to sample multiple times in the latent space (if not K = M = 1)
        missing_mask_ori = missing_mask.repeat(self.K * self.M, 1, 1).type(torch.bool) 
        X = mcar(X_ori, p=self.p) #missing completely at random, missingness proba = 0.3
        X, missing_mask = fill_and_get_mask_torch(X) 
        #missing_mask = (X != 0).type(torch.bool) # just to be sure this is what the missing mask returns
        return X_ori, missing_mask_ori, X, missing_mask.type(torch.bool)


    def validate_elbo(self, elbo, nll_recon, nll_imputation, kl, z, qz_x, X_ori, X, time_steps, px_z, tl):
        """Perform assertions, debugging, and optional plotting."""
        assert not (elbo.abs() > 1e6).any(), print('elbo too big', nll_recon.mean().item(), nll_imputation.mean().item(), kl.mean().item())
        assert not (elbo > 50), print('elbo negative', elbo.item(), nll_recon.mean().item(), nll_imputation.mean().item(), kl.mean().item())

        
        if len(self.loss_history['elbo']) % 100 == 0: # plot randomly with proba 1/1000 je sais c'est degueu mais j'aime bien
            self.plot_latent_series_and_reconstruction(z, px_z, qz_x, X_ori.detach(), X.detach(), self.latent_dim, time_steps, -elbo, kl, tl)

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
        #plt.semilogy(iterations, self.loss_history['dependence_loss'][::n], label='Dependence Loss')

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
        z_mean = qz_x.mean[0].detach().cpu().numpy()
        z_var = qz_x.variance[0].detach().cpu().numpy()

        X_ori_np = torch.clone(X_ori).detach().cpu().numpy()
        X_np = X.detach().cpu().numpy()

        z_mean_ori = self.encode(X_ori).mean.detach()

        # Sample 10 times from the posterior to get 10 reconstructions
        reconstructions = []
        for i in range(10):
            z_sample = qz_x.rsample()  # Sample from the posterior
            px_z_sample = self.decode(z_sample)  # Reconstruct the data
            X_recon_sample = px_z_sample.mean.detach().cpu().numpy()  # Get the mean of the reconstruction
            # set first half of non reconstructed values to nan
            num_missing_vals = np.sum(X_ori_np == 0)
            X_recon_sample[X_ori_np == 0][:num_missing_vals//2] == np.nan
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
        log_prob = qz_x
        #plt.plot(qz_x.log_prob(z_mean_ori)[0].detach())
        mask, mask_ori = (X!=0), (X_ori!=0)
        plt.plot(self.latent_imputation_error(qz_x, X_ori, mask, mask_ori, for_plotting=True)[0].detach())

        plt.title('Log proba of z original belonging to the z corrupted gaussian')

        losses = f'kl = {kl.mean().item()} - nll = {nll.mean().item()} - temporal {tl.item()}'
        plt.subplot(4, 1, 1)
        for dim in range(latent_dim):
            plt.plot(range(time_steps), z_mean[:, dim], label=f'Latent dim {dim} Mean')
            plt.fill_between(range(time_steps), z_mean[:, dim] - z_var[:, dim]**.5, z_mean[:, dim] + z_var[:, dim]**.5, alpha=0.2)
            plt.scatter(range(time_steps), z_mean_ori[0][:, dim], label=f'Latent dim {dim} Mean')

        plt.title('Latent Time Series (Mean and Variance) ' + losses)
        plt.xlabel('Time Steps')
        plt.ylabel('Latent Values')

        # Plot scales for prior and posterior
        plt.subplot(4, 1, 2)
        nb_missing_vals = (X_np[0] == 0).sum(axis=1)
        missing_ratio = (X_np[0] != 0).mean(axis=1)
        prior_scale = (1 - missing_ratio) ** 0.5
        plt.semilogy(z_var, alpha = .5)
        # plt errors
        plt.semilogy((z_mean - z_mean_ori[0].numpy())**2, alpha = .5)

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
        #plt.legend()

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

def compute_kl_divergence(qz_x, prior_variance=0.1):
    """
    Computes the KL divergence between the approximate posterior q(z|x)
    and the prior p(z) with a specified variance.
    """
    # qz_x.mean and qz_x.variance should be tensors of shape [batch_size, latent_dim]

    mu_q = qz_x.mean  # Mean of q(z|x)
    sigma_q2 = qz_x.variance  # Variance of q(z|x); ensure this is variance, not std dev

    # Prior parameters
    mu_p = torch.zeros_like(mu_q)  # Prior mean is zero
    sigma_p2 = prior_variance  # Prior variance is 0.1

    # Compute the KL divergence components
    # Avoid division by zero by adding a small epsilon if necessary
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