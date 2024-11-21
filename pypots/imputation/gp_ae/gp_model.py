from .kernels import *

# for probabilistic GP
import torch.nn.functional as F
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

import matplotlib.pyplot as plt

# For kernel parameters estimation
import gpytorch
from .kernels import SparseGPModel, BatchIndependentMultitaskGPModel, ExactGPModel

import numpy as np
import torch
from torch.utils.data import DataLoader

class ProbabilisticGP:
    """
    A sub-module of a VAE that corrects the latent time-series via the Probabilistic GP regression scheme
    """
    def __init__(self, AEmodel, assemble_data):
        self.encoder = AEmodel.encoder
        self.enforce_variance_bias = False
        self.latent_size = AEmodel.latent_dim
        self.gp_models = []
        self.likelihoods = []
        self.mll = []
        self.optimizer = []
        self.assemble_data = assemble_data

    def instantiate_gp_models(self, training_loader):
        self.gp_models = []
        self.likelihoods = []
        self.mll = []
        self.optimizer = []

        for j in range(self.latent_size):
            # Define inducing points for the sparse GP model (batch size 8 assumed here)
            #n_inducing_pts = 48
            #inducing_points = torch.linspace(0, n_inducing_pts - 1, n_inducing_pts).reshape(1, -1).repeat(8, 1)
            
            # Instantiate model and likelihood
            #gp_model = SparseGPModel(inducing_points=inducing_points)
            
            # Ensure noise aligns with the batch and time steps: noise [8, 48] for batched FixedNoiseGaussianLikelihood
            #likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
            #    noise=torch.ones(8, 48),  # Adjust to batch_size x num_time_steps
            #    noise_constraint=gpytorch.constraints.GreaterThan(1e-9)
            #)

            batch_training = False
            if batch_training:

                likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=8)

                n = 48
                train_x = torch.linspace(0,n-1,n).detach()
                train_y = train_x.reshape(1,n).repeat(8,1).detach()

                gp_model = BatchIndependentMultitaskGPModel(train_x, train_y, likelihood)

            else:
                    
                likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
                    noise=torch.ones(48)*.1,  # Adjust to batch_size x num_time_steps
                    noise_constraint=gpytorch.constraints.GreaterThan(1e-9))

                n = 48
                train_x = torch.linspace(0,n-1,n).detach()
                train_y = train_x.detach()

                gp_model = ExactGPModel(train_x, train_y, likelihood)


            if torch.cuda.is_available():
                gp_model = gp_model.cuda()
                likelihood = likelihood.cuda()

            gp_model.train()
            likelihood.train()

            # Define optimizer and marginal log-likelihood
            optimizer = torch.optim.Adam(gp_model.parameters(), lr=0.1)

            # Define MLL for the Exact Marginal Log Likelihood
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)

            # Append model and likelihood to lists
            self.gp_models.append(gp_model)
            self.likelihoods.append(likelihood)
            
            # Append optimizer and mll
            self.optimizer.append(optimizer)
            self.mll.append(mll)

# Main function using train_on_batch
    def fit_kernel(self, training_loader, training_iter=50, batch_iter=1):
        # Instantiate parameters
        self.instantiate_gp_models(training_loader)
        self.kernel_params = {j: [] for j in range(self.latent_size)}
        self.dims_to_train = list(np.arange(self.latent_size))

        for i in range(training_iter):

            while len(self.dims_to_train) > 0:

                for training_step, data in enumerate(training_loader):


                    inputs = self.assemble_data(data)
                    x_input = inputs['X']

                    # Encode the data
                    qz_x = self.encoder(x_input)
                    z_mu, z_var = qz_x.mean.detach(), qz_x.variance.detach()

                    # Train on the batch for batch_iter iterations
                    train_on_batch(
                        model=self,
                        gp_models=self.gp_models,
                        mlls=self.mll,
                        optimizers=self.optimizer,
                        batch_x=x_input,
                        z_mu=z_mu,
                        z_var=z_var,
                        n_iter=batch_iter
                    )

                    # Optional: Plotting kernel parameters
                    n_plot = 30
                    if training_step % n_plot == n_plot - 1:
                        for j in range(self.latent_size):

                            # Plot GP reconstruction for the first sample
                            plot_gp_reconstruction(
                                torch.arange(z_mu.size(1)).float(),
                                z_mu[0, :, j],
                                self.gp_models[j],
                                self.likelihoods[j],
                                j,
                                training_step
                            )

                        plt.figure()

                        for j in range(self.latent_size):
                            plt.semilogy(self.kernel_params[j])

                        plt.title(f'Kernel Parameter Progression for Dimension {j}')
                        plt.xlabel('Iteration')
                        plt.ylabel('Lengthscale')

                    
                        plt.savefig('latent_plots/gp_plots/kernels_params.png')
                        plt.close()
                        #plt.show()

            # Optionally print progress after each training iteration
            try:
                print(f"RBF Lengthscale: {self.gp_models[0].covar_module.base_kernel.kernels[0].lengthscale.item():.3f}")
                print(f"Periodic Lengthscale: {self.gp_models[0].covar_module.base_kernel.kernels[1].lengthscale.item():.3f}")
                print(f"Linear variance: {self.gp_models[0].covar_module.base_kernel.kernels[2].lengthscale.item():.3f}")
                print(f"Noise: {self.likelihoods[0].noise.item():.3f}")
            except:
                continue

    def select_values_for_GP_inference(self, z_mu, z_var, q = .8):
        """
        Returns a set of values above a certain quantile
        """
        x = torch.arange(len(z_mu))
        quantile = torch.quantile(z_var, q = q)
        mask = z_var > quantile
        return x[mask], z_mu[mask]

    def batch_select_values_for_GP_inference(self, z_mu, z_var, q = .8):
        """
        Returns a set of values above a certain quantile
        """

        k = 8
        X, Z = [], []
        x = torch.arange(z_mu.shape[1])

        for i in range(z_mu.shape[0]):
            # Sort the indices of z_var[i] in descending order and get top `k` indices
            top_k_indices = torch.topk(z_var[i], k, largest=True).indices

            # Select values of `x` and `z_mu` corresponding to these indices
            X.append(x[top_k_indices])
            Z.append(z_mu[i][top_k_indices])

        # Stack results to ensure same shape
        X_stacked = torch.stack(X)
        Z_stacked = torch.stack(Z)

        #print(X_stacked.shape, Z_stacked.shape)
        
        return X_stacked, Z_stacked

def train_on_batch(model, gp_models, mlls, optimizers, batch_x, z_mu, z_var, n_iter):
    """
    Train the GP models on a single batch for multiple iterations.
    
    Parameters:
    - model: Main model object (self).
    - gp_models: List of GP models for each latent dimension.
    - mlls: List of Marginal Log Likelihood objects for each latent dimension.
    - optimizers: List of optimizers for each latent dimension.
    - batch_x: Input data for the batch.
    - z_mu: Mean of the latent encoding for the batch.
    - z_var: Variance of the latent encoding for the batch.
    - dims_to_train: List of dimensions (indices) currently being trained.
    - n_iter: Number of iterations to run training for each batch.
    """
    batch_size, num_points, latent_size = z_mu.size()
    
    for _ in range(n_iter):
        # Loop over each dimension to train
        for j in model.dims_to_train:
            cumulative_loss = 0.0  # Reset cumulative loss for dimension j

            # Loop over each sample in the batch
            for b in range(batch_size):
                # Create time step positions for sample `b`
                x_positions = torch.arange(num_points).float().detach()
                
                # Extract z_mu and z_var for sample `b` and dimension `j`
                z_mu_b_j = z_mu[b, :, j].detach()
                z_var_b_j = z_var[b, :, j].detach()

                model.likelihoods[j].noise = z_var_b_j.detach() / (z_var_b_j.mean().detach() * 100)
                
                # Update GP model’s training data for dimension `j`
                gp_models[j].set_train_data(inputs=x_positions, targets=z_mu_b_j, strict=False)
                
                # Compute GP output and loss for dimension `j` and sample `b`
                output = gp_models[j](x_positions)
                loss = -mlls[j](output, z_mu_b_j)
                cumulative_loss += loss
            
            # Backward pass and optimizer step for dimension `j`
            cumulative_loss.backward(retain_graph=True)
            optimizers[j].step()
            optimizers[j].zero_grad()
            
            # Update kernel parameters
            model.kernel_params[j].append(
                gp_models[j].covar_module.base_kernel.lengthscale.item()
            )

            # Early stopping for dimension `j`
            criterion = np.std(model.kernel_params[j][-20:]) < 1e-4
            if (
                len(model.kernel_params[j]) > 30
                and criterion
            ):
                model.dims_to_train.remove(j)
                print(f'No further training for dim {j}')

def plot_gp_reconstruction(x, z_mu_j, gp_model_j, likelihood_j, j, training_step):
    # Set models to evaluation mode
    gp_model_j.eval()
    likelihood_j.eval()
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # Ensure x is a tensor of shape [num_points]
        test_x = x.float()
        
        # Make predictions
        predictions = likelihood_j(gp_model_j(test_x))
        try:
            mean = predictions.mean[:,0]
        except:
            mean = predictions.mean
        lower, upper = predictions.confidence_region()
    
    # Plotting
    plt.figure(figsize=(8, 4))
    # Plot observed data as black stars
    plt.plot(x.numpy(), z_mu_j.detach().numpy(), 'k*', label='Observed Data')
    # Plot predictive mean as blue line
    plt.plot(test_x.numpy(), mean.numpy(), 'b', label='Predictive Mean')
    # Shade in confidence intervals
    try:
        plt.fill_between(test_x.numpy(), lower.numpy()[:,0], upper.numpy()[:,0], alpha=0.5, label='Confidence Interval')
    except:
        plt.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5, label='Confidence Interval')
    plt.ylim([z_mu_j.min().item() - 1, z_mu_j.max().item() + 1])
    plt.legend()
    plt.title(f'GP Reconstruction for Latent Dimension {j} at Step {training_step}')
    plt.xlabel('Time Steps')
    plt.ylabel('Latent Variable Value')

    plt.savefig(f'latent_plots/gp_plots/dim_{j}_iter_{training_step}.png')
    plt.close()
    #plt.show()
    
    # Set models back to training mode
    gp_model_j.train()
    likelihood_j.train()