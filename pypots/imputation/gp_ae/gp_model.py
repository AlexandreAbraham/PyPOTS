from .kernels import *

# for probabilistic GP
import torch.nn.functional as F
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

import matplotlib.pyplot as plt

# For kernel parameters estimation
import gpytorch
from .kernels import SparseGPModel

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
            n_inducing_pts = 48
            inducing_points = torch.linspace(0, n_inducing_pts - 1, n_inducing_pts).reshape(1, -1).repeat(8, 1)
            
            # Instantiate model and likelihood
            gp_model = SparseGPModel(inducing_points=inducing_points)
            
            # Ensure noise aligns with the batch and time steps: noise [8, 48] for batched FixedNoiseGaussianLikelihood
            likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
                noise=torch.ones(8, 48),  # Adjust to batch_size x num_time_steps
                noise_constraint=gpytorch.constraints.GreaterThan(1e-9)
            )

            if torch.cuda.is_available():
                gp_model = gp_model.cuda()
                likelihood = likelihood.cuda()

            gp_model.train()
            likelihood.train()

            # Append model and likelihood to lists
            self.gp_models.append(gp_model)
            self.likelihoods.append(likelihood)

            # Define optimizer and marginal log-likelihood
            optimizer = torch.optim.Adam([
                {'params': gp_model.parameters()},
                {'params': likelihood.parameters()}
            ], lr=0.1)

            # Define MLL for the Exact Marginal Log Likelihood
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)
            
            # Append optimizer and mll
            self.optimizer.append(optimizer)
            self.mll.append(mll)


    def fit_kernel(self, training_loader, training_iter = 50):

        # instantiate parameters
        self.instantiate_gp_models(training_loader)
        self.kernel_params = {j:[] for j in range(self.latent_size)}
        dims_to_train = list(np.arange(self.latent_size))

        for i in range(training_iter):
            # Zero grad optimizers
            for optimizer in self.optimizer:
                optimizer.zero_grad()

            # Start training loop
            for training_step, data in enumerate(training_loader):
                inputs = self.assemble_data(data)
                x = inputs['X']

                # Encode the data
                qz_x = self.encoder(x)
                z_mu, z_var = qz_x.mean.detach(), qz_x.variance.detach()   

                # create position var
                x = torch.arange(z_mu.shape[1]).repeat(z_mu.shape[0],1).detach() # time steps

                #print(dims_to_train)
                for j in dims_to_train:

                    cumulative_loss = 0

                    print(z_var[:,:,j].shape)
                    self.likelihoods[j].noise = z_var[:,:,j].clone()
                    out = self.gp_models[j](x)
                    #help(out)
                    print(out.covariance_matrix.shape, z_mu[:,:,j].shape)
                    loss = -self.mll[j](out, z_mu[:,:,j])

                    # this is just a first test but if the kernel parameters don't change any more we can stop learning them for each latent dim
                    if len(self.kernel_params[j]) > 30 and np.abs(self.kernel_params[j][-1] - np.array(self.kernel_params[j])[-10:-2].mean()) < 1e-3:
                        dims_to_train.remove(j)
                        print(f'No further training for dim {j}')

                    # Backward pass for the cumulative loss of dimension `j`
                    cumulative_loss.backward()
                    
                    # Optimizer step for dimension `j`
                    self.optimizer[j].step()

                    # update params for plotting
                    length_scale = self.gp_models[j].covar_module.lengthscale.item()
                    self.kernel_params[j].append(length_scale)

                    #print(training_step)
                    if training_step % 20 == 10:
                        plt.semilogy(self.kernel_params[j])
                    
                if training_step % 20 == 10:
                    plt.show()

            # Optionally print progress after each training iteration
            #print(f"Iter {i + 1}/{training_iter} - Loss: {cumulative_loss.item():.3f}")
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

