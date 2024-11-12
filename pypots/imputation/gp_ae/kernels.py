# for probabilistic GP
import torch.nn.functional as F
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.models import ApproximateGP

import torch
import gpytorch
from gpytorch.kernels import RBFKernel, PeriodicKernel, LinearKernel, ScaleKernel, MaternKernel


import matplotlib.pyplot as plt

# For kernel parameters estimation
import gpytorch

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        # Define mean and covariance modules
        batch_shape = torch.Size([8])
        self.mean_module = gpytorch.means.ConstantMean(batch_shape = batch_shape)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape = batch_shape),
            batch_shape = batch_shape)
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        #mean_x = torch.zeros(x.shape)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        # Define mean and covariance modules
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel())
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        #mean_x = torch.zeros(x.shape)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class BatchIndependentMultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        batch_shape = torch.Size([8])
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=batch_shape),
            batch_shape=batch_shape
        )
         # Define and combine kernels with consistent batch shape
        matern_kernel = MaternKernel(nu=2.5, batch_shape=batch_shape)
        self.covar_module = matern_kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )

class SparseGPModel(ApproximateGP):
    def __init__(self, inducing_points):
        # Set up batch shape
        batch_shape = torch.Size([inducing_points.size(0)])  # matches inducing_points batch size

        # Set up the variational distribution with the correct shape and number of inducing points
        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.size(1),
            batch_shape=batch_shape
        )

        # Set up the variational strategy with learnable inducing locations
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )

        # Initialize ApproximateGP with the variational strategy
        super().__init__(variational_strategy)

        # Define mean module with batch shape
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)

        # Define and combine kernels with consistent batch shape
        matern_kernel = MaternKernel(nu=2.5, batch_shape=batch_shape)
        self.covar_module = matern_kernel  # or + RBFKernel(batch_shape=batch_shape)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class VariationalComplexGPModel(ApproximateGP):
    def __init__(self, inducing_points):
        # Set up variational distribution and strategy with inducing points
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        
        super(VariationalComplexGPModel, self).__init__(variational_strategy)
        
        # Mean module
        self.mean_module = gpytorch.means.ConstantMean()
        
        # Define the composite kernel: (RBF + Periodic) * Linear

        # Initialize the components with specific values
        rbf_kernel = RBFKernel()
        rbf_kernel.lengthscale = torch.tensor(1.0)

        periodic_kernel = PeriodicKernel()
        periodic_kernel.lengthscale = torch.tensor(1.0)
        periodic_kernel.period_length = torch.tensor(2.0)

        linear_kernel = LinearKernel()
        linear_kernel.variance = torch.tensor(1.0)
        self.covar_module = ScaleKernel((rbf_kernel) )
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
