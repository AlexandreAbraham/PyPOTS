"""
The implementation of GP-VAE for the partially-observed time-series imputation task.

"""

# Created by Jun Wang <jwangfx@connect.ust.hk> and Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import os
from typing import Union, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    import nni
except ImportError:
    pass


from .core import _GP_VAE
from .data import DatasetForGPVAE
from ..base import BaseNNImputer
from ...data.checking import key_in_data_set
from ...optim.adam import Adam
from ...optim.base import Optimizer
from ...utils.logging import logger
from ...utils.metrics import calc_mse

# for probabilistic GP
import torch.nn.functional as F
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

import matplotlib.pyplot as plt

# For kernel parameters estimation
import gpytorch
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        # Define mean and covariance modules
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel())
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.models import ApproximateGP

class SparseGPModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(SparseGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()


        matern_kernel = gpytorch.kernels.MaternKernel(nu=2.5, has_lengthscale = True)  # Default is nu=2.5
        matern_kernel.lengthscale = torch.tensor(1.0)
        self.covar_module = matern_kernel
        #kernel = gpytorch.kernels.MaternKernel(nu=2.5) + gpytorch.kernels.WhiteNoiseKernel()


        #self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

import torch
import gpytorch
from gpytorch.kernels import RBFKernel, PeriodicKernel, LinearKernel, ScaleKernel

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

        # init models, optimizers and losses for training
        if False:
            for model, likelihood in zip(self.gp_models, self.likelihoods):
                model.train()
                likelihood.train()
                optimizer = torch.optim.Adam([
                    {'params': model.parameters()},
                    {'params': likelihood.parameters()},
                ], lr=0.1)
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
                self.optimizer.append(optimizer)
                self.mll.append(mll)


        for j in range(self.latent_size):

            # Define inducing points for the sparse GP model
            n_inducing_pts = 8
            inducing_points = torch.linspace(0, n_inducing_pts, n_inducing_pts)  # Adjust number of inducing points as needed
            inducing_points = inducing_points.reshape(1,-1,1).repeat(8, 1, 1)
            #print(inducing_points.shape)

            # Instantiate model and likelihood
            gp_model = SparseGPModel(inducing_points=inducing_points)
            #gp_model = VariationalComplexGPModel(inducing_points = inducing_points)    
            likelihood = gpytorch.likelihoods.GaussianLikelihood()

            if torch.cuda.is_available():
                gp_model = gp_model.cuda()
                likelihood = likelihood.cuda()

            gp_model.train()
            likelihood.train()

            self.gp_models.append(gp_model)
            self.likelihoods.append(likelihood)


            # Collect parameters from each component
            gp_model_params = list(self.gp_models[j].parameters())
            likelihood_params = list(self.likelihoods[j].parameters())

            # Find shared parameters between gp_model_params and likelihood_params
            gp_model_params_set = set(gp_model_params)
            likelihood_params_set = set(likelihood_params)
            
            unique_gp_model_params = list(gp_model_params_set - likelihood_params_set)
            unique_likelihood_params = list(likelihood_params_set - gp_model_params_set)
            shared_params = list(gp_model_params_set & likelihood_params_set)

            # Define optimizer with separate parameter groups for unique and shared parameters
            optimizer = torch.optim.Adam([
                {'params': unique_gp_model_params},
                {'params': unique_likelihood_params},
                {'params': shared_params}  # only added once
            ], lr=0.1)

            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihoods[j], self.gp_models[j]) #, num_data = 
            

            # Append the optimizer and mll to their respective lists
            self.optimizer.append(optimizer)
            self.mll.append(mll)

    def fit_kernel(self, training_loader):

        self.instantiate_gp_models(training_loader)

        self.kernel_params = {j:[] for j in range(self.latent_size)}

        dims_to_train = list(np.arange(self.latent_size))

        training_iter = 50
        for i in range(training_iter):
            # Zero grad optimizers
            for optimizer in self.optimizer:
                optimizer.zero_grad()

            # Start training loop
            training_step = 0
            for idx, data in enumerate(training_loader):
                training_step += 1
                inputs = self.assemble_data(data)
                x = inputs['X']

                # Encode the data
                qz_x = self.encoder(x)
                z_mu, z_var = qz_x.mean.detach(), qz_x.variance.detach()   

                #print(dims_to_train)
                for j in dims_to_train:

                    if len(self.kernel_params[j]) > 30 and np.abs(self.kernel_params[j][-1] - np.array(self.kernel_params[j])[-10:-2].mean()) < 1e-3:
                        dims_to_train.remove(j)
                        print(f'No further training for dim {j}')


                    #if len(self.kernel_params[j]) > 10 and (self.kernel_params[j][-1] - self.kernel_params[j][-10:-2].mean()).abs() > 1e-3:
                        # Initialize cumulative loss for dimension `j`
                    cumulative_loss = 0

                    use_batches = True
                    if not use_batches:

                        x_selected, y_selected = self.batch_select_values_for_GP_inference(z_mu[:,:,j], z_var[:,:,j])

                        #print(x_selected.shape)

                        out = self.gp_models[j](x_selected).reshape(-1)
                        loss = -self.mll[j](out, y_selected.reshape(-1))

                        #print(loss.shape)

                        # Accumulate the loss
                        cumulative_loss += loss.mean()

                    else:
                    
                        for batch in range(z_mu.shape[0]):

                            # Select values with highest certainty
                            x_selected, y_selected = self.select_values_for_GP_inference(z_mu[batch, :, j], z_var[batch, :, j])

                            #plt.plot(x_selected.detach().numpy(), y_selected.detach().numpy(), marker = 'o')

                            # Compute loss on those selected observations only
                            out = self.gp_models[j](x_selected)
                            loss = -self.mll[j](out, y_selected)

                            # Accumulate the loss
                            cumulative_loss += loss

                    length_scale = self.gp_models[j].covar_module.lengthscale.item()
                    self.kernel_params[j].append(length_scale)

                    # Backward pass for the cumulative loss of dimension `j`
                    cumulative_loss.backward()
                    
                    # Optimizer step for dimension `j`
                    self.optimizer[j].step()

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


class GP_VAE(BaseNNImputer):
    """The PyTorch implementation of the GPVAE model :cite:`fortuin2020gpvae`.

    Parameters
    ----------
    n_steps :
        The number of time steps in the time-series data sample.

    n_features :
        The number of features in the time-series data sample.

    latent_size : int,
        The feature dimension of the latent embedding

    encoder_sizes : tuple,
        The tuple of the network size in encoder

    decoder_sizes : tuple,
        The tuple of the network size in decoder

    beta : float,
        The weight of KL divergence in ELBO.

    M : int,
        The number of Monte Carlo samples for ELBO estimation during training.

    K : int,
        The number of importance weights for IWAE model training loss.

    kernel: str
        The type of kernel function chosen in the Gaussain Process Proir. ["cauchy", "diffusion", "rbf", "matern"]

    sigma : float,
        The scale parameter for a kernel function

    length_scale : float,
        The length scale parameter for a kernel function

    kernel_scales : int,
        The number of different length scales over latent space dimensions

    window_size : int,
        Window size for the inference CNN.

    batch_size : int
        The batch size for training and evaluating the model.

    epochs : int
        The number of epochs for training the model.

    patience : int
        The patience for the early-stopping mechanism. Given a positive integer, the training process will be
        stopped when the model does not perform better after that number of epochs.
        Leaving it default as None will disable the early-stopping.

    optimizer : pypots.optim.base.Optimizer
        The optimizer for model training.
        If not given, will use a default Adam optimizer.

    num_workers : int
        The number of subprocesses to use for data loading.
        `0` means data loading will be in the main process, i.e. there won't be subprocesses.

    device : :class:`torch.device` or list
        The device for the model to run on. It can be a string, a :class:`torch.device` object, or a list of them.
        If not given, will try to use CUDA devices first (will use the default CUDA device if there are multiple),
        then CPUs, considering CUDA and CPU are so far the main devices for people to train ML models.
        If given a list of devices, e.g. ['cuda:0', 'cuda:1'], or [torch.device('cuda:0'), torch.device('cuda:1')] , the
        model will be parallely trained on the multiple devices (so far only support parallel training on CUDA devices).
        Other devices like Google TPU and Apple Silicon accelerator MPS may be added in the future.

    saving_path : str
        The path for automatically saving model checkpoints and tensorboard files (i.e. loss values recorded during
        training into a tensorboard file). Will not save if not given.

    model_saving_strategy : str
        The strategy to save model checkpoints. It has to be one of [None, "best", "better"].
        No model will be saved when it is set as None.
        The "best" strategy will only automatically save the best model after the training finished.
        The "better" strategy will automatically save the model during training whenever the model performs
        better than in previous epochs.

    """

    def __init__(
        self,
        n_steps: int,
        n_features: int,
        latent_size: int,
        encoder_sizes: tuple = (64, 64),
        decoder_sizes: tuple = (64, 64),
        kernel: str = "cauchy",
        beta: float = 0.2,
        M: int = 1,
        K: int = 1,
        sigma: float = 1.0,
        length_scale: float = 7.0,
        kernel_scales: int = 1,
        window_size: int = 3,
        batch_size: int = 32,
        epochs: int = 100,
        patience: Optional[int] = None,
        optimizer: Optional[Optimizer] = Adam(),
        num_workers: int = 0,
        device: Optional[Union[str, torch.device, list]] = None,
        saving_path: str = None,
        model_saving_strategy: Optional[str] = "best",
        verbose: bool = True,
    ):
        super().__init__(
            batch_size,
            epochs,
            patience,
            num_workers,
            device,
            saving_path,
            model_saving_strategy,
            verbose,
        )
        available_kernel_type = ["cauchy", "diffusion", "rbf", "matern"]
        assert kernel in available_kernel_type, f"kernel should be one of {available_kernel_type}, but got {kernel}"

        self.n_steps = n_steps
        self.n_features = n_features
        self.latent_size = latent_size
        self.kernel = kernel
        self.encoder_sizes = encoder_sizes
        self.decoder_sizes = decoder_sizes
        self.beta = beta
        self.M = M
        self.K = K
        self.sigma = sigma
        self.length_scale = length_scale
        self.kernel_scales = kernel_scales

        # set up the model
        self.model = _GP_VAE(
            input_dim=self.n_features,
            time_length=self.n_steps,
            latent_dim=self.latent_size,
            kernel=self.kernel,
            encoder_sizes=self.encoder_sizes,
            decoder_sizes=self.decoder_sizes,
            beta=self.beta,
            M=self.M,
            K=self.K,
            sigma=self.sigma,
            length_scale=self.length_scale,
            kernel_scales=self.kernel_scales,
            window_size=window_size,
        )
        self._send_model_to_given_device()
        self._print_model_size()

        # set up the optimizer
        self.optimizer = optimizer
        self.optimizer.init_optimizer(self.model.parameters())

        # set gp
        self.gp = ProbabilisticGP(self.model.backbone, assemble_data = self._assemble_input_for_training)

    def _assemble_input_for_training(self, data: list) -> dict:
        # fetch data
        (
            indices,
            X,
            missing_mask,
        ) = self._send_data_to_given_device(data)

        # assemble input data
        inputs = {
            "indices": indices,
            "X": X,
            "missing_mask": missing_mask,
        }

        return inputs

    def _assemble_input_for_validating(self, data: list) -> dict:
        # fetch data
        (
            indices,
            X,
            missing_mask,
            X_ori,
            indicating_mask,
        ) = self._send_data_to_given_device(data)

        # assemble input data
        inputs = {
            "indices": indices,
            "X": X,
            "missing_mask": missing_mask,
            "X_ori": X_ori,
            "indicating_mask": indicating_mask,
        }

        return inputs

    def _assemble_input_for_testing(self, data: list) -> dict:
        return self._assemble_input_for_training(data)

    def _train_model(
        self,
        training_loader: DataLoader,
        val_loader: DataLoader = None,
    ) -> None:
        # each training starts from the very beginning, so reset the loss and model dict here
        self.best_loss = float("inf")
        self.best_model_dict = None

        try:
            training_step = 0.
            for epoch in range(1, self.epochs + 1):
                self.model.train()
                self.model.backbone.temperature = epoch / (self.epochs + 1)
                epoch_train_loss_collector = []
                for idx, data in enumerate(training_loader):
                    training_step += 1
                    inputs = self._assemble_input_for_training(data)
                    self.optimizer.zero_grad()
                    results = self.model.forward(inputs)
                    # use sum() before backward() in case of multi-gpu training
                    results["loss"].sum().backward()
                    #clip gradients
                    #torch.nn.utils.clip_grad_norm_(v_1, max_norm=1.0, norm_type=2)
                    self.optimizer.step()
                    epoch_train_loss_collector.append(results["loss"].sum().item())

                    # save training loss logs into the tensorboard file for every step if in need
                    if self.summary_writer is not None:
                        self._save_log_into_tb_file(training_step, "training", results)

                # mean training loss of the current epoch
                mean_train_loss = np.mean(epoch_train_loss_collector)

                if val_loader is not None:
                    self.model.eval()
                    imputation_loss_collector = []
                    with torch.no_grad():
                        for idx, data in enumerate(val_loader):
                            inputs = self._assemble_input_for_validating(data)
                            results = self.model.forward(inputs, training=False, n_sampling_times=1)
                            imputed_data = results["imputed_data"].mean(axis=1)
                            imputation_mse = (
                                calc_mse(
                                    imputed_data,
                                    inputs["X_ori"],
                                    inputs["indicating_mask"],
                                )
                                .sum()
                                .detach()
                                .item()
                            )
                            imputation_loss_collector.append(imputation_mse)

                    mean_val_loss = np.mean(imputation_loss_collector)

                    # save validation loss logs into the tensorboard file for every epoch if in need
                    if self.summary_writer is not None:
                        val_loss_dict = {
                            "imputation_loss": mean_val_loss,
                        }
                        self._save_log_into_tb_file(epoch, "validating", val_loss_dict)

                    logger.info(
                        f"Epoch {epoch:03d} - "
                        f"training loss: {mean_train_loss:.4f}, "
                        f"validation loss: {mean_val_loss:.4f}"
                    )
                    mean_loss = mean_val_loss
                else:
                    logger.info(f"Epoch {epoch:03d} - training loss: {mean_train_loss:.4f}")
                    mean_loss = mean_train_loss

                if np.isnan(mean_loss):
                    logger.warning(f"‼️ Attention: got NaN loss in Epoch {epoch}. This may lead to unexpected errors.")

                if mean_loss < self.best_loss:
                    self.best_epoch = epoch
                    self.best_loss = mean_loss
                    self.best_model_dict = self.model.state_dict()
                    self.patience = self.original_patience
                else:
                    self.patience -= 1

                # save the model if necessary
                self._auto_save_model_if_necessary(
                    confirm_saving=self.best_epoch == epoch and self.model_saving_strategy == "better",
                    saving_name=f"{self.__class__.__name__}_epoch{epoch}_loss{mean_loss:.4f}",
                )

                if os.getenv("enable_tuning", False):
                    nni.report_intermediate_result(mean_loss)
                    if epoch == self.epochs - 1 or self.patience == 0:
                        nni.report_final_result(self.best_loss)

                if self.patience == 0:
                    logger.info("Exceeded the training patience. Terminating the training procedure...")
                    break

        except KeyboardInterrupt:  # if keyboard interrupt, only warning
            logger.warning("‼️ Training got interrupted by the user. Exist now ...")
        except Exception as e:  # other kind of exception follows below processing
            logger.error(f"❌ Exception: {e}")
            if self.best_model_dict is None:  # if no best model, raise error
                raise RuntimeError(
                    "Training got interrupted. Model was not trained. Please investigate the error printed above."
                )
            else:
                RuntimeWarning(
                    "Training got interrupted. Please investigate the error printed above.\n"
                    "Model got trained and will load the best checkpoint so far for testing.\n"
                    "If you don't want it, please try fit() again."
                )

        if np.isnan(self.best_loss):
            raise ValueError("Something is wrong. best_loss is Nan after training.")

        logger.info(f"Finished training. The best model is from epoch#{self.best_epoch}.")

    def fit(
        self,
        train_set: Union[dict, str],
        val_set: Optional[Union[dict, str]] = None,
        file_type: str = "hdf5",
    ) -> None:
        # Step 1: wrap the input data with classes Dataset and DataLoader
        training_set = DatasetForGPVAE(train_set, return_X_ori=False, return_y=False, file_type=file_type)
        training_loader = DataLoader(
            training_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        val_loader = None
        if val_set is not None:
            if not key_in_data_set("X_ori", val_set):
                raise ValueError("val_set must contain 'X_ori' for model validation.")
            val_set = DatasetForGPVAE(val_set, return_X_ori=True, return_y=False, file_type=file_type)
            val_loader = DataLoader(
                val_set,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
        
        # Step 2: train the AE model and freeze it
        self._train_model(training_loader, val_loader)
        self.model.load_state_dict(self.best_model_dict)
        self.model.eval()  # set the model as eval status to freeze it.

        # Step 2bis: learn the kernel
        print('fitting kernel')
        self._fit_kernel(training_loader)

        # Step 3: save the model if necessary
        self._auto_save_model_if_necessary(confirm_saving=self.model_saving_strategy == "best")

    def predict(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
        n_sampling_times: int = 1,
    ) -> dict:
        """

        Parameters
        ----------
        test_set : dict or str
            The dataset for model validating, should be a dictionary including keys as 'X' and 'y',
            or a path string locating a data file.
            If it is a dict, X should be array-like of shape [n_samples, sequence length (n_steps), n_features],
            which is time-series data for validating, can contain missing values, and y should be array-like of shape
            [n_samples], which is classification labels of X.
            If it is a path string, the path should point to a data file, e.g. a h5 file, which contains
            key-value pairs like a dict, and it has to include keys as 'X' and 'y'.

        file_type :
            The type of the given file if test_set is a path string.

        n_sampling_times:
            The number of sampling times for the model to produce predictions.

        Returns
        -------
        result_dict: dict
            Prediction results in a Python Dictionary for the given samples.
            It should be a dictionary including a key named 'imputation'.

        """
        assert n_sampling_times > 0, "n_sampling_times should be greater than 0."

        self.model.eval()  # set the model as eval status to freeze it.
        test_set = DatasetForGPVAE(test_set, return_X_ori=False, return_y=False, file_type=file_type)
        test_loader = DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        imputation_collector = []

        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                inputs = self._assemble_input_for_testing(data)
                #results = self.model.forward(inputs, training=False, n_sampling_times=n_sampling_times)
                #imputed_data = results["imputed_data"]

                # embed data in latent space
                embedding = self.model.encode(inputs, training-False, n_sampling_times=n_sampling_times)
                # correct with gaussian process
                imputed_data = self.gp.infer(embedding)
                imputation_collector.append(imputed_data)

        imputation = torch.cat(imputation_collector).cpu().detach().numpy()
        result_dict = {
            "imputation": imputation,
        }
        return result_dict

    def impute(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
    ) -> np.ndarray:
        """Impute missing values in the given data with the trained model.

        Parameters
        ----------
        test_set :
            The data samples for testing, should be array-like of shape [n_samples, sequence length (n_steps),
            n_features], or a path string locating a data file, e.g. h5 file.

        file_type :
            The type of the given file if X is a path string.

        Returns
        -------
        array-like, shape [n_samples, sequence length (n_steps), n_features],
            Imputed data.
        """

        results_dict = self.predict(test_set, file_type=file_type)
        return results_dict["imputation"]

    def _fit_kernel(
        self,
        training_loader) -> None:

        self.gp.fit_kernel(training_loader)

    def fit_kernel(
        self,
        train_set: Union[dict, str],
        val_set: Optional[Union[dict, str]] = None,
        file_type: str = "hdf5",
    ) -> None:
        # Step 1: wrap the input data with classes Dataset and DataLoader
        training_set = DatasetForGPVAE(train_set, return_X_ori=False, return_y=False, file_type=file_type)
        training_loader = DataLoader(
            training_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        val_loader = None
        if val_set is not None:
            if not key_in_data_set("X_ori", val_set):
                raise ValueError("val_set must contain 'X_ori' for model validation.")
            val_set = DatasetForGPVAE(val_set, return_X_ori=True, return_y=False, file_type=file_type)
            val_loader = DataLoader(
                val_set,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
        
        # Step 2bis: learn the kernel
        print('fitting kernel')
        self._fit_kernel(training_loader)

        # Step 3: save the model if necessary
        self._auto_save_model_if_necessary(confirm_saving=self.model_saving_strategy == "best")

        # deprecated !!!!
        for dim in self.gp.kernel_params.keys():
            print(self.gp.kernel_params[dim])
            l, n = self.gp.kernel_params[dim]['length_scale'], self.gp.kernel_params[dim]['noise']
            plt.subplot(3,1,1)
            plt.hist(l)
            plt.subplot(3,1,2)
            plt.hist(n)
            plt.subplot(3,1,3)
            plt.scatter(l,n, alpha = .5)
            plt.show()

    class SequentialGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(SequentialGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            # Use an RBF kernel with ARD to handle multiple dimensions
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=latent_dim)
            )

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

##### OLD ########

class ProbabilisticGP_:
    """
    A sub-module of a VAE that corrects the latent time-seties via the Probabilistic GP regression scheme
    """
    def __init__(self):
        self.encoder = None # this should be a neural network
        self.kernel = None
        self.enforce_variance_bias = False
        self.kernel_params = {}

    def correct_embedding_with_gp(self, z_mu, z_sigma):
        """
        Correct the latent embeddings using Gaussian Process regression.

        Args:
            z_mu (Tensor): Mean of the latent variables.
            z_sigma (Tensor): Standard deviation of the latent variables.

        Returns:
            Tensor: Corrected latent variables.
        """
        if self.kernel is None:
            self.fit_gp_kernels(z_mu.detach(), z_sigma.detach())

        n_samples, n_dims = z_mu.shape
        z_corrected = torch.zeros_like(z_mu)

        x = torch.arange(n_samples).reshape(-1, 1)
        v = z_sigma.clone()

        # Enforce variance bias adjustment
        if self.enforce_variance_bias:
            v = self.adjust_variances(v)

        a,b,temp = self.set_kernel(z_mu)

        for j in range(n_dims):
            self.kernel[j].fit(x, z_mu[:,j], temp*v[:,j])
            z_corrected[:,j], var_corrected = self.kernel[j].predict() #predict(x)

        return z_corrected

    def adjust_variances(self, v, window_size=4):
            """
            Adjusts variances by enforcing variance bias based on local variance patterns.

            Args:
                v (torch.Tensor): Variance tensor to adjust.
                window_size (int): Number of elements to include on each side for local averaging.

            Returns:
                torch.Tensor: Adjusted variance tensor.
            """
            kernel_size = 2 * window_size + 1
            kernel = torch.ones(kernel_size, device=v.device)
            kernel[window_size] = 0  # Exclude the current element
            kernel /= (kernel_size - 1)

            # Pad the variance tensor
            v_padded = F.pad(v.unsqueeze(0).unsqueeze(0), (window_size, window_size), mode='reflect')

            # Compute local mean variances
            local_means = F.conv1d(v_padded, kernel.view(1, 1, -1))[0, 0]

            # Calculate variance differences
            v_diff = (local_means - v).clamp(-1e3, 1e3)
            v_diff_std = v_diff.std()

            # Avoid division by zero
            if v_diff_std == 0:
                added_variance_bias = torch.ones_like(v_diff)
            else:
                added_variance_bias = torch.max(torch.tensor(1.0, device=v.device), v_diff / v_diff_std)

            # Apply the adjustment
            v_adjusted = v * added_variance_bias
            return v_adjusted

    def fit_kernel(self, training_loader, noise_ratio = 100):
        """
        Fit Gaussian Process kernels for each latent dimension.

        Args:
            z_mu (Tensor): Mean of the latent variables.
            z_sigma (Tensor): Standard deviation of the latent variables.
        """
        
        desired_quantile = 0.9

        likelihood = {}
        gp_model = {}
        optimizer = {}
        mll = {}

        # init Gp models
        for latent_dim in range(self.latent_size):

            # Initialize the likelihood and model with empty training data
            likelihood[latent_dim] = gpytorch.likelihoods.GaussianLikelihood()
            initial_train_x = torch.empty(0, 1)
            initial_train_y = torch.empty(0)
            gp_model[latent_dim] = SequentialGPModel(initial_train_x, initial_train_y, likelihood)

            # Set the model and likelihood to training mode
            gp_model[latent_dim].train()
            likelihood[latent_dim].train()

            # Initialize the optimizer
            optimizer[latent_dim] = torch.optim.Adam(gp_model[latent_dim].parameters(), lr=0.01)

            # Define the marginal log likelihood
            mll[latent_dim] = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood[latent_dim], gp_model[latent_dim])

        # start training loop
        training_step = 0
        for idx, data in enumerate(training_loader):
            training_step += 1
            inputs = self._assemble_input_for_training(data)
            x = inputs['X']

            # encode the data
            qz_x = self.model.backbone.encode(x)   
            z_mu, z_var = qz_x.mean, qz_x.variance

            for latent_dim in range(self.latent_size):
                self.fit_kernel_1dim(gp_model[latent_dim], optimizer[latent_dim], z_mu[:,latent_dim], z_var[:,latent_dim])
                
                optimizer[latent_dim].zero_grad()

                z_approx, z_true = gp_model[latent_dim]()

                loss = -mll[latent_dim](z_approx, z_true)

                loss.backward()

                optimizer[latent_dim].step()

        # Once data is fit, update kernel for every dim
        for latent_dim in range(self.latent_size):
            self.kernel_params[latent_dim]['length_scale'] += [length_scale]
            self.kernel_params[latent_dim]['noise'] += [noise]

            
        def fit_kernel_1dim(self, gp_model, optimizer, z_mu, z_var):
            """
            Fit the kernel for one latent variable
            """

            # Define time steps
            T = torch.arange(len(z_mu))

            # filter on points with small variance
            desired_quantile = .9
            q = torch.quantile(z_var, desired_quantile)
            # find a way to update this
            mask = z_var < q
            Z_filtered, T_filtered = z_mu[mask], T[mask]

            # Check if there's data after filtering
            if Z_filtered.shape[0] > 0:

                # Update the GP model's training data
                gp_model.set_train_data(inputs=T_filtered, targets=Z_filtered, strict=False)

                # Zero gradients from previous iteration
                optimizer.zero_grad()

                # Forward pass: Compute the GP model output
                output = gp_model(T_filtered)

                return output, Z_filtered

class ProbabilisticGaussianProcessRegressor: #using torch linalg solve
    def __init__(self, length_scale=1.0, noise=1e-6):
        self.length_scale = length_scale
        self.noise = noise
        #print(noise, 'noise')
        self.compute_variance = False
        self.alpha = 1.

    def rbf_kernel(self, X1, X2):
        """Compute the RBF kernel (Gaussian kernel)."""
        dists = torch.cdist(X1, X2, p=2)  # Pairwise distances
        K = torch.exp(-0.5 * (dists / self.length_scale) ** 2)
        return K

    def fit(self, X_train, y_train, variance):
        """Fit the Gaussian Process model with training data and probabilities."""
        # Detach X and y
        X_train = X_train.detach().reshape(-1,1).float()
        y_train = y_train.detach().float()
        
        self.X_train = X_train
        self.y_train = y_train
        #self.probabilities = probabilities

        # Compute the kernel matrix for the training data
        self.K_star = self.rbf_kernel(X_train, X_train)
        
        # Add noise term to the diagonal (regularization)
        #self.alpha = .5

        # Normalize probas
        #self.probabilities = self.probabilities / self.probabilities.max()
        #variance = (1 - self.probabilities)/self.probabilities
        #variance = -1 * torch.log(self.probabilities)
        K_regularisation = self.noise * torch.diag(variance).clone()
        self.K = self.K_star + K_regularisation * self.alpha

        self.bias = torch.sign(y_train) * variance * .05

    def predict(self):
        """Predict the mean and variance for the training points themselves."""
        
        # Mean prediction
        mu_s = self.K_star @ torch.linalg.solve(self.K, self.y_train + self.bias)
        #K_star = K_star / K_star.sum(axis=0)[None,:]
        #mu_s = K_star @ (self.y_train * self.probabilities)
        
        if self.compute_variance:
            # Compute the inverse of the kernel matrix
            self.K_inv = torch.linalg.solve(self.K, torch.eye(self.K.size(0), device=self.K.device))

            # Variance (which will be 0 for exact points)
            K_star = self.K_star
            K_s_s = K_star - K_star @ self.K_inv @ K_star
            sigma_s = K_s_s.diag()

            M = K_star @ self.K_inv
            idx = M.shape[0]//2
            plt.imshow(K_star.detach())
            plt.title('K_star')
            plt.show()
            plt.imshow(M.detach())
            plt.show()
            plt.plot(M[idx].detach(), label = 'M')
            plt.legend()
            plt.show()
            plt.plot((M[idx]*self.probabilities).detach())
            plt.plot((M[idx//2]*self.probabilities).detach())
            plt.show()
            plt.legend()
            plt.show()
        else:
            sigma_s = 1

        
        return mu_s, sigma_s

    def set_kernel(self, z):


        out = self.encoder.kernel_params(z.mean(axis=0).reshape(1,z.shape[1]))[0]

        a = torch.abs(out[0])*10
        b = torch.abs(out[1])*10
        temp = torch.exp(out[2])*10

        a, b = torch.tensor(1), torch.tensor(.5)
        #temp = torch.tensor(10)

        return a, b, temp
