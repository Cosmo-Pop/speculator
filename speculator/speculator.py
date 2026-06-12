import pickle

import numpy as np
import torch
import wandb
from sklearn.decomposition import IncrementalPCA
from torch.utils.data import DataLoader, TensorDataset

class Speculator(torch.nn.Module):
    """
    SPECULATOR model for emulating SEDs.

    Attributes
    ----------
    n_parameters : int
        Number of SPS parameters used as input.
    n_wavelengths : int
        Number of wavelengths in grid.
    n_pcas : int
        Number of principal components used in decomposition.
    n_hidden : list of int
        Number of units per hidden layer.
    wavelengths : torch.tensor
        Wavelength grid.
    architecture : list of int
        Input and output dimensions of layers.
    n_layers : int
        Number of network layers.
    parameters_shift : torch.tensor
        Shift applied to parameters before feeding into network.
    parameters_scale : torch.tensor
        Scale applied to parameters before feeding into network.
    pca_shift : torch.tensor
        Shift applied to PCA scores predicted by the network.
    pca_scale : torch.tensor
        Scale applied to PCA scores predicted by the network.
    log_spectrum_shift : torch.tensor
        Shift applied to spectrum predicted by the network.
    log_spectrum_scale : torch.tensor
        Scale applied to spectrum predicted by the network.
    pca_transform_matrix : torch.tensor
        Matrix that transforms from PCA scores to log spectrum.
    W : torch.nn.ParameterList
        Neural network weights.
    b : torch.nn.ParameterList
        Neural network biases.
    alphas : torch.nn.ParameterList
        Scaling in non-linear activation function (beta in Alsing+20 eq. 8).
    betas : torch.nn.ParameterList
        Offset in activation function (gamma in Alsing+20 eq. 8).
    params : torch.nn.ParameterList
        Full set of trainable parameters.
    optimizer : torch.optim.Optimizer
        Optimizer targetting `params`.
    """

    def __init__(
        self,
        n_parameters=None,
        wavelengths=None,
        pca_transform_matrix=None,
        parameters_shift=None,
        parameters_scale=None,
        pca_shift=None,
        pca_scale=None,
        log_spectrum_shift=None,
        log_spectrum_scale=None,
        n_hidden=[50, 50],
        optimizer=None,
        restore=False,
        restore_filename=None,
        restore_weights_only=False,
        device="cpu",
    ):
        """
        Constructor.

        Parameters
        ----------
        n_parameters : int, optional
            Number of input SPS parameters.
        wavelengths : torch.tensor, optional
            Target wavelength grid.
        pca_transform_matrix : torch.tensor, optional
            Matrix with PCA basis vectors, shape `(n_pcas, n_wavelengths)`.
        parameters_shift : torch.tensor, optional
            Shift applied to inputs.
        parameters_scale : torch.tensor, optional
            Scale applied to inputs.
        pca_shift : torch.tensor, optional
            Shift applied to output PCA scores.
        pca_scale : torch.tensor, optional
            Scale applied to output PCA scores.
        log_spectrum_shift : torch.tensor, optional
            Shift applied to output log spectrum.
        log_spectrum_scale : torch.tensor, optional
            Scale applied to output log spectrum.
        n_hidden : list of int, optional
            Number of units per hidden layer. The network will be initialized
            with `len(n_hidden)` hidden layers, plus an input and output layer.
        optimizer : torch.optim.Optimizer, optional
            Optimizer (currently ignored/not implemented; defaults to Adam).
        restore : bool, optional
            If `True`, attempts to load a state dict from `restore_filename`.
        restore_filename : str, optional
            Path to a trained `Speculator` model (.pt file).
        restore_weights_only : bool, optional
            If `True`, tries to call `torch.load` with `weights_only=True`.
            Default is `False` (i.e. living dangerously).
        device : str, optional
            Device to build everything on. 
        """

        # super
        super(Speculator, self).__init__()

        # parameters
        self.n_parameters = n_parameters
        self.n_wavelengths = pca_transform_matrix.shape[-1]
        self.n_pcas = pca_transform_matrix.shape[0]
        self.n_hidden = n_hidden
        self.wavelengths = wavelengths

        # architecture
        self.architecture = [self.n_parameters] + self.n_hidden + [self.n_pcas]
        self.n_layers = len(self.architecture) - 1

        # shifts and scales and transform matrix

        # input parameters shift and scale
        self.parameters_shift = torch.tensor(
            (
                parameters_shift
                if parameters_shift is not None
                else np.zeros(self.n_parameters)
            ),
            dtype=torch.float32,
        ).to(device)
        self.parameters_scale = torch.tensor(
            (
                parameters_scale
                if parameters_scale is not None
                else np.ones(self.n_parameters)
            ),
            dtype=torch.float32,
        ).to(device)

        # PCA shift and scale
        self.pca_shift = torch.tensor(
            pca_shift if pca_shift is not None else np.zeros(self.n_pcas),
            dtype=torch.float32,
        ).to(device)
        self.pca_scale = torch.tensor(
            pca_scale if pca_scale is not None else np.ones(self.n_pcas),
            dtype=torch.float32,
        ).to(device)

        # spectrum shift and scale
        self.log_spectrum_shift = torch.tensor(
            (
                log_spectrum_shift
                if log_spectrum_shift is not None
                else np.zeros(self.n_wavelengths)
            ),
            dtype=torch.float32,
        ).to(device)
        self.log_spectrum_scale = torch.tensor(
            (
                log_spectrum_scale
                if log_spectrum_scale is not None
                else np.ones(self.n_wavelengths)
            ),
            dtype=torch.float32,
        ).to(device)

        # pca transform matrix
        self.pca_transform_matrix = torch.tensor(
            pca_transform_matrix, dtype=torch.float32
        ).to(device)

        # trainable variables...

        # weights, biases and activation function parameters for each layer of the network
        self.W = torch.nn.ParameterList()
        self.b = torch.nn.ParameterList()
        self.alphas = torch.nn.ParameterList()
        self.betas = torch.nn.ParameterList()
        for i in range(self.n_layers):
            # FIX 1: Correctly apply He weight initialization for each layer
            # Use self.architecture[i] (current layer's input size) for scaling.
            input_dim = self.architecture[i]
            output_dim = self.architecture[i+1]
            
            # Weights initialized based on the layer's input dimension
            std_dev = torch.sqrt(torch.tensor(2.0 / input_dim))
            self.W.append(
                torch.nn.Parameter(std_dev * torch.randn((input_dim, output_dim)))
            )
            
            # Biases initialized to zero (as before, which is correct)
            self.b.append(
                torch.nn.Parameter(torch.zeros(output_dim))
            )
        for i in range(self.n_layers - 1):
            output_dim = self.architecture[i+1]
            self.alphas.append(
                torch.nn.Parameter(torch.randn((self.architecture[i + 1]))).to(device)
            )
            self.betas.append(
                torch.nn.Parameter(torch.randn((self.architecture[i + 1]))).to(device)
            )

        self.to(device)
        self.params = torch.nn.ParameterList()
        self.params.extend(self.W)
        self.params.extend(self.b)
        self.params.extend(self.alphas)
        self.params.extend(self.betas)

        # optimizer
        self.optimizer = torch.optim.Adam(self.params, lr=1e-3)

        # backwards-compatible loading from state dict
        if restore:
            restore_state_dict = torch.load(restore_filename, map_location=device, weights_only=restore_weights_only)
            try:
                missing, unexpected = self.load_state_dict(restore_state_dict)
            except RuntimeError as err:
                # this catches an old-style state dict with different structure
                # it attempts to build something compatible with the new class by copying keys
                missing, unexpected = self.load_state_dict(restore_state_dict, strict=False)
                if len(unexpected) == 0 and len(missing) == len(restore_state_dict.keys()):
                    for i in range(self.n_layers):
                        restore_state_dict[f'W.{i}'] = restore_state_dict[f'params.{i}']
                        restore_state_dict[f'b.{i}'] = restore_state_dict[f'params.{self.n_layers + i}']
                    for i in range(self.n_layers - 1):    
                        restore_state_dict[f'alphas.{i}'] = restore_state_dict[f'params.{2*self.n_layers + i}']
                        restore_state_dict[f'betas.{i}'] = restore_state_dict[f'params.{3*self.n_layers - 1 + i}']
                    self.load_state_dict(restore_state_dict)
                # if things don't look compatible, raise the error anyway
                else:
                    raise err



    def set_device(self, device):
        """
        Change device.

        Parameters
        ----------
        device : str or torch.device
            Device to move all class attributes to.
        """
        self.parameters_shift = self.parameters_shift.to(device)
        self.parameters_scale = self.parameters_scale.to(device)

        self.pca_shift = self.pca_shift.to(device)
        self.pca_scale = self.pca_scale.to(device)

        self.log_spectrum_shift = self.log_spectrum_shift.to(device)
        self.log_spectrum_scale = self.log_spectrum_scale.to(device)

        self.pca_transform_matrix = self.pca_transform_matrix.to(device)

        for i in range(self.n_layers):
            self.W[i] = torch.nn.Parameter(self.W[i].data.to(device))
            self.b[i] = torch.nn.Parameter(self.b[i].data.to(device))
        for i in range(self.n_layers - 1):
            self.alphas[i] = self.alphas[i].to(device)
            self.betas[i] = self.betas[i].to(device)

        self.params = torch.nn.ParameterList(self.W + self.b + self.alphas + self.betas)

    def activation(self, x, alpha, beta):
        """
        Non-linear activation function (Alsing+20 eq. 8).

        Parameters
        ----------
        x : torch.tensor
            Inputs.
        alpha : torch.tensor
            Scaling parameter (beta in Alsing+20 eq. 8).
        beta : torch.tensor
            Offset parameter (gamma in Alsing+20 eq. 8).

        Returns
        -------
        a : torch.tensor
            Activation function evaluated for `x`.
        """
        return torch.multiply(
            torch.add(
                beta,
                torch.multiply(
                    torch.sigmoid(torch.multiply(alpha, x)), torch.subtract(1.0, beta)
                ),
            ),
            x,
        )

    def forward(self, parameters):
        """
        Forward pass through the network to predict PCA coefficients.

        Parameters
        ----------
        parameters : torch.Tensor
            Input SPS parameters in real space.

        Returns
        -------
        output : torch.Tensor
            Output PCA coefficients.
        """
        output = torch.divide(
            torch.subtract(parameters, self.parameters_shift), self.parameters_scale
        )
        for i in range(self.n_layers - 1):

            # non-linear activation function
            output = self.activation(
                torch.add(torch.matmul(output, self.W[i]), self.b[i]),
                self.alphas[i],
                self.betas[i],
            )

        # linear output layer
        output = torch.add(torch.matmul(output, self.W[-1]), self.b[-1])

        # rescale the output
        output = torch.add(torch.multiply(output, self.pca_scale), self.pca_shift)

        return output

    def save(self, filename):
        """
        Save the state dict for later restore.

        Parameters
        ----------
        filename : str
            Path to save the state dict to.
        """
        torch.save(self.state_dict(), filename)

    def log_spectrum(self, parameters):
        """
        Forward pass to predict a log spectrum.

        Parameters
        ----------
        parameters : torch.Tensor
            Input SPS parameters in real space.

        Returns
        -------
        log_spectrum : torch.Tensor
            Output log spectrum.
        """

        # pass through network to compute PCA coefficients
        pca_coefficients = self.forward(parameters)

        # transform from PCA to normalized spectrum basis; 
        # shift and re-scale normalized spectrum -> spectrum
        return torch.add(
            torch.multiply(
                torch.matmul(pca_coefficients, self.pca_transform_matrix),
                self.log_spectrum_scale,
            ),
            self.log_spectrum_shift,
        )

    ### Infrastructure for network training ###
    def compute_loss_spectra(self, spectra, parameters, noise_floor):
        """
        Compute loss in spectrum space.

        Parameters
        ----------
        spectra : torch.Tensor
            Target spectra (units of Lsun/Hz).
        parameters : torch.Tensor
            SPS parameters.
        noise_floor : torch.Tensor
            Error tolerance per wavelength bin.

        Returns
        -------
        loss : torch.Tensor
            Root mean squared error weighted by `noise_floor`.
        """
        return torch.sqrt(
            torch.mean(
                torch.divide(
                    torch.square(
                        torch.subtract(
                            torch.exp(self.log_spectrum(parameters)), spectra
                        )
                    ),
                    torch.square(noise_floor),
                )
            )
        )

    def compute_loss_pca(self, pca, parameters):
        """
        Compute loss in space of PCA coefficients.

        Parameters
        ----------
        pca : torch.Tensor
            Target PCA coefficients.
        parameters : torch.Tensor
            SPS parameters.

        Returns
        -------
        loss : torch.Tensor
            Root mean squared error.
        """
        return torch.sqrt(
            torch.mean(torch.square(torch.subtract(self.forward(parameters), pca)))
        )

    def compute_loss_log_spectra(self, log_spectra, parameters):
        """
        Compute loss in space of log spectra.

        Parameters
        ----------
        log_spectra : torch.Tensor
            Target log(spectra).
        parameters : torch.Tensor
            SPS parameters.

        Returns
        -------
        loss : torch.Tensor
            Root mean squared error.
        """
        return torch.sqrt(
            torch.mean(
                torch.square(torch.subtract(self.log_spectrum(parameters), log_spectra))
            )
        )

    def training_step(
        self, theta, outputs, maxbatch=10000, loss_type="pca", noise_floor=None
    ):
        """
        Training epoch with sub-batching.

        Parameters
        ----------
        theta : torch.Tensor
            Input SPS parameters.
        outputs : torch.Tensor
            Target outputs (PCA coefficients, log spectra, or spectra).
        maxbatch : int, optional
            Maximum batch size before sub-batching will be used. Default is 10000.
        loss_type : str, optional
            Space to compute loss in. Should match `outputs` and be one of
            `['pca', 'log_spectra', 'spectra']`. Default (recommended) is `'pca'`.
        noise_floor : torch.Tensor, optional
            Wavelength weighting to use if `loss_type='spectra'`. Not used otherwise.

        Returns
        -------
        loss : torch.Tensor
            Accumulated loss.
        """
        self.optimizer.zero_grad()
        if theta.shape[0] < maxbatch:

            # loss
            if loss_type == "pca":
                loss = self.compute_loss_pca(outputs, theta)
            elif loss_type == "log_spectra":
                loss = self.compute_loss_log_spectra(outputs, theta)
            elif loss_type == "spectra":
                loss = self.compute_loss_spectra(outputs, theta, noise_floor)

            # backprop
            loss.backward()

            # update
            self.optimizer.step()

            return loss

        else:

            # create iterable dataset
            dataloader = DataLoader(TensorDataset(theta, outputs), batch_size=maxbatch)

            # loop over sub batches
            for theta_, outputs_ in dataloader:
                with torch.set_grad_enabled(True):

                    # loss
                    if loss_type == "pca":
                        loss = (
                            self.compute_loss_pca(theta_, outputs_)
                            * theta_.shape[0]
                            / theta.shape[0]
                        )
                    elif loss_type == "log_spectra":
                        loss = (
                            self.compute_loss_log_spectra(theta_, outputs_)
                            * theta_.shape[0]
                            / theta.shape[0]
                        )
                    elif loss_type == "spectra":
                        loss = self.compute_loss_spectra(theta_, outputs_, noise_floor)

                    # backprop
                    loss.backward()

            # update parameters
            self.optimizer.step()
            self.optimizer.zero_grad()

            return loss


class SpectrumPCA:
    """
    SPECULATOR PCA compression class.

    Attributes
    ----------
    n_parameters : int
            Number of SPS parameters.
    n_wavelengths : int
        Number of wavelengths in spectra.
    n_pcas : int
        Number of principal components to use.
    n_batches : int
        Number of batches of training data `len(log_spectrum_filenames)`.
    log_spectrum_filenames : list of str
        Paths to files containing log spectra to build PCA from.
    parameter_filenames : list of str
        Paths to files containing SPS parameters.
    parameter_selection : callable
        Function that returns a boolean array applying a selection
        cut given SPS parameters.
    PCA : sklearn.decomposition.IncrementalPCA
        Object that builds the PCA.
    log_spectrum_shift : np.array
        Shift applied to log spectra.
    log_spectrum_scale : np.array
        Scale applied to output log spectra.
    parameter_shift : np.array
        Shift applied to SPS parameters.
    parameter_scale : np.array
        Scale applied to SPS parameters.
    pca_transform_matrix : np.array
        Matrix containing PCA basic vectors.
    """

    def __init__(
        self,
        n_parameters,
        n_wavelengths,
        n_pcas,
        log_spectrum_filenames,
        parameter_filenames,
        parameter_selection=None,
    ):
        """
        Constructor.
        
        Parameters
        ----------
        n_parameters : int
            Number of SPS parameters.
        n_wavelengths : int
            Number of wavelengths in spectra.
        n_pcas : int
            Number of principal components to use.
        log_spectrum_filenames : list of str
            Paths to files containing log spectra to build PCA from.
        parameter_filenames : list of str
            Paths to files containing SPS parameters.
        parameter_selection : callable, optional
            Function that returns a boolean array applying a selection
            cut given SPS parameters.
        """

        # input parameters
        self.n_parameters = n_parameters
        self.n_wavelengths = n_wavelengths
        self.n_pcas = n_pcas
        self.log_spectrum_filenames = log_spectrum_filenames
        self.parameter_filenames = parameter_filenames
        self.n_batches = len(self.parameter_filenames)

        # PCA object
        self.PCA = IncrementalPCA(n_components=self.n_pcas)

        # parameter selection (implementing any cuts on strange parts of parameter space)
        self.parameter_selection = parameter_selection

    def compute_spectrum_parameters_shift_and_scale(self):
        """
        Compute shift and scale for inputs and outputs.

        Sets attributes `log_spectrum_shift`, `log_spectrum_scale`,
        `parameter_shift`, `parameter_scale`.
        """
        # shift and scale
        self.log_spectrum_shift = np.zeros(self.n_wavelengths)
        self.log_spectrum_scale = np.zeros(self.n_wavelengths)
        self.parameter_shift = np.zeros(self.n_parameters)
        self.parameter_scale = np.zeros(self.n_parameters)

        # loop over training data files, accumulate means and std deviations
        for i in range(self.n_batches):

            # accumulate assuming no parameter selection
            if self.parameter_selection is None:
                self.log_spectrum_shift += (
                    np.mean(np.load(self.log_spectrum_filenames[i]), axis=0)
                    / self.n_batches
                )
                self.log_spectrum_scale += (
                    np.std(np.load(self.log_spectrum_filenames[i]), axis=0)
                    / self.n_batches
                )
                self.parameter_shift += (
                    np.mean(np.load(self.parameter_filenames[i]), axis=0)
                    / self.n_batches
                )
                self.parameter_scale += (
                    np.std(np.load(self.parameter_filenames[i]), axis=0)
                    / self.n_batches
                )
            # else make selections and accumulate
            else:
                # import spectra and make parameter-based cut
                log_spectra = np.load(self.log_spectrum_filenames[i])
                parameters = np.load(self.parameter_filenames[i])
                selection = self.parameter_selection(parameters)

                # update shifts and scales
                self.log_spectrum_shift += (
                    np.mean(log_spectra[selection, :], axis=0) / self.n_batches
                )
                self.log_spectrum_scale += (
                    np.std(log_spectra[selection, :], axis=0) / self.n_batches
                )
                self.parameter_shift += (
                    np.mean(parameters[selection, :], axis=0) / self.n_batches
                )
                self.parameter_scale += (
                    np.std(parameters[selection, :], axis=0) / self.n_batches
                )

    def train_pca(self):
        """
        Train PCA incrementally.

        Updates the `PCA` attribute and sets `pca_transform_matrix`.
        """
        # loop over training data files, increment PCA
        for i in range(self.n_batches):

            if self.parameter_selection is None:

                # load spectra and shift+scale
                normalized_log_spectra = (
                    np.load(self.log_spectrum_filenames[i]) - self.log_spectrum_shift
                ) / self.log_spectrum_scale

                # partial PCA fit
                self.PCA.partial_fit(normalized_log_spectra)

            else:

                # select based on parameters
                selection = self.parameter_selection(
                    np.load(self.parameter_filenames[i])
                )

                # load spectra and shift+scale
                normalized_log_spectra = (
                    np.load(self.log_spectrum_filenames[i])[selection, :]
                    - self.log_spectrum_shift
                ) / self.log_spectrum_scale

                # partial PCA fit
                self.PCA.partial_fit(normalized_log_spectra)

        # set the PCA transform matrix
        self.pca_transform_matrix = self.PCA.components_

    def transform_and_stack_training_data(self, filename, retain=False):
        """
        Transform the training data set to PCA basis.

        Sets attributes `pca_shift`, `pca_scale`, 
        optionally sets `training_pca`, `training_parameters`.

        Saves stacked data to `filename + '_pca.npy'` 
        and `filename + '_parameters.npy'`.

        Parameters
        ----------
        filename : str
            Path to save stacked and transformed data to.
        retain : bool, optional
            If `True`, stores the stacked data as class attributes.
        """
        # transform the spectra to PCA basis
        training_pca = np.concatenate(
            [
                self.PCA.transform(
                    (np.load(self.log_spectrum_filenames[i]) - self.log_spectrum_shift)
                    / self.log_spectrum_scale
                )
                for i in range(self.n_batches)
            ]
        )

        # stack the input parameters
        training_parameters = np.concatenate(
            [np.load(self.parameter_filenames[i]) for i in range(self.n_batches)]
        )

        if self.parameter_selection is not None:
            selection = self.parameter_selection(training_parameters)
            training_pca = training_pca[selection, :]
            training_parameters = training_parameters[selection, :]

        # shift and scale of PCA basis
        self.pca_shift = np.mean(training_pca, axis=0)
        self.pca_scale = np.std(training_pca, axis=0)

        # save stacked transformed training data
        np.save(filename + "_pca.npy", training_pca)
        np.save(filename + "_parameters.npy", training_parameters)

        # retain training data as attributes if retain == True
        if retain:
            self.training_pca = training_pca
            self.training_parameters = training_parameters

    def validate_pca_basis(self, log_spectrum_filename, parameter_filename=None):
        """
        Apply the PCA transform to some unseen data.

        Parameters
        ----------
        log_spectrum_filename : str
            Path to validation data (log spectra).
        parameter_filename : str, optional
            Path to corresponding SPS parameters (if parameter selection is needed).

        Returns
        -------
        log_spectra : np.array
            Original spectra loaded from `log_spectrum_filename`.
        log_spectra_in_basis : np.array
            Reconstruction of spectra after projecting to/from PCA coefficients.
        """
        # load in the data (and select based on parameter selection if neccessary)
        if self.parameter_selection is None:

            # load spectra and shift+scale
            log_spectra = np.load(log_spectrum_filename)
            normalized_log_spectra = (
                log_spectra - self.log_spectrum_shift
            ) / self.log_spectrum_scale

        else:

            # select based on parameters
            selection = self.parameter_selection(np.load(self.parameter_filename))

            # load spectra and shift+scale
            log_spectra = np.load(log_spectrum_filename)[selection, :]
            normalized_log_spectra = (
                log_spectra - self.log_spectrum_shift
            ) / self.log_spectrum_scale

        # transform to PCA basis and back
        log_spectra_pca = self.PCA.transform(normalized_log_spectra)
        log_spectra_in_basis = (
            np.dot(log_spectra_pca, self.pca_transform_matrix) * self.log_spectrum_scale
            + self.log_spectrum_shift
        )

        # return raw spectra and spectra in basis
        return log_spectra, log_spectra_in_basis


class PhotulatorBasic(torch.nn.Module):
    """
    PHOTULATOR model for emulating photometry.

    This is a minimal, simplified version with untrainable activation functions.

    Attributes
    ----------
    n_parameters : int
        Number of SPS parameters.
    n_hidden : list of int
        Number of units per hidden layer.
    filters : list of str
        Names of filters being emulated.
    n_filters : int
        Number of filters being emulated, `len(filters)`.
    parameter_names : list of str
        Names for the SPS parameters
    architecture : list of int
        Input and output dimensions of the network layers.
    n_layers : int
        Number of network layers.
    activation : torch.nn.Module
        Activation function.
    parameters_shift : torch.Tensor
        Shift for the input SPS parameters.
    parameters_scale : torch.Tensor
        Scale for the input SPS parameters.
    magnitudes_shift : torch.Tensor
        Shift for the output magnitudes.
    magnitudes_scale : torch.Tensor
        Scale for the output magnitudes.
    network : torch.nn.Sequential
        Emulator network.
    f_b : torch.Tensor
        Flux softening parameter (b in Lupton+99 eq. 3).
    ln10 : torch.Tensor
        Natural logarithm of 10.
    """

    def __init__(
        self,
        n_parameters=None,
        filters=None,
        parameters_shift=None,
        parameters_scale=None,
        magnitudes_shift=None,
        magnitudes_scale=None,
        f_b=None,
        n_hidden=[128, 128],
        sigma_init=1e-3,
        activation=torch.nn.SiLU,
        parameter_names=None,
    ):
        """
        Constructor.
        
        Parameters
        ----------
        n_parameters : int, optional
            Number of input SPS parameters.
        filters : list of str, optional
            Names of the filter(s) being emulated.
        parameters_shift : torch.Tensor, optional
            Shift for the input SPS parameters.
        parameters_scale : torch.Tensor, optional
            Scale for the input SPS parameters.
        magnitudes_shift : torch.Tensor, optional
            Shift for the output magnitudes.
        magnitudes_scale : torch.Tensor, optional
            Scale for the output magnitudes.
        f_b : torch.Tensor, optional
            Flux softening parameter (b in Lupton+99 eq. 3) when
            predicting asinh magnitudes (luptitudes).
        n_hidden : list of int, optional
            Number of units per hidden layer.
        sigma_init : torch.Tensor, optional
            Initial weight standard deviation. Unused.
        activation : torch.nn.Module, optional
            Activation function.
        parameter_names : list of str, optional
            Names of input SPS parameters.
        """

        # super
        super(PhotulatorBasic, self).__init__()

        # parameters
        self.n_parameters = n_parameters
        self.n_hidden = n_hidden
        self.filters = filters
        self.n_filters = len(filters)
        self.parameter_names = parameter_names

        # architecture
        self.architecture = [self.n_parameters] + self.n_hidden + [self.n_filters]
        self.n_layers = len(self.architecture) - 1
        self.activation = activation

        # shifts and scales...

        # shifts and scales and transform matrix into tensorflow constants...

        # input parameters shift and scale
        self.register_buffer(
            "parameters_shift",
            torch.tensor(
                (
                    parameters_shift
                    if parameters_shift is not None
                    else np.zeros(self.n_parameters)
                ),
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "parameters_scale",
            torch.tensor(
                (
                    parameters_scale
                    if parameters_scale is not None
                    else np.ones(self.n_parameters)
                ),
                dtype=torch.float32,
            ),
        )

        # spectrum shift and scale
        self.register_buffer(
            "magnitudes_shift",
            torch.tensor(
                (
                    magnitudes_shift
                    if magnitudes_shift is not None
                    else np.zeros(self.n_filters)
                ),
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "magnitudes_scale",
            torch.tensor(
                (
                    magnitudes_scale
                    if magnitudes_scale is not None
                    else np.ones(self.n_filters)
                ),
                dtype=torch.float32,
            ),
        )

        # network
        self.network = torch.nn.Sequential()
        for layer in range(self.n_layers):
            self.network.add_module(
                "layer {}".format(layer),
                torch.nn.Linear(self.architecture[layer], self.architecture[layer + 1]),
            )
            self.network.add_module("activation {}".format(layer), self.activation())

        # luptitude parameters
        self.register_buffer(
            "f_b",
            (
                torch.tensor(0.0, dtype=torch.float32)
                if f_b is None
                else torch.tensor(f_b, dtype=torch.float32)
            ),
        )
        self.register_buffer("ln10", torch.tensor(np.log(10), dtype=torch.float32))

    def forward(self, parameters):
        """
        Forward pass through the network.

        Predicts absolute magnitude per unit stellar mass formed.

        Parameters
        ----------
        parameters : torch.Tensor
            Input SPS parameters.

        Returns
        -------
        output : torch.Tensor
            Absolute magnitudes per unit mass.
        """
        # shift and scale the inputs
        output = torch.divide(
            torch.subtract(parameters, self.parameters_shift), self.parameters_scale
        )

        # network
        output = self.network(output)

        # rescale the output
        output = torch.add(
            torch.multiply(output, self.magnitudes_scale), self.magnitudes_shift
        )

        return output

    def flux(self, parameters, N):
        """
        Compute flux in maggies.

        Parameters
        ----------
        parameters : torch.Tensor
            Input SPS parameters.
        N : torch.Tensor
            Normalisation factor, distmod - 2.5*log10(M/Msun).

        Returns
        -------
        flux : torch.Tensor
            Flux in maggies (AB system).
        """
        return torch.exp(
            torch.multiply(
                torch.multiply(
                    torch.tensor(-0.4, dtype=torch.float32, device=parameters.device),
                    self.magnitudes(parameters, N),
                ),
                self.ln10,
            )
        )

    def magnitudes(self, parameters, N):
        """
        Compute apparent magnitude.

        Parameters
        ----------
        parameters : torch.Tensor
            Input SPS parameters.
        N : torch.Tensor
            Normalisation factor, distmod - 2.5*log10(M/Msun).

        Returns
        -------
        mag : torch.Tensor
            Apparent magnitude (AB system).
        """
        return torch.add(self.forward(parameters), N)

    def luptitudes(self, parameters, N):
        """
        Compute asinh magnitude (Lupton+99).

        Parameters
        ----------
        parameters : torch.Tensor
            Input SPS parameters.
        N : torch.Tensor
            Normalisation factor, distmod - 2.5*log10(M/Msun).

        Returns
        -------
        asinh_mag : torch.Tensor
            Asinh magnitudes using internal `f_b` attribute as softening scale.
        """
        # absolute magnitudes -> flux in nano maggies
        flux = torch.multiply(
            self.flux(parameters, N),
            torch.tensor(1e9, dtype=torch.float32, device=parameters.device),
        )

        # flux in nano maggies -> luptitudes (in mormal magnitude units)
        return torch.multiply(
            torch.tensor(
                -1.0857362047581294, dtype=torch.float32, device=parameters.device
            ),
            torch.subtract(
                torch.arcsinh(
                    torch.divide(
                        flux,
                        torch.multiply(
                            torch.tensor(
                                2.0, dtype=torch.float32, device=parameters.device
                            ),
                            self.f_b,
                        ),
                    )
                ),
                torch.log(
                    torch.divide(
                        torch.tensor(
                            1e9, dtype=torch.float32, device=parameters.device
                        ),
                        self.f_b,
                    )
                ),
            ),
        )


class Photulator(torch.nn.Module):
    """
    PHOTULATOR model for emulating photometry.

    Full-featured version. 
    Includes parametrized activation function from Alsing+20.

    Attributes
    ----------
    n_parameters : int
        Number of SPS parameters.
    n_hidden : list of int
        Number of units per hidden layer.
    filters : list of str
        Names of filters being emulated.
    n_filters : int
        Number of filters being emulated, `len(filters)`.
    parameter_names : list of str
        Names for the SPS parameters
    architecture : list of int
        Input and output dimensions of the network layers.
    n_layers : int
        Number of network layers.
    parameters_shift : torch.Tensor
        Shift for the input SPS parameters.
    parameters_scale : torch.Tensor
        Scale for the input SPS parameters.
    magnitudes_shift : torch.Tensor
        Shift for the output magnitudes.
    magnitudes_scale : torch.Tensor
        Scale for the output magnitudes.
    W : torch.nn.ParameterList
        Neural network weights.
    b : torch.nn.ParameterList
        Neural network biases.
    alphas : torch.nn.ParameterList
        Scaling in non-linear activation function (beta in Alsing+20 eq. 8).
    betas : torch.nn.ParameterList
        Offset in activation function (gamma in Alsing+20 eq. 8).
    f_b : torch.Tensor
        Flux softening parameter (b in Lupton+99 eq. 3).
    ln10 : torch.Tensor
        Natural logarithm of 10.
    """

    def __init__(
        self,
        n_parameters=None,
        filters=None,
        parameters_shift=None,
        parameters_scale=None,
        magnitudes_shift=None,
        magnitudes_scale=None,
        f_b=None,
        n_hidden=[50, 50],
        sigma_init=1e-3,
        parameter_names=None,
    ):
        """
        Constructor.
        
        Parameters
        ----------
        n_parameters : int, optional
            Number of input SPS parameters.
        filters : list of str, optional
            Names of the filter(s) being emulated.
        parameters_shift : torch.Tensor, optional
            Shift for the input SPS parameters.
        parameters_scale : torch.Tensor, optional
            Scale for the input SPS parameters.
        magnitudes_shift : torch.Tensor, optional
            Shift for the output magnitudes.
        magnitudes_scale : torch.Tensor, optional
            Scale for the output magnitudes.
        f_b : torch.Tensor, optional
            Flux softening parameter (b in Lupton+99 eq. 3).
        n_hidden : list of int, optional
            Number of units per hidden layer.
        sigma_init : torch.Tensor, optional
            Initial weight standard deviation. Unused.
        parameter_names : list of str, optional
            Names of input SPS parameters.
        """

        # super
        super(Photulator, self).__init__()

        # parameters
        self.n_parameters = n_parameters
        self.n_hidden = n_hidden
        self.filters = filters
        self.n_filters = len(filters)
        self.parameter_names = parameter_names

        # architecture
        self.architecture = [self.n_parameters] + self.n_hidden + [self.n_filters]
        self.n_layers = len(self.architecture) - 1

        # input parameters shift and scale
        self.register_buffer(
            "parameters_shift",
            torch.tensor(
                (
                    parameters_shift
                    if parameters_shift is not None
                    else np.zeros(self.n_parameters)
                ),
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "parameters_scale",
            torch.tensor(
                (
                    parameters_scale
                    if parameters_scale is not None
                    else np.ones(self.n_parameters)
                ),
                dtype=torch.float32,
            ),
        )

        # spectrum shift and scale
        self.register_buffer(
            "magnitudes_shift",
            torch.tensor(
                (
                    magnitudes_shift
                    if magnitudes_shift is not None
                    else np.zeros(self.n_filters)
                ),
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "magnitudes_scale",
            torch.tensor(
                (
                    magnitudes_scale
                    if magnitudes_scale is not None
                    else np.ones(self.n_filters)
                ),
                dtype=torch.float32,
            ),
        )

        # trainable variables...

        # weights, biases and activation function parameters for each layer of the network
        self.W = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    sigma_init
                    * torch.randn((self.architecture[i], self.architecture[i + 1]))
                )
                for i in range(self.n_layers)
            ]
        )
        self.b = torch.nn.ParameterList(
            [
                torch.nn.Parameter(sigma_init * torch.randn((self.architecture[i + 1])))
                for i in range(self.n_layers)
            ]
        )
        self.alphas = torch.nn.ParameterList(
            [
                torch.nn.Parameter(sigma_init * torch.randn((self.architecture[i + 1])))
                for i in range(self.n_layers)
            ]
        )
        self.betas = torch.nn.ParameterList(
            [
                torch.nn.Parameter(sigma_init * torch.randn((self.architecture[i + 1])))
                for i in range(self.n_layers)
            ]
        )

        # luptitude parameters
        self.register_buffer(
            "f_b",
            (
                torch.tensor(0.0, dtype=torch.float32)
                if f_b is None
                else torch.tensor(f_b, dtype=torch.float32)
            ),
        )
        self.register_buffer("ln10", torch.tensor(np.log(10), dtype=torch.float32))

    def activation(self, x, alpha, beta):
        """
        Non-linear activation function (Alsing+20 eq. 8).

        Parameters
        ----------
        x : torch.tensor
            Inputs.
        alpha : torch.tensor
            Scaling parameter (beta in Alsing+20 eq. 8).
        beta : torch.tensor
            Offset parameter (gamma in Alsing+20 eq. 8).

        Returns
        -------
        a : torch.tensor
            Activation function evaluated for `x`.
        """
        return torch.multiply(
            torch.add(
                beta,
                torch.multiply(torch.sigmoid(torch.multiply(alpha, x)), 1.0 - beta),
            ),
            x,
        )


    def forward(self, parameters):
        """
        Forward pass through the network.

        Predicts absolute magnitude per unit stellar mass formed.

        Parameters
        ----------
        parameters : torch.Tensor
            Input SPS parameters.

        Returns
        -------
        output : torch.Tensor
            Absolute magnitudes per unit mass.
        """
        # shift and scale
        output = torch.divide(
            torch.subtract(parameters, self.parameters_shift), self.parameters_scale
        )

        # layers
        for i, (W, b, alpha, beta) in enumerate(
            zip(self.W, self.b, self.alphas, self.betas)
        ):

            # non-linear activation function
            output = self.activation(torch.add(torch.matmul(output, W), b), alpha, beta)

        # rescale the output
        output = torch.add(
            torch.multiply(output, self.magnitudes_scale), self.magnitudes_shift
        )

        return output

    def flux(self, parameters, N):
        """
        Compute flux in maggies.

        Parameters
        ----------
        parameters : torch.Tensor
            Input SPS parameters.
        N : torch.Tensor
            Normalisation factor, distmod - 2.5*log10(M/Msun).

        Returns
        -------
        flux : torch.Tensor
            Flux in maggies (AB system).
        """
        return torch.exp(
            torch.multiply(
                torch.multiply(
                    torch.tensor(-0.4, dtype=torch.float32, device=parameters.device),
                    self.magnitudes(parameters, N),
                ),
                self.ln10,
            )
        )

    def magnitudes(self, parameters, N):
        """
        Compute apparent magnitude.

        Parameters
        ----------
        parameters : torch.Tensor
            Input SPS parameters.
        N : torch.Tensor
            Normalisation factor, distmod - 2.5*log10(M/Msun).

        Returns
        -------
        mag : torch.Tensor
            Apparent magnitude (AB system).
        """
        return torch.add(self.forward(parameters), N)

    def luptitudes(self, parameters, N):
        """
        Compute asinh magnitude (Lupton+99).

        Parameters
        ----------
        parameters : torch.Tensor
            Input SPS parameters.
        N : torch.Tensor
            Normalisation factor, distmod - 2.5*log10(M/Msun).

        Returns
        -------
        asinh_mag : torch.Tensor
            Asinh magnitudes using internal `f_b` attribute as softening scale.
        """
        # absolute magnitudes -> flux in nano maggies
        flux = torch.multiply(
            self.flux(parameters, N),
            torch.tensor(1e9, dtype=torch.float32, device=parameters.device),
        )

        # flux in nano maggies -> luptitudes (in mormal magnitude units)
        return torch.multiply(
            torch.tensor(
                -1.0857362047581294, dtype=torch.float32, device=parameters.device
            ),
            torch.subtract(
                torch.arcsinh(
                    torch.divide(
                        flux,
                        torch.multiply(
                            torch.tensor(
                                2.0, dtype=torch.float32, device=parameters.device
                            ),
                            self.f_b,
                        ),
                    )
                ),
                torch.log(
                    torch.divide(
                        torch.tensor(
                            1e9, dtype=torch.float32, device=parameters.device
                        ),
                        self.f_b,
                    )
                ),
            ),
        )


class PhotulatorModelStack:
    """
    Stack of Photulator models for many bands.

    Attributes
    ----------
    n_emulators : int
        Number of Photulator models in the stack.
    emulators : list of Photulator or PhotulatorBasic
        Photometry emulators.
    """
    def __init__(self, root_dir, filenames, device="cpu", weights_only=False):
        """
        Constructor.

        Parameters
        ----------
        root_dir : str
            Path to a parent directory where the emulators live.
        filenames : list of str
            Names of the saved emulators within `root_dir`.
        device : str, optional
            Device to load the emulators onto.
        weights_only : bool, optional
            If `True`, calls `torch.load` with `weights_only=True`. Default is `False`.
        """
        # how many emulators?
        self.n_emulators = len(filenames)

        # load emulator models
        self.emulators = [torch.load(root_dir + filename, weights_only=weights_only).to(device) for filename in filenames]

    def fluxes(self, theta, N):
        """
        Compute flux in maggies.

        Parameters
        ----------
        theta : torch.Tensor
            Input SPS parameters.
        N : torch.Tensor
            Normalisation factor, distmod - 2.5*log10(M/Msun).

        Returns
        -------
        flux : torch.Tensor
            Flux in maggies (AB system).
        """
        return torch.concat(
            [self.emulators[i].fluxes(theta, N) for i in range(self.n_emulators)],
            axis=-1,
        )

    def magnitudes(self, theta, N):
        """
        Compute apparent magnitude.

        Parameters
        ----------
        theta : torch.Tensor
            Input SPS parameters.
        N : torch.Tensor
            Normalisation factor, distmod - 2.5*log10(M/Msun).

        Returns
        -------
        mag : torch.Tensor
            Apparent magnitude (AB system).
        """
        return torch.concat(
            [self.emulators[i].magnitudes(theta, N) for i in range(self.n_emulators)],
            axis=-1,
        )

    def luptitudes(self, theta, N):
        """
        Compute asinh magnitude (Lupton+99).

        Parameters
        ----------
        theta : torch.Tensor
            Input SPS parameters.
        N : torch.Tensor
            Normalisation factor, distmod - 2.5*log10(M/Msun).

        Returns
        -------
        asinh_mag : torch.Tensor
            Asinh magnitudes using internal `f_b` attributes as softening scale.
        """
        return torch.concat(
            [self.emulators[i].luptitudes(theta, N) for i in range(self.n_emulators)],
            axis=-1,
        )


### train photulator model stack ###
def train_photulator_stack(
    training_theta,
    training_N,
    training_mag,
    parameters_shift,
    parameters_scale,
    magnitudes_shift,
    magnitudes_scale,
    parameter_names=None,
    n_layers=4,
    n_units=128,
    filters=None,
    validation_split=0.1,
    lr=[1e-3, 1e-4, 1e-5],
    batch_size=[1000, 10000, 100000],
    maxepochs=500,
    patience=20,
    root_dir="",
    verbose=True,
    device="cuda",
    wandb_init=None,
    loss_in="absmag",
    f_b=None,
    sigma_init=1e-2,
    activation="alsing20",
    cuda_graphs=False,
):
    """
    Trains a stack of Photulator models.

    Parameters
    ----------
    training_theta : torch.Tensor
        SPS parameters for training data, shape `(n_samples, n_parameters)`.
    training_N : torch.Tensor
        Normalization factors for training data, shape `(n_samples,)`.
        Normalisation factor defined as N = distmod - 2.5*log10(M/Msun).
    training_mag : torch.Tensor
        Magnitude values for training data, shape `(n_samples, n_filters)`.
    parameters_shift : torch.Tensor
        Mean shift for parameter normalization.
    parameters_scale : torch.Tensor
        Standard deviation for parameter normalization.
    magnitudes_shift : list
        Mean shift for magnitude normalization, one per filter.
    magnitudes_scale : list
        Standard deviation for magnitude normalization, one per filter.
    parameter_names : list of str, optional
        Names of the SPS parameters.
    n_layers : int, optional
        Number of hidden layers in the neural network. Default is 4.
    n_units : int, optional
        Number of units per hidden layer. Default is 128.
    filters : list of str, optional
        List of filter names to train models for.
    validation_split : float, optional
        Fraction of data to hold out for validation. Default is 0.1.
    lr : list of float, optional
        Learning rates for each training round. Default is `[1e-3, 1e-4, 1e-5]`.
    batch_size : list of int, optional
        Batch sizes for each training round. Default is `[1000, 10000, 100000]`.
        This should have the same length as `lr`.
    maxepochs : int, optional
        Maximum number of epochs per training round. Default is 500.
    patience : int, optional
        Number of epochs with no improvement before early stopping. Default is 20.
    root_dir : str, optional
        Directory to save trained models. Default is the current directory.
    verbose : bool, optional
        Whether to print progress information. Default is `True`.
    device : str, optional
        Device to use for training (`'cuda'` or `'cpu'`).
    wandb_init : dict, optional
        Weights & Biases initialization parameters.
    loss_in : str, optional
        Loss function to use (`'absmag'` or `'asinhmag'`).
    f_b : array-like, optional
        Softening parameters for asinh magnitudes, one per filter.
    sigma_init : float, optional
        Standard deviation for for weight initialization.  Default is `1e-2`.
    activation : str, optional
        Activation function for neural network layers. Default is `'alsing20'`.
        Options are `('alsing20', 'tanh', 'silu', 'leakyrelu')`.
    cuda_graphs : bool, optional
        Whether to use CUDA graphs for optimization. Default is `False`.

    Notes
    -----
    For lower error rates on the models it is recommended to use a small learning
    rate and high batch size during the last training round.
    """
    # Set matmuls to high
    torch.set_float32_matmul_precision("high")

    # architecture
    n_hidden = [n_units] * n_layers

    # how many training rounds to do?
    rounds = len(lr)

    # train each band in turn
    for f in range(len(filters)):

        if wandb_init is not None:
            wandb.init(
                name=wandb_init["name"] + "_" + filters[f],
                project=wandb_init["project"],
            )

        if verbose is True:
            print("filter " + filters[f] + "...")

        # construct the PHOTULATOR model
        if activation == "alsing20":
            photulator = Photulator(
                n_parameters=training_theta.shape[-1],
                filters=[filters[f]],
                parameters_shift=parameters_shift,
                parameters_scale=parameters_scale,
                magnitudes_shift=magnitudes_shift[f],
                magnitudes_scale=magnitudes_scale[f],
                n_hidden=[n_units] * n_layers,
                f_b=f_b[f],
                sigma_init=sigma_init,
                parameter_names=parameter_names,
            ).cuda()
        else:
            activation_functions = {
                "tanh": torch.nn.Tanh,
                "silu": torch.nn.SiLU,
                "leakyrelu": torch.nn.LeakyReLU,
            }
            photulator = PhotulatorBasic(
                n_parameters=training_theta.shape[-1],
                filters=[filters[f]],
                parameters_shift=parameters_shift,
                parameters_scale=parameters_scale,
                magnitudes_shift=magnitudes_shift[f],
                magnitudes_scale=magnitudes_scale[f],
                n_hidden=[n_units] * n_layers,
                f_b=f_b[f],
                sigma_init=sigma_init,
                parameter_names=parameter_names,
                activation=activation_functions[activation],
            ).cuda()
        # location for saving the model
        save_location = (
            root_dir
            + "model_{}x{}_{}_".format(n_layers, n_units, activation)
            + filters[f]
        )

        # Compile with reduce-overhead mode and fullgraph for CUDA graphs
        if cuda_graphs is True:
            photulator = torch.compile(
                photulator, mode="reduce-overhead", fullgraph=True
            )

        # construct an optimizer
        optimizer = torch.optim.Adam(photulator.parameters())

        # train using cooling/heating schedule for lr/batch-size
        for i in range(rounds):

            if verbose is True:
                print(
                    "learning rate = "
                    + str(lr[i])
                    + ", batch size = "
                    + str(batch_size[i])
                )

            # set learning rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr[i]

            # dataset and dataloader
            dataset = TensorDataset(
                training_theta, training_N, torch.unsqueeze(training_mag[:, f], -1)
            )
            training_data, validation_data = torch.utils.data.random_split(
                dataset,
                [
                    int(len(dataset) * (1.0 - validation_split)),
                    len(dataset) - int(len(dataset) * (1.0 - validation_split)),
                ],
            )
            validation_theta, validation_N, validation_mag = validation_data[:]
            training_dataloader = DataLoader(
                training_data,
                shuffle=True,
                batch_size=batch_size[i],
                num_workers=8,
                pin_memory=True,
                persistent_workers=True,
            )
            epochs_per_step = 1.0 / len(training_dataloader)
            epoch = 0.0

            # set up training loss
            training_loss = [np.inf]
            validation_loss = [np.inf]
            best_loss = np.inf
            best_state = photulator.state_dict()
            patience_counter = 0

            # which training step to use?
            if loss_in == "absmag":
                compute_loss = lambda theta, N, mag: torch.sqrt(
                    torch.mean(
                        torch.square(torch.subtract(photulator.forward(theta), mag))
                    )
                )
            elif loss_in == "asinhmag":
                # compute_loss = lambda theta, N, mag: torch.sqrt(torch.mean( torch.square(torch.subtract(flux2asinhmag(photulator.flux(theta, N) * 1e9, photulator.f_b), mag)) ))
                compute_loss = lambda theta, N, mag: torch.sqrt(
                    torch.mean(
                        torch.square(
                            torch.subtract(photulator.luptitudes(theta, N), mag)
                        )
                    )
                )

            # loop over epochs
            while patience_counter < patience and epoch < maxepochs:

                # loop over batches for a single epoch
                for theta, N, mag in training_dataloader:

                    # zero gradients
                    optimizer.zero_grad()

                    # backprop and step
                    loss = compute_loss(
                        theta.to(device, non_blocking=True),
                        N.to(device, non_blocking=True),
                        mag.to(device, non_blocking=True),
                    )
                    loss.backward()
                    optimizer.step()

                    # increment epoch
                    epoch += epochs_per_step

                    # update wandb if needed
                    if wandb_init is not None:
                        wandb.log(
                            {"train_loss": loss.detach().cpu().item(), "epoch": epoch}
                        )

                # compute total loss and validation loss
                validation_loss.append(
                    compute_loss(
                        validation_theta.to(device, non_blocking=True),
                        validation_N.to(device, non_blocking=True),
                        validation_mag.to(device, non_blocking=True),
                    )
                    .cpu()
                    .detach()
                    .numpy()
                )

                # early stopping condition
                if validation_loss[-1] < best_loss:
                    best_loss = validation_loss[-1]
                    best_state = photulator.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    photulator.load_state_dict(best_state)
                    torch.save(best_state, save_location + "_state.pt")
                    if verbose is True:
                        print("Validation loss = " + str(best_loss))
                    break

                # update wandb if needed
                if wandb_init is not None:
                    wandb.log(
                        {
                            "val_loss": validation_loss[-1],
                            "best_loss": best_loss,
                            "patience_counter": patience_counter,
                            "epoch": epoch,
                        }
                    )

        if wandb_init is not None:
            wandb.finish()

        # save CPU and GPU versions of the model
        if activation == "alsing20":
            photulator_cpu = Photulator(
                n_parameters=training_theta.shape[-1],
                filters=[filters[f]],
                parameters_shift=parameters_shift,
                parameters_scale=parameters_scale,
                magnitudes_shift=magnitudes_shift[f],
                magnitudes_scale=magnitudes_scale[f],
                n_hidden=[n_units] * n_layers,
                f_b=f_b[f],
                sigma_init=sigma_init,
                parameter_names=parameter_names,
            ).to("cpu")
        else:
            photulator_cpu = PhotulatorBasic(
                n_parameters=training_theta.shape[-1],
                filters=[filters[f]],
                parameters_shift=parameters_shift,
                parameters_scale=parameters_scale,
                magnitudes_shift=magnitudes_shift[f],
                magnitudes_scale=magnitudes_scale[f],
                n_hidden=[n_units] * n_layers,
                f_b=f_b[f],
                sigma_init=sigma_init,
                parameter_names=parameter_names,
                activation=activation_functions[activation],
            ).to("cpu")
        best_state_cpu = torch.load(
            save_location + "_state.pt", map_location=torch.device("cpu"), weights_only=False
        )
        photulator_cpu.load_state_dict(best_state_cpu)
        torch.save(photulator_cpu, save_location + "_cpu.pt")
        torch.save(photulator_cpu.state_dict(), save_location + "_state_cpu.pt")

        photulator_gpu = photulator_cpu.to("cuda")
        torch.save(photulator_gpu, save_location + "_gpu.pt")
        torch.save(photulator_gpu.state_dict(), save_location + "_state_gpu.pt")


### magnitude conversion functions ###
def flux2mag(flux):
    """
    Convert flux to apparent magnitude.

    Parameters
    ----------
    flux : torch.tensor
        Flux in nanomaggies.

    Returns
    -------
    mag : torch.Tensor
        AB magnitude.
    """
    return -2.5 * torch.log10(flux) + 22.5


def flux2asinhmag(flux, f_b):
    """
    Convert flux to asinh magnitude.
    
    Parameters
    ----------
    flux : torch.Tensor 
        Flux in nanomaggies.
    f_b : torch.Tensor
        Flux softening parameter (b in Lupton+99 eq. 3). Sets the flux
        below which asinh mag becomes approximately linear (rather than
        logarithmic. Units of nanomaggies.

    Returns
    -------
    asinh_mag : torch.Tensor
        Asinh magnitudes.
    """

    asinh_mag = -1.0857362047581294 * (
        torch.arcsinh(flux / (2.0 * f_b)) - torch.log(10**9 / f_b)
    )

    return asinh_mag


def asinhmag2flux(asinh_mag, f_b):
    """
    Convert asinh magnitude to flux.
    
    Parameters
    ----------
    asinh_mag : torch.Tensor 
        Asinh magnitude assuming a softening of `f_b`.
    f_b : torch.Tensor
        Flux softening parameter (b in Lupton+99 eq. 3). Sets the flux
        below which asinh mag becomes approximately linear (rather than
        logarithmic. Units of nanomaggies.

    Returns
    -------
    flux : torch.Tensor
        Flux in nanomaggies.
    """
    return (
        torch.sinh(-(asinh_mag / -1.0857362047581294) + torch.log(10**9 / f_b))
        * 2
        * f_b
    )


def mag2asinhmag(mag, f_b):
    """
    Convert magnitude to asinh magnitude.
    
    Parameters
    ----------
    mag : torch.Tensor 
        Logarithmic AB magnitude.
    f_b : torch.Tensor
        Flux softening parameter (b in Lupton+99 eq. 3). Units of nanomaggies.

    Returns
    -------
    asinh_mag : torch.Tensor
        Asinh magnitudes.
    """
    return flux2asinhmag(10 ** (-0.4 * (mag - 22.5)), f_b)


def asinhmag2mag(asinhmag, f_b):
    """
    Convert asinh magnitude to magnitude.
    
    Parameters
    ----------
    asinh_mag : torch.Tensor 
        Asinh magnitude assuming a softening of `f_b`.
    f_b : torch.Tensor
        Flux softening parameter (b in Lupton+99 eq. 3). Units of nanomaggies.

    Returns
    -------
    mag : torch.Tensor
        Logarithmic AB magnitude.
    """
    return flux2mag(
        torch.sinh(asinhmag / (-1.0857362047581294) + torch.log(10**9 / f_b))
        * 2.0
        * f_b
    )
