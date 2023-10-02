from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from torchdiffeq import odeint
from torchdiffeq import odeint_adjoint

from einops import rearrange

from lnpde.interp import Interpolator
from lnpde.spatial_agg import SpatialAggregator


Tensor = torch.Tensor
Module = nn.Module
Parameter = nn.parameter.Parameter


class ITransitionFunction(Module, ABC):
    @abstractmethod
    def forward(self, s: Tensor, t: Tensor) -> Tensor:
        """Moves latent states forward in time.
        Args:
            s: Latent states. Has shape (S, B, N, d).
            t: Time grids at which the latent states are evaluated (except the first time point).
                The first time point of each time grid must be the temporal position
                of the corresponding latent state. Has shape (S, B, block_size+1).
        Returns:
            s_new: New latent states. Has shape (S, B*block_size, N, d).
        """
        pass

    @abstractmethod
    def set_param(self, param: Tensor) -> None:
        """Sets parameters of the module to `params`.
        Args:
            param: New parameter values.
        """
        pass

    @abstractmethod
    def param_count(self) -> int:
        """Calculates the number of parameters over which the posterior is to be evaluated.
        Returns:
            The number of parameters.
        """
        pass


class NeuralTransitionFunctionBase(ITransitionFunction):
    def __init__(self, f: Module, layers_to_count: list = []):
        super().__init__()
        self.f = f
        self.layers_to_count = [nn.Linear, nn.LayerNorm, nn.BatchNorm1d]  # default
        self.layers_to_count.extend(layers_to_count)  # user-specified (must contain weight and bias)

    def forward(self, s: Tensor, t: Tensor) -> Tensor:
        raise NotImplementedError()

    def set_param(self, param: Tensor) -> None:
        assert self.param_count() == param.numel(), (
            f"The size of param ({param.numel()}) must be the same as self.param_count()"
            f"({self.param_count()})"
        )
        layers = self._get_layers(self.f, self.layers_to_count)
        self._set_layer_param_to_vec(layers, param)

    def param_count(self) -> int:
        """Each layer must contain weight and bias variables."""
        param_count = 0
        layers = self._get_layers(self.f, self.layers_to_count)
        for layer in layers:
            self._check_weight_and_bias_of_layer(layer)
            layer_param_count = layer.weight.numel() + layer.bias.numel()
            param_count += layer_param_count
        return param_count

    def _get_layers(self, f, layer_types: list) -> list:
        """Returns all layers in `f` whose type is present in `layer_types`.
        Args:
            layer_types: A list with the requred layer types (e.g. [nn.Linear]).
        Returns:
            A list of layers in `f` whose types are in `layer_types`
        """
        return_layers = []
        for fi in f.modules():
            if type(fi) in layer_types:
                return_layers.append(fi)
        return return_layers

    def _set_layer_param_to_vec(self, layers: list[Module], vec: torch.Tensor) -> None:
        """Sets parameters of Modules in `layers` to elements of `vec`.
        Args:
            layers: A list of Modules whose parameters need to be set.
            vec: A 1D Tensor with the parameters.
        """
        pointer = 0
        for layer in layers:
            self._check_weight_and_bias_of_layer(layer)

            layer_param_count = layer.weight.numel() + layer.bias.numel()  # type: ignore
            layer_weight_count = layer.weight.numel()  # type: ignore

            layer_param = vec[pointer:pointer + layer_param_count]
            layer_weight = layer_param[:layer_weight_count].view_as(layer.weight)  # type: ignore
            layer_bias = layer_param[layer_weight_count:].view_as(layer.bias)  # type: ignore

            self._del_set_layer_attr(layer, "weight", layer_weight)
            self._del_set_layer_attr(layer, "bias", layer_bias)

            pointer += layer_param_count

    def _del_set_layer_attr(self, layer, attr_name, attr_val):
        delattr(layer, attr_name)
        setattr(layer, attr_name, attr_val)

    def _check_weight_and_bias_of_layer(self, layer: Module) -> None:
        assert (type(layer.weight) is Tensor or type(layer.weight) is Parameter), (
            f"weight of layer {layer} must be Tensor or Parameter.")
        assert (type(layer.bias) is Tensor or type(layer.bias) is Parameter), (
            f"bias of layer {layer} must be Tensor or Parameter.")


class ODETransitionFunction(NeuralTransitionFunctionBase):
    def __init__(self, f: Module, layers_to_count: list = [], solver_kwargs: dict = {}):
        super().__init__(f, layers_to_count=layers_to_count)

        if "adjoint" in solver_kwargs.keys():
            self.adjoint = solver_kwargs["adjoint"] == 1
            del solver_kwargs["adjoint"]
        else:
            self.adjoint = False

        self.solver_kwargs = solver_kwargs

    def forward(self, s: Tensor, t: Tensor, Phi: Tensor, b: Tensor) -> Tensor:
        S, B, N, d, block_size = *s.shape, t.shape[2] - 1
        s_new = torch.zeros((S, B, block_size, N, d), dtype=s.dtype, device=s.device)

        delta = torch.diff(t, dim=2).to(s.dtype)
        t_sim = torch.tensor([0., 1.], dtype=t.dtype, device=t.device)

        for i in range(block_size):

            if i == 0:
                s0 = s.unsqueeze(2)
            else:
                s0 = s_new[:, :, [i-1], :, :]

            if self.adjoint is True:
                f = DynModule(self.f, Phi, b, delta[:, :, [i], None])
                s_new[:, :, i, :, :] = odeint_adjoint(f, s0.squeeze(2), t_sim, **self.solver_kwargs, adjoint_options={"norm": "seminorm"})[-1]  # type: ignore

            else:
                f = self.get_scaled_dynamics_function(delta[:, :, [i], None], Phi, b)
                s_new[:, :, i, :, :] = odeint(f, s0.squeeze(2), t_sim, **self.solver_kwargs)[-1]  # type: ignore

        return rearrange(s_new, "S B block_size N d -> S (B block_size) N d")

    def get_scaled_dynamics_function(self, delta, Phi, b):

        def scaled_dynamics_function(t, x):
            return self.f(t, x, Phi, b) * delta

        return scaled_dynamics_function


class DynamicsFunction(Module):
    def __init__(self, hI: Interpolator, hS: SpatialAggregator) -> None:
        super().__init__()
        self.hI = hI
        self.hS = hS

    def forward(self, t: Tensor, z: Tensor, Phi: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
        z_neighb = self.hI(z, Phi, b)
        z_spatial = self.hS(z, z_neighb)
        return z_spatial


class DynModule(Module):
    def __init__(self, f, Phi, b, delta):
        super().__init__()
        self.f = f
        self.Phi = Phi
        self.b = b
        self.delta = delta

    def forward(self, t, x):
        return self.f(t, x, self.Phi, self.b) * self.delta
