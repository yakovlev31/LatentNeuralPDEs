from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from lnpde.trans_func import ITransitionFunction
from lnpde.encoder import Encoder
from lnpde.utils.utils import extract_time_grids


Tensor = torch.Tensor
Module = nn.Module
ParameterDict = nn.ParameterDict


class IVariationalPosterior(ABC, Module):
    @property
    @abstractmethod
    def posterior_param(self) -> nn.ParameterDict:
        """Returns parameters of the posterior distribution."""
        pass

    @abstractmethod
    def sample_s(
        self,
        t: Tensor,
        u: Tensor,
        block_size: int,
        Phi_h: Tensor, b_h: Tensor,
        Phi_F: Tensor, b_F: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Samples shooting variables (s_1, ..., s_B) from the posterior q(s|u).
        Also returns states (z_1, ..., z_M).
        Args:
            t: Time points at which to evaluate the latent states. Has shape (S, M, 1).
            u: Observations corresponding to the latent states. Used only for
                amortized variational inference. Has shape (S, M, N, D).
            block_size: Size of the blocks.
            Phi_*: Interpolation matrices. Has shape (S, N_eval, N).
            b_*: Boundary condition vectors. Has shape (S, N_eval, 1).
        Returns:
            A sample of the shooting variables with shape (S, B, N, d)
                and the corresponding latent states with shape (S, M, N, d).
        """
        pass

    @abstractmethod
    def sample_theta(self) -> dict[str, Tensor]:
        """Samples parameters of g and F from the posterior.
        Returns:
            Dictionary with a sample of the parameters.
        """
        pass


class VariationalPosteriorBase(IVariationalPosterior):
    def __init__(self, posterior_param_dict: ParameterDict):
        super().__init__()
        self._check_param_shapes(posterior_param_dict)
        self._posterior_param = posterior_param_dict

    @property
    def posterior_param(self):
        return self._posterior_param

    def sample_theta(self):
        mu_g, sig_g = self.posterior_param["mu_theta_g"], torch.exp(self.posterior_param["log_sig_theta_g"])
        mu_F, sig_F = self.posterior_param["mu_theta_F"], torch.exp(self.posterior_param["log_sig_theta_F"])
        theta = {
            "theta_g": mu_g + sig_g * torch.randn_like(sig_g),
            "theta_F": mu_F + sig_F * torch.randn_like(sig_F),
        }
        return theta

    def _check_param_shapes(self, p: ParameterDict) -> None:
        raise NotImplementedError()

    def sample_s(self, t: Tensor, u: Tensor, block_size: int) -> tuple[Tensor, Tensor]:
        raise NotImplementedError()


class AmortizedMultipleShootingPosterior(VariationalPosteriorBase):
    def __init__(
        self,
        posterior_param_dict: ParameterDict,
        h: Encoder,
        F: ITransitionFunction,
    ) -> None:

        super().__init__(posterior_param_dict)
        self.h = h
        self.F = F

        self.gamma: Tensor
        self.tau: Tensor

    def sample_s(
        self,
        t: Tensor,
        u: Tensor,
        block_size: int,
        Phi_h: Tensor, b_h: Tensor,
        Phi_F: Tensor, b_F: Tensor,
    ) -> tuple[Tensor, Tensor]:

        gamma, tau = self.h(t, t, u, Phi_h, b_h)  # coords x not used, so just pass t
        self.gamma, self.tau = gamma[:, :-1, :, :], tau[:, :-1, :, :]
        gamma, tau = self.gamma[:, ::block_size, :, :], self.tau[:, ::block_size, :, :]

        s = gamma + tau * torch.randn_like(tau)

        S, N, d = u.shape[0], u.shape[2], s.shape[3]
        M, B = u.shape[1], s.shape[1]

        z = torch.zeros((S, M, N, d), device=tau.device)

        z[:, [0], :, :] = s[:, [0], :, :]
        z[:, 1:, :, :] = self.F(s, extract_time_grids(t, n_blocks=B), Phi_F, b_F)

        return s, z

    def _check_param_shapes(self, p: dict[str, Tensor]) -> None:
        model_parameter_names = ["mu_theta_g", "mu_theta_F", "log_sig_theta_g", "log_sig_theta_F"]
        for param_name in model_parameter_names:
            assert len(p[param_name].shape) == 1, f"{param_name} must have shape (num_parameters, ) but has {p[param_name].shape}."
