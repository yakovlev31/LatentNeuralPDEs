from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from torch.distributions.normal import Normal

from einops import reduce

from lnpde.decoder import IDecoder
from lnpde.trans_func import ITransitionFunction
# from lnpde.utils.utils import extract_time_grids


Tensor = torch.Tensor
ParameterDict = nn.ParameterDict


class IModel(ABC, nn.Module):
    @property
    @abstractmethod
    def g(self) -> IDecoder:
        """Returns the decoder."""
        pass

    @property
    @abstractmethod
    def F(self) -> ITransitionFunction:
        """Returns the transition function."""
        pass

    @property
    @abstractmethod
    def prior_param(self) -> ParameterDict:
        """Returns parameters of prior distributions."""
        pass

    @abstractmethod
    def loglik(self, u: Tensor, z: Tensor) -> Tensor:
        """Evaluates log likelihood p(u|z) for each snapshot.
        Args:
            u: Observations. Has shape (S, M, N, D).
            z: Latent states. Has shape (S, M, N, d).
        Returns:
            Log likelihood for each snapshot. Has shape (S, M, 1).
        """
        pass

    @abstractmethod
    def set_theta(self, theta: dict[str, Tensor]) -> None:
        """Sets parameters of g and F to theta["theta_g"] and theta["theta_F"] respectively.
        Args:
            theta: Dictionary with new parameter values. Must contain keys
                theta_g and theta_F.
        """
        pass


class ModelBase(IModel):
    def __init__(
        self,
        prior_param_dict: ParameterDict,
        g: IDecoder,
        F: ITransitionFunction,
    ) -> None:
        super().__init__()
        self._check_param_shapes(prior_param_dict)
        self._prior_param = prior_param_dict
        self._g = g
        self._F = F

    @property
    def g(self) -> IDecoder:
        return self._g

    @property
    def F(self) -> ITransitionFunction:
        return self._F

    @property
    def prior_param(self) -> ParameterDict:
        return self._prior_param

    def loglik(self, u: Tensor, z: Tensor) -> Tensor:
        return self._eval_loglik(u, z)

    def set_theta(self, theta: dict[str, Tensor]) -> None:
        self.g.set_param(theta["theta_g"])
        self.F.set_param(theta["theta_F"])

    def _check_param_shapes(self, d: ParameterDict) -> None:
        scalar_param_names = ["sig_c", "mu_theta", "sig_theta"]
        for param_name in scalar_param_names:
            assert d[param_name].shape == torch.Size([1]), f"{param_name} must have shape (1, ) but has {d[param_name].shape}."
        assert len(d["mu0"].shape) == 1, f"mu0 must have shape (K, ) but has {d['mu0'].shape}."
        assert len(d["sig0"].shape) == 1, f"sig0 must have shape (K, ) but has {d['sig0'].shape}."

    def _sample_lik(self, z: Tensor) -> Tensor:
        raise NotImplementedError()

    def _eval_loglik(self, u: Tensor, z: Tensor) -> Tensor:
        raise NotImplementedError()


class ModelNormal(ModelBase):
    def _sample_lik(self, z: Tensor) -> Tensor:
        param = self.g(z)
        mu, sig = param[..., 0], param[..., 1]
        y = Normal(mu, sig).rsample()
        return y

    def _eval_loglik(self, y: Tensor, x: Tensor) -> Tensor:
        param = self.g(x)
        mu, sig = param[..., 0], param[..., 1]
        loglik = Normal(mu, sig).log_prob(y)
        loglik = reduce(loglik, "s m n d -> s m ()", "sum")
        return loglik
