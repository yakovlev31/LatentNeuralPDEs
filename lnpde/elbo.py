from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from lnpde.model import IModel
from lnpde.posterior import IVariationalPosterior, AmortizedMultipleShootingPosterior

from einops import repeat


Tensor = torch.Tensor


class IELBO(nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        t: Tensor,
        u: Tensor,
        block_size: int,
        scaler: float,
        Phi_h: Tensor, b_h: Tensor,
        Phi_F: Tensor, b_F: Tensor,
    ) -> tuple[Tensor, ...]:
        """Evaluates ELBO for the observations u.
        Args:
            t: Time grid for the observations. Has shape (S, M, 1).
            u: A batch of observations. Has shape (S, M, N, D).
            block_size: Block size.
            scaler: Scaler for KL(q(s_i)||p(s_i|s_i-1)) terms.
            Phi_*: Interpolation matrices. Has shape (S, N_eval, N).
            b_*: Boundary condition vectors. Has shape (S, N_eval, 1).
        Returns:
            Parts of the ELBO (L1, L2, L3), states, and shooting variables.
        """
        pass


class ELBOBase(IELBO):
    def __init__(
        self,
        p: IModel,
        q: IVariationalPosterior,
    ) -> None:

        super().__init__()
        self.p = p
        self.q = q

    def forward(
        self,
        t: Tensor,
        u: Tensor,
        block_size: int,
        scaler: float,
        Phi_h: Tensor, b_h: Tensor,
        Phi_F: Tensor, b_F: Tensor,
    ) -> tuple[Tensor, ...]:

        # Sample approximate posterior.
        self.p.set_theta(self.q.sample_theta())
        s, z = self.q.sample_s(t, u, block_size, Phi_h, b_h, Phi_F, b_F)

        # Calculate parts of ELBO.
        L1 = self.calc_L1(z, u)
        L2 = self.calc_L2(z, block_size, scaler)
        L3 = self.calc_L3()

        return L1, L2, L3, z, s

    def calc_L1(self, z: Tensor, u: Tensor) -> Tensor:
        return self.p.loglik(u, z).sum()

    def calc_L2(self, z: Tensor, block_size: int, scaler: float) -> Tensor:
        raise NotImplementedError()

    def calc_L3(self) -> Tensor:
        n = self.q.posterior_param["mu_theta_g"].numel()
        L3_0 = self.kl_norm_norm(
            self.q.posterior_param["mu_theta_g"],
            self.p.prior_param["mu_theta"].expand(n),
            torch.exp(self.q.posterior_param["log_sig_theta_g"]),
            self.p.prior_param["sig_theta"].expand(n),
        ).sum()

        n = self.q.posterior_param["mu_theta_F"].numel()
        L3_1 = self.kl_norm_norm(
            self.q.posterior_param["mu_theta_F"],
            self.p.prior_param["mu_theta"].expand(n),
            torch.exp(self.q.posterior_param["log_sig_theta_F"]),
            self.p.prior_param["sig_theta"].expand(n),
        ).sum()
        return L3_0 + L3_1

    def kl_norm_norm(self, mu0: Tensor, mu1: Tensor, sig0: Tensor, sig1: Tensor) -> Tensor:
        """Calculates KL divergence between two K-dimensional Normal
            distributions with diagonal covariance matrices.
        Args:
            mu0: Mean of the first distribution. Has shape (*, K).
            mu1: Mean of the second distribution. Has shape (*, K).
            std0: Diagonal of the covatiance matrix of the first distribution. Has shape (*, K).
            std1: Diagonal of the covatiance matrix of the second distribution. Has shape (*, K).
        Returns:
            KL divergence between the distributions. Has shape (*, 1).
        """
        assert mu0.shape == mu1.shape == sig0.shape == sig1.shape, (f"{mu0.shape=} {mu1.shape=} {sig0.shape=} {sig1.shape=}")
        a = (sig0 / sig1).pow(2).sum(-1, keepdim=True)
        b = ((mu1 - mu0).pow(2) / sig1**2).sum(-1, keepdim=True)
        c = 2 * (torch.log(sig1) - torch.log(sig0)).sum(-1, keepdim=True)
        kl = 0.5 * (a + b + c - mu0.shape[-1])
        return kl


class AmortizedMultipleShootingELBO(ELBOBase):
    def __init__(self, p: IModel, q: AmortizedMultipleShootingPosterior) -> None:
        super().__init__(p, q)
        self.q = q

    def calc_L2(self, z: Tensor, block_size: int, scaler: float) -> Tensor:
        gamma = self.q.gamma[:, ::block_size, :]
        tau = self.q.tau[:, ::block_size, :]

        z_sub = z[:, 0:-1:block_size, :]
        S, B, N, d = z_sub.shape

        L2_0 = self.kl_norm_norm(
            gamma[:, 0, :, :],
            repeat(self.p.prior_param["mu0"], "d -> s n d", s=S, n=N, d=d),
            tau[:, 0, :, :],
            repeat(self.p.prior_param["sig0"], "d -> s n d", s=S, n=N, d=d)
        ).sum()

        L2_1 = self.kl_norm_norm(
            gamma[:, 1:, :, :],
            z_sub[:, 1:, :, :],
            tau[:, 1:, :, :],
            repeat(self.p.prior_param["sig_c"], "() -> s b n d", s=S, b=B-1, n=N, d=d)
        ).sum()

        return L2_0 + scaler * L2_1
