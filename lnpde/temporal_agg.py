from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange


Module = nn.Module
Tensor = torch.Tensor
Sequential = nn.Sequential


class IAttention(Module, ABC):
    @abstractmethod
    def forward(
        self,
        x: Tensor,
        return_weights: bool = True
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, None]:
        """Maps input sequence x to output sequence.
        Args:
            x: Input sequence. Has shape (S, M, K).
            return_weights: If True, returns attention weights. Otherwise, returns None.
        Returns:
            y: Output sequence. Has shape (S, M, K).
            attn_weights: Attention weights. Has shape (S, M, M).
                None is returned if `return_weights` is False.
        """
        pass

    @abstractmethod
    def update_time_grid(self, t: Tensor) -> None:
        """Updates all parts of the class that depend on time grids (except submodules
            which might also depend on time grids, those must be upated separately
            (see lnpde.rec_net)).
        Args:
            t: New time grids. Has shape (S, M, 1).
        """
        pass


class AttentionBase(IAttention):
    def __init__(self, d_model: int, rpe: Module | None = None, drop_prob: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.rpe = rpe
        self.drop_prob = drop_prob

    def forward(self, x: Tensor, return_weights: bool = True) -> tuple[Tensor, Tensor] | tuple[Tensor, None]:
        attn_weights = self._eval_attn_weights(x)
        output = self._eval_output(attn_weights, x)
        if return_weights:
            return output, attn_weights
        else:
            return output, None

    def drop(self, w: Tensor) -> Tensor:
        """Sets an element of w to -inf with probability self.drop_prob.
            Does not drop the diagonal and one of the neighboring elements.
        """

        dont_drop = torch.eye(w.shape[1], dtype=w.dtype, device=w.device)  # leave the diagonal

        inds = torch.arange(0, w.shape[1], 1, device=w.device)
        shift = torch.randint(low=0, high=2, size=(w.shape[1],), device=w.device)
        shift[0] = 1  # leave right neighbor for y1
        shift[-1] = -1  # leave left neighbor for yM
        shift[shift == 0] = -1  # randomly leave left or right neighbor for y2,...yM-1
        dont_drop[inds, inds + shift] = 1

        prob = torch.ones_like(w) * (1.0 - self.drop_prob)
        prob = torch.clip(prob + dont_drop, 0, 1)

        mask = torch.bernoulli(prob)  # 1 - don't drop, 0 - drop
        mask[mask == 0] = torch.inf
        mask[mask == 1] = 0

        return w - mask

    def update_time_grid(self, t: Tensor) -> None:
        pass

    def _eval_attn_weights(self, x: Tensor) -> Tensor:
        raise NotImplementedError()

    def _eval_output(self, attn_weights: Tensor, x: Tensor) -> Tensor:
        raise NotImplementedError()


class TemporalDotProductAttention(AttentionBase):
    def __init__(
        self,
        d_model: int,
        t: Tensor,
        eps: float,
        delta_r: float,
        p: float,
        rpe: Module | None = None,
        drop_prob: float = 0.0,
        **kwargs,
    ) -> None:

        super().__init__(d_model, rpe, drop_prob)
        self.eps = eps
        self.delta_r = delta_r
        self.p = p if p != -1 else torch.inf

        self.W_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_v = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_q = nn.Linear(self.d_model, self.d_model, bias=False)

        self.unnorm_temporal_attn_weights: Tensor
        self.update_time_grid(t)

    def _eval_attn_weights(self, x: Tensor) -> Tensor:
        Q, K = self.W_q(x), self.W_k(x)
        unnorm_dotprod_attn_weights = torch.bmm(Q, torch.transpose(K, 1, 2)) / self.d_model**0.5
        if self.training:
            attn_weights = nn.Softmax(-1)(self.drop(unnorm_dotprod_attn_weights + self.unnorm_temporal_attn_weights))
        else:
            attn_weights = nn.Softmax(-1)(unnorm_dotprod_attn_weights + self.unnorm_temporal_attn_weights)
        return attn_weights

    def _eval_output(self, attn_weights: Tensor, x: Tensor) -> Tensor:
        assert x.shape[0:2] == attn_weights.shape[0:2], (
            "Batch size and number of time points in `x` and `attn_weights` must be the same. "
            f"Currently {x.shape=} and {attn_weights.shape=}."
        )
        V = self.W_v(x)
        if self.rpe is None:
            output = torch.bmm(attn_weights, V)
        else:
            output = torch.bmm(attn_weights, V) + (attn_weights.unsqueeze(-1) * self.rpe()).sum(2)
        return output

    @torch.no_grad()
    def update_time_grid(self, t: Tensor) -> None:
        dt = torch.cdist(t, t, p=1).float()
        self.unnorm_temporal_attn_weights = np.log(self.eps) * torch.pow(dt / self.delta_r, self.p)


class RelativePositionalEncoding(Module):
    def __init__(self, f: Module | Sequential, t: Tensor, delta_r: float, **kwargs):
        super().__init__()

        self.f = f
        self.delta_r = delta_r
        self.squish_fn = nn.Hardtanh()

        self.update_time_grid(t)

    def forward(self) -> Tensor:
        rpe = self.f(self.dt_prime_mat)
        return rpe

    def update_time_grid(self, t: Tensor) -> None:
        # t: Tensor, shape (S, M, 1).
        dt_mat = self._get_dt_matrix(t)
        self.dt_prime_mat = self.squish_fn(dt_mat / self.delta_r).float()

    def _get_dt_matrix(self, t: Tensor) -> Tensor:
        """Calculates the matrix of relative distances between all time points in `t`."""
        dist_mat = torch.cdist(t, t, p=1)  # (S, M, M)
        dir_mat = torch.ones_like(dist_mat).triu() - torch.ones_like(dist_mat).tril()  # (S, M, M)
        dt_mat = (dir_mat * dist_mat).unsqueeze(-1)  # (S, M, M, 1)
        return dt_mat


class TFEncoder(nn.Module):
    # Modified https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoderLayer
    def __init__(
        self,
        d_model: int,
        self_attn: Module,
        t: Tensor,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
        **kwargs,
    ) -> None:

        super().__init__()

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm([d_model], eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm([d_model], eps=layer_norm_eps)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = torch.nn.functional.relu  # type: ignore
        self.self_attn = self_attn

    def forward(self, x: Tensor) -> Tensor:

        # Post-norm
        x = self.norm1(x + self._sa_block(x))
        x = self.norm2(x + self._ff_block(x))

        # Pre-norm
        # x = x + self._sa_block(self.norm1(x))
        # x = x + self._ff_block(self.norm2(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor) -> Tensor:
        x = self.self_attn(x, return_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TemporalAggregator(Module):
    """Computes temporal aggregations.

    Args:
        f: Mapping from input sequence with shape (S, M, K) to output sequence
            with the same shape.
    """
    def __init__(self, f: Module) -> None:
        super().__init__()
        self.f = f

    def forward(self, t: Tensor, u: Tensor) -> Tensor:
        """Computes temporal aggregations of `u` at time points `t`.

        Args:
            t: Time grids. Has shape (S, M, 1).
            u: Input sequences. Has shape (S, M, N, agg_dim).

        Returns:
            Temporal aggregations. Has shape (S, M, N, agg_dim).
        """
        S, _, N, _ = u.shape

        self.update_time_grids(torch.repeat_interleave(t, N, dim=0))
        u = rearrange(u, "s m n agg_dim -> (s n) m agg_dim")
        u = rearrange(self.f(u), "(s n) m agg_dim -> s m n agg_dim", s=S, n=N)
        return u

    def update_time_grids(self, t: Tensor) -> None:
        """Updates all parts of `self.f` that depend on time grids."""
        for module in self.f.modules():
            if not hasattr(module, "update_time_grid"):
                continue
            if callable(getattr(module, "update_time_grid")):
                module.update_time_grid(t)  # type: ignore
