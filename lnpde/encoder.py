import torch
# import torch.nn as nn

from lnpde.interp import Interpolator
from lnpde.spatial_agg import SpatialAggregator
from lnpde.temporal_agg import TemporalAggregator
from lnpde.read_func import ReadFunction


Module = torch.nn.Module
Tensor = torch.Tensor
ModuleList = torch.nn.ModuleList


class Encoder(Module):
    def __init__(
        self,
        hI: Interpolator,
        hS: SpatialAggregator,
        hT: TemporalAggregator,
        hR: ReadFunction,
    ) -> None:
        super().__init__()
        self.hI = hI
        self.hS = hS
        self.hT = hT
        self.hR = hR

    def forward(self, t: Tensor, x: Tensor, u: Tensor, Phi: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
        u_neighb = self.hI(u, Phi, b)
        u_spatial = self.hS(u, u_neighb)
        u_spatiotemp = self.hT(t, u_spatial)
        gamma, tau = self.hR(u_spatiotemp)
        return gamma, tau


# class SpatioTemporalAggBlock(Module):
#     def __init__(
#         self,
#         hI: Interpolator,
#         hS: SpatialAggregator,
#         hT: TemporalAggregator,
#         lin_proj: Module,
#     ) -> None:
#         super().__init__()
#         self.hI = hI
#         self.hS = hS
#         self.hT = hT
#         self.lin_proj = lin_proj  # D_agg -> D_zeta

#     def forward(self, t: Tensor, x: Tensor, u: Tensor, Phi: Tensor, b: Tensor) -> tuple[Tensor]:
#         u_neighb = self.hI(u, Phi, b)
#         u_spatial = self.hS(u, u_neighb)
#         u_spatiotemp = self.hT(t, u_spatial)
#         zeta = self.lin_proj(u_spatiotemp)
#         return zeta


# class Encoder2(Module):
#     def __init__(self, st_blocks: ModuleList, hR: ReadFunction, D_zeta: int, last_residual=False) -> None:
#         super().__init__()
#         self.st_blocks = st_blocks
#         self.hR = hR
#         self.D_zeta = D_zeta
#         self.layer_norms = nn.ModuleList([nn.LayerNorm([self.D_zeta]) for _ in range(len(self.st_blocks))])
#         self.last_residual = last_residual

#     def forward(self, t: Tensor, x: Tensor, u: Tensor, Phi: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
#         zeta = torch.zeros(u.shape[0], u.shape[1], u.shape[2], self.D_zeta, device=u.device)
#         for i in range(len(self.st_blocks)):
#             u_zeta = concat_fields(u, zeta)  # (S, M, N, D + D_zeta)

#             # zeta = self.st_blocks[i](t, x, u_zeta, Phi, b)  # (S, M, N, D_zeta)

#             # zeta = self.st_blocks[i](t, x, u_zeta, Phi, b)  # (S, M, N, D_zeta)
#             # zeta = self.layer_norms[i](zeta)

#             # zeta = zeta + self.st_blocks[i](t, x, u_zeta, Phi, b)  # (S, M, N, D_zeta)
#             # zeta = self.layer_norms[i](zeta)

#             if i == len(self.st_blocks) - 1 and self.last_residual:
#                 zeta = zeta + self.st_blocks[i](t, x, u_zeta, Phi, b)
#             else:
#                 zeta = self.st_blocks[i](t, x, u_zeta, Phi, b)
#         gamma, tau = self.hR(zeta)

#         return gamma, tau


# def concat_fields(u: Tensor, zeta: Tensor) -> Tensor:
#     return torch.cat([u, zeta], dim=-1)
