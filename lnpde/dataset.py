from typing import Union
import torch
from torch.utils.data import Dataset


Tensor = torch.Tensor


class SpatiotemporalDataset(Dataset):
    """Class for spatiotemporal data.

    Args:
        t: Time grids. Has shape (S, M, 1).
        x: Coordinates. Has shape (S, N, 2).
        u: Observations. Has shape (S, M, N, D).
        Phi_h: Interpolation matrices for encoder. Has shape (S, N_eval, N).
        b_h: Boundary condition vectors for encoder. Has shape (S, N_eval, 1).
        Phi_F: Interpolation matrices for transition function. Has shape (S, N_eval, N).
        b_F: Boundary condition vectors for transition function. Has shape (S, N_eval, 1).
        max_len: Length of the subtrajectory selected from each trajectory.
    """
    def __init__(
        self,
        t: Tensor, x: Tensor, u: Tensor,
        Phi_h: Tensor, b_h: Tensor,
        Phi_F: Tensor, b_F: Tensor,
        max_len: Union[None, int] = None
    ) -> None:
        self.t = t
        self.x = x
        self.u = u
        self.Phi_h = Phi_h
        self.b_h = b_h
        self.Phi_F = Phi_F
        self.b_F = b_F
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.t)

    def __getitem__(self, idx: int) -> tuple[Tensor, ...]:
        t = self.t[idx]
        x = self.x[idx]
        u = self.u[idx]
        Phi_h = self.Phi_h[idx]
        b_h = self.b_h[idx]
        Phi_F = self.Phi_F[idx]
        b_F = self.b_F[idx]
        traj_ind = torch.tensor(idx, dtype=torch.long)

        if self.max_len is not None:
            t, u = t[:self.max_len], u[:self.max_len]

        return t, x, u, Phi_h, b_h, Phi_F, b_F, traj_ind
