import torch
from einops import rearrange


Module = torch.nn.Module
Tensor = torch.Tensor


class SpatialAggregator(Module):
    """Aggregates spatial information from `u` and its interpolation `u_neighb`.

    Args:
        f: Mapping from a node value and its neighbors' values to spatial aggregation.
            Input size is D_u * (1 + N_neighb), where D_u is dimensionality of `u`,
            and N_neighb is the number of neighbors for each node.
    """
    def __init__(self, f: Module) -> None:
        super().__init__()
        self.f = f

    def forward(self, u: Tensor, u_neighb: Tensor) -> Tensor:
        """Computes spatial aggregations.

        Args:
            u: Observations. Has shape (S, M, N, D).
            u_neighb: Interpolation of `u` evaluated at spatial neighborhood nodes.
                Has shape (S, M, N_eval, D).

        Returns:
            Spatial aggregations. Has shape (S, M, N, agg_dim).
        """
        assert u_neighb.shape[2] % u.shape[2] == 0, "Each point must have the same number of neighbors."
        u_full = torch.cat(
            (
                u,
                rearrange(u_neighb, "s m (n n_neighb) d -> s m n (n_neighb d)", n=u.shape[2], n_neighb=int(u_neighb.shape[2] / u.shape[2])),
            ),
            dim=-1
        )
        output = self.f(u_full)
        return output
