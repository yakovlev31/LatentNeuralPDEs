import torch


Module = torch.nn.Module
Tensor = torch.Tensor


class ReadFunction(Module):
    """Maps aggregations to parameters of the approximate posteriors.

    Args:
        f_gamma: Mapping from aggregation to the mean.
        f_tau: Mapping from aggregation to log std.
        tau_min: Minimum standard deviation of the approximate posteriors.
    """
    def __init__(self, f_gamma: Module, f_tau: Module, tau_min: float) -> None:
        super().__init__()
        self.f_gamma = f_gamma
        self.f_tau = f_tau
        self.tau_min = tau_min

    def forward(self, u: Tensor) -> tuple[Tensor, Tensor]:
        """Maps aggregations to parameters of the approximate posteriors.

        Args:
            u: Spatiotemporal aggregations. Has shape (S, M, N, D_agg).

        Returns:
            Mean and std of the approximate posteriros. Both have shape (S, M, N, d).
        """

        gamma = self.f_gamma(u)
        tau = torch.exp(self.f_tau(u)) + self.tau_min
        return gamma, tau
