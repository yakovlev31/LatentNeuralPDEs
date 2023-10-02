from types import SimpleNamespace

import torch
import torch.nn as nn

from lnpde.temporal_agg import TemporalDotProductAttention, RelativePositionalEncoding, TFEncoder


Tensor = torch.Tensor
Sequential = nn.Sequential


def get_stencil_size(stencil_shape: str) -> int:
    sizes = {
        "2cross": 4,
        "4cross": 8,
        "2square": 8,
        "4square": 16,
        "1circle": 8 * 1,
        "2circle": 8 * 2,
        "3circle": 8 * 3,
        "4circle": 8 * 4,
        "5circle": 8 * 5,
        "6circle": 8 * 6,
        "7circle": 8 * 7,
        "8circle": 8 * 8,
    }
    return sizes[stencil_shape]


def create_temporal_aggregation_function(param: SimpleNamespace) -> Sequential:
    t_init = torch.linspace(0, 1, 3).view(1, -1, 1)
    pos_enc_args = {
        "d_model": param.D_agg,
        "t": t_init,
        "delta_r": param.delta_T,
        "f": nn.Linear(1, param.D_agg, bias=False),
    }
    attn_args = {
        "d_model": param.D_agg,
        "t": t_init,
        "eps": 1e-2,
        "delta_r": param.delta_T,
        "p": param.p,
        "drop_prob": param.drop_prob,
    }
    modules = []
    pos_enc = RelativePositionalEncoding(**pos_enc_args)
    for _ in range(param.n_tf_enc_layers):
        self_attn = TemporalDotProductAttention(rpe=pos_enc, **attn_args)
        tf_enc_block = TFEncoder(
            d_model=param.D_agg,
            self_attn=self_attn,
            t=t_init,
            dim_feedforward=2*param.D_agg,
        )
        modules.append(tf_enc_block)
    return nn.Sequential(*modules)


def extract_time_grids(t: Tensor, n_blocks: int) -> Tensor:
    """Extracts overlapping sub-grids from `t` for the given number of blocks.
    Args:
        t: Full time grids. Has shape (S, M, 1).
        n_blocks: Number of blocks.
    Returns:
        sub_t: Overlapping sub-grids. Has shape (S, n_blocks, grid_size).
    Simplified example:
        For t=(t1, t2, t3, t4, t5) and b_blocks=2 returns (t1, t2, t3), (t3, t4, t5).
    """

    S, M = t.shape[0:2]
    assert (M - 1) % n_blocks == 0, "All blocks must be of equal size."

    grid_size = int((M - 1) / n_blocks) + 1
    sub_t = torch.empty((S, n_blocks, grid_size), dtype=t.dtype, device=t.device)

    for b, i in enumerate(range(0, M-grid_size+1, grid_size-1)):
        sub_t[:, [b], :] = torch.transpose(t[:, i:i+grid_size, :], 1, 2)

    return sub_t
