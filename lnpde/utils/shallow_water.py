import os
import pickle
import argparse
from collections import deque

from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

from lnpde.model import ModelNormal
from lnpde.posterior import AmortizedMultipleShootingPosterior
from lnpde.elbo import AmortizedMultipleShootingELBO

from lnpde.dataset import SpatiotemporalDataset

from lnpde.encoder import Encoder  # , Encoder2, SpatioTemporalAggBlock
from lnpde.trans_func import ODETransitionFunction, DynamicsFunction
from lnpde.decoder import NeuralDecoder

from lnpde.interp import Interpolator
from lnpde.spatial_agg import SpatialAggregator
from lnpde.temporal_agg import TemporalAggregator
from lnpde.read_func import ReadFunction

from lnpde.interp import coord_to_interp_data
from lnpde.utils.utils import get_stencil_size, create_temporal_aggregation_function, extract_time_grids


sns.set_style("whitegrid")


ndarray = np.ndarray
Tensor = torch.Tensor
Sequential = nn.Sequential
DataDict = dict[str, dict[str, ndarray]]
TensorDataDict = dict[str, dict[str, Tensor]]
Module = nn.Module


DATASET_NAME = "SHALLOW_WATER"


def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # Data.
    parser.add_argument("--data_folder", type=str, default="./experiments/data/datasets/shallow_water/partial/", help="Path to the dataset.")
    parser.add_argument("--D", type=int, default=1, help="Dimensionality of observed field.")
    parser.add_argument("--max_len", type=int, default=None, help="Truncation length for trajectories.")
    parser.add_argument("--sig_u", type=float, default=1e-3, help="Observation noise.")
    parser.add_argument("--sig_add", type=float, default=0, help="Added noise.")
    parser.add_argument("--train_frac", type=float, default=1.0, help="Fraction of training data to use, between 0 and 1.")

    # Model (common).
    parser.add_argument("--d", type=int, default=3, help="Latent space dimension.")
    parser.add_argument("--sig_c", type=float, default=1e-3, help="Standard deviation of the continuity prior.")
    parser.add_argument("--block_size", type=int, default=6, help="Number of time points in each block.")
    parser.add_argument("--boundary_cond", type=str, default="periodic", choices=["periodic", "free"], help="Boundary conditions for the spatial domain.")

    # Model (g).
    # ...

    # Model (F).
    parser.add_argument("--F_nonlin", type=str, default="relu", help="Nonlinearity for F.")
    parser.add_argument("--F_interp", type=str, default="linear", choices=["linear", "knn", "idw"], help="Interpolation method used by transition function.")
    parser.add_argument("--F_stencil_shape", type=str, default="2circle", help="Shape of the evaluation stencil used by transition function.")  # choices=["2cross", "4cross", "2square", "4square", "1circle", "2circle", "3circle"]
    parser.add_argument("--F_stencil_size", type=float, default=0.1, help="Size of the evaluation stencil used by transition function (maximum L1 distance from central node to neighbors")
    parser.add_argument("--F_hid_size", type=int, default=1024, help="Hidden size for the dynamics function.")
    parser.add_argument("--F_n_hid_layers", type=int, default=1, help="Number of hidden layers for the dynamics function.")

    # Model (h).
    # parser.add_argument("--D_zeta", type=int, default=2, help="Output dimension for spatiotemporal aggregation blocks.")
    # parser.add_argument("--n_isp_layers", type=int, default=4, help="Number of ISP layers.")

    parser.add_argument("--D_agg", type=int, default=128, help="Dimensionality of spatio-temporal aggregation for encoder.")
    parser.add_argument("--delta_T", type=float, default=0.1, help="Attention time span for temporal aggregation function.")
    parser.add_argument("--drop_prob", type=float, default=0.1, help="Attention dropout probability.")
    parser.add_argument("--p", type=float, default=-1, help="Exponent for temporal attention (use -1 for p=inf).")
    parser.add_argument("--n_tf_enc_layers", type=int, default=6, help="Number of TFEncoder layers for temporal aggregation function.")
    parser.add_argument("--h_interp", type=str, default="linear", choices=["linear", "knn", "idw"], help="Interpolation method used by encoder.")
    parser.add_argument("--h_stencil_shape", type=str, default="2circle", help="Shape of the evaluation stencil used by encoder.")  # choices=["2cross", "4cross", "2square", "4square", "1circle", "2circle", "3circle"]
    parser.add_argument("--h_stencil_size", type=float, default=0.1, help="Size of the spatial stencil used by encoder")
    parser.add_argument("--tau_min", type=float, default=1e-2, help="Lower bound on the variance of q(s_ij).")

    # Training/validation/testing.
    parser.add_argument("--scaler", type=float, default=1, help="Scaler for ELBO L2 term.")
    parser.add_argument("--n_iters", type=int, default=20000, help="Number of training iterations.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")

    parser.add_argument("--solver", type=str, default="dopri5", help="Name of the ODE solver (see torchdiffeq).")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance for ODE solver.")
    parser.add_argument("--atol", type=float, default=1e-4, help="Absolute tolerance for ODE solver.")
    parser.add_argument("--adjoint", type=int, default=0, help="Use adjoint to evaluate gradient flag (0 - no, 1 - yes).")

    parser.add_argument("--device", type=str, default="cuda", help="Device (cpu or cuda)")
    parser.add_argument("--seed", type=int, default=13, help="Random seed.")
    parser.add_argument("--group", default="None", help="Group for wandb.")
    parser.add_argument("--tags", default=["no_tag"], nargs="+", help="Tags for wandb.")
    parser.add_argument("--name", type=str, default="noname", help="Name of the run.")

    parser.add_argument("--visualize", type=int, default=0, help="Visualize predictions on validation set flag (0 - no, 1 - yes).")
    parser.add_argument("--n_mc_samples", type=int, default=10, help="Number of samples for Monte Carlo integration.")
    parser.add_argument("--delta_inf", type=float, default=0.21, help="Fraction of obsevations used for z0 inference at test time.")

    parser.add_argument("--model_folder", type=str, default="./models/shallow_water/", help="Folder for saving/loading models.")

    return parser


def create_datasets(param: SimpleNamespace, device=None) -> tuple[SpatiotemporalDataset, ...]:
    print("read_data()...")
    data = read_data(param.data_folder)

    print(f"Removing {(1.0 - param.train_frac)*100}% of training data.")
    data["train"] = remove_data(data["train"], param.train_frac)

    print("preprocess_data()...")
    data = preprocess_data(data)

    # print("add_noise()...")
    # data = add_noise(data, sig=param.sig_add, seed=param.seed)

    print("construct_interp_data()...")
    data = construct_interp_data(
        data, param.boundary_cond,
        param.h_interp, param.h_stencil_shape, param.h_stencil_size,
        param.F_interp, param.F_stencil_shape, param.F_stencil_size,
    )

    print("to_tensors()...")
    data = to_tensors(data, device)

    print("creating datasets...")
    train_dataset = SpatiotemporalDataset(
        data["train"]["t"], data["train"]["x"], data["train"]["u"],
        data["train"]["Phi_h"], data["train"]["b_h"],
        data["train"]["Phi_F"], data["train"]["b_F"],
        param.max_len,
    )
    val_dataset = SpatiotemporalDataset(
        data["val"]["t"], data["val"]["x"], data["val"]["u"],
        data["val"]["Phi_h"], data["val"]["b_h"],
        data["val"]["Phi_F"], data["val"]["b_F"],
        param.max_len,
    )
    test_dataset = SpatiotemporalDataset(
        data["test"]["t"], data["test"]["x"], data["test"]["u"],
        data["test"]["Phi_h"], data["test"]["b_h"],
        data["test"]["Phi_F"], data["test"]["b_F"],
        param.max_len,
    )
    print("finished create_datasets()")

    return train_dataset, val_dataset, test_dataset


def read_data(path: str) -> DataDict:
    """Reads data from folder `path` which contains subfolders train, val and test.
    Each subfolder contains ndarrays with time grids, coordinates and trajectories
    stored as t.pkl, x.pkl, and u.pkl files."""
    data = {}
    data["train"] = read_pickle(["t", "x", "u"], path+"train/")
    data["val"] = read_pickle(["t", "x", "u"], path+"val/")
    data["test"] = read_pickle(["t", "x", "u"], path+"test/")
    return data


def remove_data(data: dict[str, ndarray], frac: float) -> dict[str, ndarray]:
    n = data["t"].shape[0]
    n_keep = int(n * frac)
    data["t"] = data["t"][:n_keep]
    data["x"] = data["x"][:n_keep]
    data["u"] = data["u"][:n_keep]
    return data


def preprocess_data(data: DataDict) -> DataDict:
    data["train"], train_stats = _preprocess_data(data["train"])
    data["val"], _ = _preprocess_data(data["val"], train_stats)
    data["test"], _ = _preprocess_data(data["test"], train_stats)
    return data


def construct_interp_data(
    data: DataDict,
    boundary_cond: str,
    h_interp: str, h_stencil_shape: str, h_stencil_size: float,
    F_interp: str, F_stencil_shape: str, F_stencil_size: float,
) -> DataDict:

    data["train"]["Phi_h"], data["train"]["b_h"] = coords_to_interp_data(data["train"]["x"], h_interp, h_stencil_shape, h_stencil_size, boundary_cond)
    data["val"]["Phi_h"], data["val"]["b_h"] = coords_to_interp_data(data["val"]["x"], h_interp, h_stencil_shape, h_stencil_size, boundary_cond)
    data["test"]["Phi_h"], data["test"]["b_h"] = coords_to_interp_data(data["test"]["x"], h_interp, h_stencil_shape, h_stencil_size, boundary_cond)

    data["train"]["Phi_F"], data["train"]["b_F"] = coords_to_interp_data(data["train"]["x"], F_interp, F_stencil_shape, F_stencil_size, boundary_cond)
    data["val"]["Phi_F"], data["val"]["b_F"] = coords_to_interp_data(data["val"]["x"], F_interp, F_stencil_shape, F_stencil_size, boundary_cond)
    data["test"]["Phi_F"], data["test"]["b_F"] = coords_to_interp_data(data["test"]["x"], F_interp, F_stencil_shape, F_stencil_size, boundary_cond)

    return data


def coords_to_interp_data(
    x: ndarray,
    interp: str,
    stencil_shape: str,
    stencil_size: float,
    boundary_cond: str,
) -> tuple[ndarray, ndarray]:
    Phi, b = [], []
    for xi in tqdm(x):
        Phi_i, b_i = coord_to_interp_data(xi, interp, stencil_shape, stencil_size, boundary_cond)
        Phi.append(Phi_i)
        b.append(b_i)
    return np.array(Phi), np.array(b)  # shapes: (S, n_eval_pts, N) and (S, n_eval_pts, 1)


def add_noise(data: DataDict, sig: float, seed: int) -> DataDict:
    np.random.seed(seed)
    for i in range(len(data["train"]["u"])):
        data["train"]["u"][i] += np.random.randn(*data["train"]["u"][i].shape) * sig
    for i in range(len(data["val"]["u"])):
        data["val"]["u"][i] += np.random.randn(*data["val"]["u"][i].shape) * sig
    for i in range(len(data["test"]["u"])):
        data["test"]["u"][i] += np.random.randn(*data["test"]["u"][i].shape) * sig
    return data


def to_tensors(data: DataDict, device=None) -> TensorDataDict:
    tensor_data = {}
    tensor_data["train"] = {
        "t": torch.tensor(data["train"]["t"], dtype=torch.float64, device=device),
        "x": torch.tensor(data["train"]["x"], dtype=torch.float32, device=device),
        "u": torch.tensor(data["train"]["u"], dtype=torch.float32, device=device),
        "Phi_h": torch.stack([torch.sparse_coo_tensor(torch.tensor([Phi_i.row, Phi_i.col]), torch.tensor(Phi_i.data, dtype=torch.float32), size=Phi_i.shape, device=device) for Phi_i in data["train"]["Phi_h"]]),
        "Phi_F": torch.stack([torch.sparse_coo_tensor(torch.tensor([Phi_i.row, Phi_i.col]), torch.tensor(Phi_i.data, dtype=torch.float32), size=Phi_i.shape, device=device) for Phi_i in data["train"]["Phi_F"]]),
        "b_h": torch.tensor(data["train"]["b_h"], dtype=torch.float32, device=device),
        "b_F": torch.tensor(data["train"]["b_F"], dtype=torch.float32, device=device),
    }
    tensor_data["val"] = {
        "t": torch.tensor(data["val"]["t"], dtype=torch.float64, device=device),
        "x": torch.tensor(data["val"]["x"], dtype=torch.float32, device=device),
        "u": torch.tensor(data["val"]["u"], dtype=torch.float32, device=device),
        "Phi_h": torch.stack([torch.sparse_coo_tensor(torch.tensor([Phi_i.row, Phi_i.col]), torch.tensor(Phi_i.data, dtype=torch.float32), size=Phi_i.shape, device=device) for Phi_i in data["val"]["Phi_h"]]),
        "Phi_F": torch.stack([torch.sparse_coo_tensor(torch.tensor([Phi_i.row, Phi_i.col]), torch.tensor(Phi_i.data, dtype=torch.float32), size=Phi_i.shape, device=device) for Phi_i in data["val"]["Phi_F"]]),
        "b_h": torch.tensor(data["val"]["b_h"], dtype=torch.float32, device=device),
        "b_F": torch.tensor(data["val"]["b_F"], dtype=torch.float32, device=device),
    }
    tensor_data["test"] = {
        "t": torch.tensor(data["test"]["t"], dtype=torch.float64, device=device),
        "x": torch.tensor(data["test"]["x"], dtype=torch.float32, device=device),
        "u": torch.tensor(data["test"]["u"], dtype=torch.float32, device=device),
        "Phi_h": torch.stack([torch.sparse_coo_tensor(torch.tensor([Phi_i.row, Phi_i.col]), torch.tensor(Phi_i.data, dtype=torch.float32), size=Phi_i.shape, device=device) for Phi_i in data["test"]["Phi_h"]]),
        "Phi_F": torch.stack([torch.sparse_coo_tensor(torch.tensor([Phi_i.row, Phi_i.col]), torch.tensor(Phi_i.data, dtype=torch.float32), size=Phi_i.shape, device=device) for Phi_i in data["test"]["Phi_F"]]),
        "b_h": torch.tensor(data["test"]["b_h"], dtype=torch.float32, device=device),
        "b_F": torch.tensor(data["test"]["b_F"], dtype=torch.float32, device=device),
    }
    return tensor_data


def create_dataloaders(
    param: SimpleNamespace,
    train_dataset: SpatiotemporalDataset,
    val_dataset: SpatiotemporalDataset,
    test_dataset: SpatiotemporalDataset
) -> tuple[DataLoader, ...]:

    train_loader = DataLoader(
        train_dataset,
        batch_size=param.batch_size,
        shuffle=True,
        pin_memory=False,
    )
    val_loader = DataLoader(val_dataset, batch_size=param.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=param.batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def _preprocess_data(
    data: dict[str, ndarray],
    stats: dict | None = None
) -> tuple[dict[str, ndarray], dict | None]:

    is_train = stats is None
    if is_train:
        stats = {
            "T_min": data["t"].min(),
            "T_max": data["t"].max(),
            "x_min": data["x"].min(),
            "x_max": data["x"].max(),
            "u_min": data["u"].min(),
            "u_max": data["u"].max(),
        }

    # Normalize time grid.
    # data["t"] = (data["t"] - stats["T_min"]) / (stats["T_max"] - stats["T_min"])

    # Normalize spatial grid.
    # data["x"] = (data["x"] - stats["x_min"]) / (stats["x_max"] - stats["x_min"])

    # Normalize observations.
    # data["u"] = (data["u"] - stats["u_min"]) / (stats["u_max"] - stats["u_min"])

    if is_train:
        return data, stats
    else:
        return data, None


def read_pickle(keys: list[str], path: str = "./") -> dict[str, ndarray]:
    data_dict = {}
    for key in keys:
        with open(path+key+".pkl", "rb") as f:
            data_dict[key] = pickle.load(f)
    return data_dict


def get_model_components(param: SimpleNamespace) -> tuple[Encoder, ODETransitionFunction, NeuralDecoder]:
    nonlins = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
        "mish": nn.Mish,
    }
    solver_kwargs = {
        "method": param.solver,
        "rtol": param.rtol,
        "atol": param.atol,
        "adjoint": param.adjoint,
    }

    h = Encoder(
        hI=Interpolator(),
        hS=SpatialAggregator(
            nn.Sequential(
                nn.Linear(param.D * (1 + get_stencil_size(param.h_stencil_shape)), param.D_agg)
            )
        ),
        hT=TemporalAggregator(
            create_temporal_aggregation_function(param),
        ),
        hR=ReadFunction(
            f_gamma=nn.Sequential(nn.Linear(param.D_agg, param.d)),
            f_tau=nn.Sequential(nn.Linear(param.D_agg, param.d)),
            tau_min=param.tau_min,
        ),
    )

    F_hid_layers = []
    for _ in range(param.F_n_hid_layers - 1):
        F_hid_layers.append(nn.Linear(param.F_hid_size, param.F_hid_size))
        F_hid_layers.append(nonlins[param.F_nonlin]())

    F = ODETransitionFunction(
        f=DynamicsFunction(
            hI=Interpolator(),
            hS=SpatialAggregator(
                nn.Sequential(
                    nn.Linear(param.d * (1 + get_stencil_size(param.F_stencil_shape)), param.F_hid_size), nonlins[param.F_nonlin](),
                    *F_hid_layers,
                    nn.Linear(param.F_hid_size, param.d)
                )
            )
        ),
        layers_to_count=[],
        solver_kwargs=solver_kwargs
    )

    g = NeuralDecoder(
        nn.Sequential(
            # nn.Linear(param.d, param.D),
            ExtractFirstComponent() if param.D == 1 else MyIdentity(),
            ToNormalParameters(param.sig_u),
        ),
        layers_to_count=[ExtractFirstComponent, MyIdentity],
    )

    print_parameter_info(h, F, g)

    return h, F, g


class MyIdentity(Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.parameter.Parameter(torch.zeros([1]))
        self.bias = torch.nn.parameter.Parameter(torch.zeros([1]))

    def forward(self, x):
        return x


class ExtractFirstComponent(Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.parameter.Parameter(torch.zeros([1]))
        self.bias = torch.nn.parameter.Parameter(torch.zeros([1]))

    def forward(self, x):
        return x[:, :, :, [0]]


def create_elbo(
    h: Encoder,
    F: ODETransitionFunction,
    g: NeuralDecoder,
    param: SimpleNamespace
) -> AmortizedMultipleShootingELBO:

    prior_param_dict = nn.ParameterDict({
        "mu0": Parameter(0.0 * torch.ones([param.d]), False),
        "sig0": Parameter(1.0 * torch.ones([param.d]), False),
        "sig_c": Parameter(param.sig_c * torch.ones([1]), False),
        "mu_theta": Parameter(0.0 * torch.ones([1]), False),
        "sig_theta": Parameter(1.0 * torch.ones([1]), False),
    })
    posterior_param_dict = nn.ParameterDict({
        "mu_theta_g": Parameter(torch.cat([par.detach().reshape(-1) for par in g.parameters()])),
        "log_sig_theta_g": Parameter(-7.0 * torch.ones(g.param_count())),
        "mu_theta_F": Parameter(torch.cat([par.detach().reshape(-1) for par in F.parameters()])),
        "log_sig_theta_F": Parameter(-7.0 * torch.ones(F.param_count())),
    })
    p = ModelNormal(prior_param_dict, g, F)
    q = AmortizedMultipleShootingPosterior(posterior_param_dict, h, F)
    elbo = AmortizedMultipleShootingELBO(p, q)
    elbo.p.set_theta(elbo.q.sample_theta())
    return elbo


def visualize_trajectories(
    coords: list[ndarray],
    traj: list[ndarray],
    vis_inds: list[int],
    title: str,
    path: str,
    img_name: str,
) -> None:

    if not os.path.isdir(path):
        os.makedirs(path)

    panel_size = 5
    n_row = len(traj)
    n_col = len(vis_inds)

    fig, ax = plt.subplots(n_row, n_col, figsize=(panel_size*n_col, panel_size*n_row), squeeze=False)

    for i in range(n_row):
        for j in range(n_col):

            ax[i, j].grid(False)  # type: ignore
            ax[i, j].get_xaxis().set_visible(False)  # type: ignore
            ax[i, j].get_yaxis().set_visible(False)  # type: ignore

            im = ax[i, j].tricontourf(coords[i][:, 0], coords[i][:, 1], traj[i][vis_inds[j], :, 0], cmap="plasma")  # type: ignore
            fig.colorbar(im, ax=ax[i, j], orientation='vertical')  # type: ignore

    fig.suptitle(title, fontsize=45)
    fig.tight_layout()
    plt.savefig(path+img_name)
    plt.close()


class ToNormalParameters(Module):
    """Converts output of MLPDecoder to parameters of p(u|z)."""
    def __init__(self, sig_u) -> None:
        super().__init__()
        self.sig_u = sig_u

    def forward(self, x):
        x_a = x.unsqueeze(-1)
        x_b = torch.ones_like(x_a) * self.sig_u  # fix standard deviation
        x = torch.cat((x_a, x_b), dim=-1)
        return x


# def get_scheduler(optimizer, n_iters, lr):
#     # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=(1e-5/lr)**(1.0/n_iters))

#     warmup_iters = int(0.01 * n_iters)
#     lin_sched = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-6, end_factor=1, total_iters=warmup_iters)
#     const_sched = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=n_iters-warmup_iters)
#     scheduler = torch.optim.lr_scheduler.ChainedScheduler([lin_sched, const_sched])

#     return scheduler


def get_scheduler(optimizer, n_iters, lr):
    warmup_iters = int(0.01 * n_iters)

    lin_sched = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-6, end_factor=1, total_iters=warmup_iters)

    const_sched = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=n_iters-warmup_iters)
    # exp_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=(1e-4/lr)**(1.0/(n_iters-warmup_iters)))
    # cos_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iters-warmup_iters)

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[lin_sched, const_sched],
        milestones=[warmup_iters],
    )

    return scheduler


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


class BatchMovingAverage():
    """Computes moving average over the last `k` mini-batches
    and stores the smallest recorded moving average in `min_avg`."""
    def __init__(self, k: int) -> None:
        self.values = deque([], maxlen=k)
        self.min_avg = np.inf

    def add_value(self, value: float) -> None:
        self.values.append(value)

    def get_average(self) -> float:
        if len(self.values) == 0:
            avg = np.nan
        else:
            avg = sum(self.values) / len(self.values)

        if avg < self.min_avg:
            self.min_avg = avg

        return avg

    def get_min_average(self):
        return self.min_avg


def get_inference_data(t: Tensor, y: Tensor, delta_inf: float) -> tuple[list[Tensor], list[Tensor]]:
    t_inf, y_inf = [], []
    for i in range(t.shape[0]):
        inf_inds = torch.argwhere(t[[i]] <= delta_inf)[:, 1]
        t_inf.append(t[[i]][:, inf_inds, :].clone())
        y_inf.append(y[[i]][:, inf_inds, :, :].clone())
    return t_inf, y_inf


def get_z0(elbo, t: list[Tensor], u: list[Tensor], Phi_h: Tensor, b_h: Tensor) -> Tensor:
    z0 = []
    for i, (ti, ui) in enumerate(zip(t, u)):
        gamma, tau = elbo.q.h(ti, ti, ui, torch.stack([Phi_h[i]], dim=0), b_h[[i]])
        z0.append(gamma[:, [0], :, :] + tau[:, [0], :, :] * torch.randn_like(tau[:, [0], :, :]))
    return torch.cat(z0)


def get_zm(elbo, t: list[Tensor], u: list[Tensor], Phi_h: Tensor, b_h: Tensor) -> Tensor:
    zm = []
    for i, (ti, ui) in enumerate(zip(t, u)):
        gamma, tau = elbo.q.h(ti, ti, ui, torch.stack([Phi_h[i]], dim=0), b_h[[i]])
        zm.append(gamma[:, [-1], :, :] + tau[:, [-1], :, :] * torch.randn_like(tau[:, [-1], :, :]))
    return torch.cat(zm)


def _pred_full_traj(elbo, t: Tensor, z0: Tensor, Phi_F: Tensor, b_F: Tensor) -> Tensor:
    elbo.p.set_theta(elbo.q.sample_theta())
    S, M, N, d = z0.shape[0], t.shape[1], Phi_F.shape[2], z0.shape[3]

    z = torch.zeros((S, M, N, d), dtype=z0.dtype, device=z0.device)
    z[:, [0], :, :] = z0

    for i in range(1, M):
        z[:, [i], :, :] = elbo.p.F(
            z[:, [i-1], :, :],
            extract_time_grids(t[:, i-1:i+1, :], n_blocks=1),
            Phi_F,
            b_F,
        )

    return elbo.p._sample_lik(z)


def pred_full_traj(
    param, elbo, t: Tensor, u: Tensor,
    Phi_h: Tensor, b_h: Tensor, Phi_F: Tensor, b_F: Tensor,
) -> Tensor:
    t_inf, u_inf = get_inference_data(t, u, param.delta_inf)
    z0 = get_z0(elbo, t_inf, u_inf, Phi_h, b_h)
    u_full_traj = _pred_full_traj(elbo, t, z0, Phi_F, b_F)
    return u_full_traj


def save_model(model, path, name):
    if not os.path.isdir(path):
        os.makedirs(path)
    torch.save(model.state_dict(), path+name+".pt")


def load_model(model, path, name, device):
    model.load_state_dict(torch.load(path+name+".pt", map_location=device), strict=False)


def create_node_mask(p, n):
    mask = torch.rand(n) < p
    return mask


def get_parameters_count(model):
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n_params


def print_parameter_info(h, F, g):
    p_enc = get_parameters_count(h)
    p_dyn = get_parameters_count(F)
    p_dec = get_parameters_count(g)
    print(
        f"Parameter info:\n"
        f"Encoder: {p_enc}\n"
        f"Dynamics: {p_dyn}\n"
        f"Decoder: {p_dec}\n"
        f"Total: {p_enc+p_dyn+p_dec}\n"
    )
