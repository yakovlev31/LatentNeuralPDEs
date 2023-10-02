from types import SimpleNamespace

import torch

from einops import reduce

import wandb
from tqdm import tqdm

import lnpde.utils.shallow_water as data_utils


torch.backends.cudnn.benchmark = True  # type: ignore


# Read parameters.
argparser = data_utils.create_argparser()
param = SimpleNamespace(**vars(argparser.parse_args()))
param.tags.append("test")

# param.tau_min = 0
param.sig_u = 1e-3
param.sig_c = 1e-3
param.latent_dim = param.d


# Load data.
train_dataset, val_dataset, test_dataset = data_utils.create_datasets(param)
train_loader, val_loader, test_loader = data_utils.create_dataloaders(param, train_dataset, val_dataset, test_dataset)


# Create model.
data_utils.set_seed(param.seed)
device = torch.device(param.device)
h, F, g = data_utils.get_model_components(param)
elbo = data_utils.create_elbo(h, F, g, param).to(device)
data_utils.load_model(elbo, param.model_folder, param.name, device)
elbo.eval()


wandb.init(
    mode="disabled",  # online/disabled
    project="L-NPDE",
    group=param.group,
    tags=param.tags,
    name=param.name,
    config=vars(param),
    save_code=True,
)

loss_fn = torch.nn.L1Loss(reduction="none")

data_utils.set_seed(param.seed)
c = 0
with torch.no_grad():
    losses = []
    for batch in tqdm(test_loader, total=len(test_loader)):
        t, x, u, Phi_h, b_h, Phi_F, b_F, traj_ind = [bi.to(device) for bi in batch]

        t_inf, u_inf = data_utils.get_inference_data(t, u, param.delta_inf)
        for i in range(len(u_inf)):
            u_inf[i] += torch.randn_like(u_inf[i]) * param.sig_add

        # Start at z0
        m = 5
        u_pd = torch.zeros_like(u)
        for i in range(param.n_mc_samples):
            z0 = data_utils.get_z0(elbo, t_inf, u_inf, Phi_h, b_h)
            u_pd += data_utils._pred_full_traj(elbo, t, z0, Phi_F, b_F)
        u_pd /= param.n_mc_samples
        loss_per_traj = reduce(loss_fn(u_pd[:, m:], u[:, m:]), "s m n d -> s () () ()", "mean").view(-1).detach().cpu().numpy().ravel()
        losses.extend(loss_per_traj)

        # Start at zm
        # assert param.batch_size == 1, ""
        # m = t_inf[0].shape[1]
        # u_pd = torch.zeros_like(u)[:, m-1:]
        # _t = t[:, m-1:].clone()
        # _t = _t - _t.min()
        # for i in range(param.n_mc_samples):
        #     zm = data_utils.get_zm(elbo, t_inf, u_inf, Phi_h, b_h)
        #     u_pd += data_utils._pred_full_traj(elbo, _t, zm, Phi_F, b_F)
        # u_pd /= param.n_mc_samples
        # loss_per_traj = reduce(loss_fn(u_pd[:, 1:], u[:, m:]), "s m n d -> s () () ()", "mean").view(-1).detach().cpu().numpy().ravel()
        # losses.extend(loss_per_traj)

        data_utils.visualize_trajectories(
            coords=[
                x[0].detach().cpu().numpy(),
                x[0].detach().cpu().numpy(),
            ],
            traj=[
                u[0, m:].detach().cpu().numpy(),
                u_pd[0, m:].detach().cpu().numpy(),
            ],
            vis_inds=list(range(0, u_pd[:, m:].shape[1]))[::],
            title=f"Trajectory {c}",
            path=f"./img/test_vis/{param.name}/",
            img_name=f"traj_{c}.png",
        )
        c += 1

mean_loss = sum(losses) / len(losses)
print(mean_loss)

wandb.run.summary.update({"mean_test_loss": mean_loss})  # type: ignore
