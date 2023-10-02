from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.optim as optim

import wandb
from tqdm import tqdm

import lnpde.utils.scalar_flow as data_utils


torch.backends.cudnn.benchmark = True  # type: ignore


# Read parameters.
argparser = data_utils.create_argparser()
param = SimpleNamespace(**vars(argparser.parse_args()))
param.tags.append("train")


# Load data.
train_dataset, val_dataset, _ = data_utils.create_datasets(param)
train_loader, val_loader, _ = data_utils.create_dataloaders(param, train_dataset, val_dataset, val_dataset)


# Create model.
data_utils.set_seed(param.seed)
device = torch.device(param.device)
h, F, g = data_utils.get_model_components(param)
elbo = data_utils.create_elbo(h, F, g, param).to(device)
print(elbo)


# Training.
optimizer = optim.Adam(elbo.parameters(), lr=param.lr)
scheduler = data_utils.get_scheduler(optimizer, param.n_iters, param.lr)

bma = data_utils.BatchMovingAverage(k=10)

wandb.init(
    mode="disabled",  # online/disabled
    project="L-NPDE",
    group=param.group,
    tags=param.tags,
    name=param.name,
    config=vars(param),
    save_code=True,
)

data_utils.set_seed(param.seed)
for i in tqdm(range(param.n_iters), total=param.n_iters):
    elbo.train()
    t, x, u, Phi_h, b_h, Phi_F, b_F, traj_ind = [bi.to(device) for bi in next(iter(train_loader))]

    L1, L2, L3, z, s = elbo(t, u, param.block_size, param.scaler, Phi_h, b_h, Phi_F, b_F)
    L1 *= len(train_dataset) / u.shape[0]
    L2 *= len(train_dataset) / u.shape[0]
    loss = -(L1 - L2 - L3)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    # Validation on full trajectory predictions.
    if i % int(0.01 * param.n_iters) == 0 or i == param.n_iters - 1:
        with torch.no_grad():
            elbo.eval()
            t_val, x_val, u_val, Phi_h_val, b_h_val, Phi_F_val, b_F_val, traj_ind_val = [bi.to(device) for bi in next(iter(val_loader))]

            pred_len = 10
            ind_a = torch.randint(0, u_val.shape[1] - pred_len - 1, (1, ))[0]
            ind_b = ind_a + pred_len
            t, u = t[:, ind_a:ind_b, :], u[:, ind_a:ind_b, :, :]
            t_val, u_val = t_val[:, ind_a:ind_b, :], u_val[:, ind_a:ind_b, :, :]
            t, t_val = t - t.min(), t_val - t_val.min()
            z = z[:, ind_a:ind_b]

            L1_val, L2_val, L3_val, z_val, s_val = elbo(t_val, u_val, 1, param.scaler, Phi_h_val, b_h_val, Phi_F_val, b_F_val)
            L1_val *= len(val_dataset) / u_val.shape[0]
            L2_val *= len(val_dataset) / u_val.shape[0]
            loss_val = -(L1_val - L2_val - L3_val)

            u_full_traj = data_utils.pred_full_traj(param, elbo, t, u, Phi_h, b_h, Phi_F, b_F)
            u_val_full_traj = data_utils.pred_full_traj(param, elbo, t_val, u_val, Phi_h_val, b_h_val, Phi_F_val, b_F_val)

            train_full_traj_mae = nn.L1Loss()(u_full_traj, u).item()
            val_full_traj_mae = nn.L1Loss()(u_val_full_traj, u_val).item()

            bma.add_value(val_full_traj_mae)
            if bma.get_average() <= bma.get_min_average():
                data_utils.save_model(elbo, param.model_folder, param.name)

            wandb.log(
                {
                    "-L1": -L1.item(),
                    "L2": L2.item(),
                    "L3": L3.item(),
                    "-ELBO": loss.item(),

                    "-L1_val": -L1_val.item(),
                    "L2_val": L2_val.item(),
                    "L3_val": L3_val.item(),
                    "-ELBO_val": loss_val.item(),

                    "train_full_traj_mae": train_full_traj_mae,
                    "val_full_traj_mae": val_full_traj_mae,

                    "lr": optimizer.param_groups[0]["lr"],
                },
                step=i
            )
            if param.visualize == 1:

                s_plot = torch.zeros_like(z)
                # s_plot[:, :] = s[:, ind_a:ind_b]

                data_utils.visualize_trajectories(
                    coords=[
                        x[0].detach().cpu().numpy(),
                        x[0].detach().cpu().numpy(),
                        # x[0].detach().cpu().numpy(),
                    ],
                    traj=[
                        u[0].detach().cpu().numpy(),
                        u_full_traj[0].detach().cpu().numpy(),
                        # s_plot[0, :, :, [0]].detach().cpu().numpy(),
                    ],
                    vis_inds=list(range(u.shape[1]-1))[:-1:max(1, int(0.09*u.shape[1]))],
                    title=f"Iteration {i}",
                    path=f"./img/{param.name}/",
                    img_name=f"iter_{i}.png",
                )
