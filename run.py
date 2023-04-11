import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from diffusion import p_sample, make_beta_schedule
from helper import visualize_env
from loader import PHYREDataset, ToTensor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = torch.load('./models/model_64_train_1_l2_9.pt')
model.eval()

# load environment model
env_encoder = torch.load('ConVAE/trained-models/conv_vae64_1_epochs.pt')
env_encoder.eval()

# ------------------------------------------------
# PARAMETERS
# ------------------------------------------------
num_steps = 51

templates = [2, 7]
redball = False

sliding_window = 9

visualize = True
# ------------------------------------------------
# PARAMETERS END
# ------------------------------------------------

# pre-load diffusion
betas = make_beta_schedule(schedule='sigmoid', n_timesteps=num_steps, start=1e-8, end=1e-1)

alphas = 1 - betas
alphas_prod = torch.cumprod(alphas, 0)
alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]], 0)
alphas_bar_sqrt = torch.sqrt(alphas_prod)
one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

posterior_mean_coef_1 = (betas * torch.sqrt(alphas_prod_p) / (1 - alphas_prod))
posterior_mean_coef_2 = ((1 - alphas_prod_p) * torch.sqrt(alphas) / (1 - alphas_prod))
posterior_variance = betas * (1 - alphas_prod_p) / (1 - alphas_prod)
posterior_log_variance_clipped = torch.log(
    torch.cat((posterior_variance[1].view(1, 1), posterior_variance[1:].view(-1, 1)), 0)).view(-1)

dataset = PHYREDataset(env_encoder=env_encoder, transform=transforms.Compose([ToTensor()]), sliding_window=1,
                       templates=templates, redball=redball, device=device)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

for sample in dataloader:

    # new task
    if sample['t'] == 0:
        true_traj = np.zeros((sample['traj_len'], 2))

    # add sample to trajectory
    true_traj[sample['t']] = sample['trajectories'][0, 0]

    # last sample from task
    if sample['t'] == sample['traj_len'] - 1:

        itr = sample['traj_len']

        start = sample['start'].view(-1, 2)
        end = sample['end'].view(-1, 2)
        z = sample['z'].view(-1, 64)
        env = sample['env'][0]  # .view(-1, 3, 256, 256)

        x_s = torch.randn([itr, 2])
        x_s[0] = start[0]
        x_s[-1] = end[0]

        for s in reversed(range(num_steps)):

            new_x_s = x_s.clone().detach()

            for i in range(((sliding_window - 1) // 2), itr - ((sliding_window - 1) // 2)):
                t = torch.Tensor(
                    np.arange(i - ((sliding_window - 1) // 2), i + ((sliding_window - 1) // 2) + 1)).long().view(-1,
                                                                                                                 sliding_window) / 100

                with torch.no_grad():
                    pred = \
                        p_sample(model, x_s[i - ((sliding_window - 1) // 2):i + ((sliding_window - 1) // 2) + 1], s, t,
                                 start, end, z, alphas, betas, one_minus_alphas_bar_sqrt, start)[
                            0, (sliding_window - 1) // 2]
                    new_x_s[i] = pred

            x_s = new_x_s
            x_s[0] = start[0]
            x_s[-1] = end[0]

        mean_l2 = np.sqrt(np.square(true_traj - np.array(x_s)).sum(axis=1)).mean()
        print('loss', mean_l2)

        if visualize:
            visualize_env(env, redball=False, traj=x_s * 50 + 128)

            visualize_env(env, redball=False, traj=true_traj * 50 + 128)

            break
