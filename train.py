import torch
import numpy as np

from torch.utils.data import DataLoader
from loader import PHYREDataset, ToTensor
from diffusion import make_beta_schedule, noise_estimation_loss
from torchvision import transforms

from model import ConditionalModel, EMA


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------
# PARAMETERS
#-----------------------------------------------

# VAE name
VAE_name = "conv_vae64_1_epochs"

# diffusion steps
num_steps = 51

# environment parameters
sliding_window = 9
batch_size = 16
training_templates = [1, 11, 12, 13, 14, 15, 8]
training_templates = [2]
redball = False

# model
use_start_end_points = True
use_sin_cos_t = True

# optimizer
learning_rate = 4e-05

# loss function
loss_function = 'l2'

# training
epochs = 1

# ----------------------------------------------
# PARAMETERS END
#-----------------------------------------------

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

# load environment model
env_encoder = torch.load(f'ConVAE/trained-models/{VAE_name}.pt')
env_encoder.eval()

# load environment
training_data = PHYREDataset(env_encoder=env_encoder, transform=transforms.Compose([ToTensor()]), sliding_window=sliding_window,
                             templates=training_templates, redball=redball, device=device)
training_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=0)

# Create Diffusion Model
model = ConditionalModel(sliding_window=sliding_window, use_start_end_points=use_start_end_points, use_sin_cos_t=use_sin_cos_t).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Create EMA model
ema = EMA(0.9)
ema.register(model)

for epoch in range(epochs):

    num_batches = len(training_dataloader)

    losses = np.zeros(num_batches)
    for i, batch_x in enumerate(training_dataloader):
        # Compute the loss.
        loss = noise_estimation_loss(model, batch_x, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, num_steps,
                                     loss_function=loss_function, device=device)
        losses[i] = loss

        # Before the backward pass, zero all of the network gradients
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to parameters
        loss.backward()

        # Perform gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)

        # Calling the step function to update the parameters
        optimizer.step()

        # Update the exponential moving average
        ema.update(model)

    # if t % 10 == 0:
    print("epoch: ", epoch, np.mean(losses))

# save model
torch.save(model, f'./models/model_64_train_{epochs}_{loss_function}_{sliding_window}.pt')
