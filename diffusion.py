import torch
import numpy as np


# forward diffusion
def make_beta_schedule(schedule='linear', n_timesteps=1000, start=1e-5, end=1e-2):
    if schedule == 'linear':
        betas = torch.linspace(start, end, n_timesteps)
    elif schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, n_timesteps)
        betas = torch.sigmoid(betas) * (end - start) + start
    return betas

def extract(input, s, x):
    shape = x.shape
    out = torch.gather(input, 0, s.to(input.device))
    reshape = [s.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)


def q_x(x_0, s, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, noise=None):
    if noise is None:
        noise = torch.randn_like(x_0)
    alphas_t = extract(alphas_bar_sqrt, s, x_0)
    alphas_1_m_t = extract(one_minus_alphas_bar_sqrt, s, x_0)

    sample = alphas_t * x_0 + alphas_1_m_t * noise

    return sample

def q_posterior_mean_variance(x_0, x_t, t, posterior_mean_coef_1, posterior_mean_coef_2, posterior_log_variance_clipped):
    coef_1 = extract(posterior_mean_coef_1, t, x_0)
    coef_2 = extract(posterior_mean_coef_2, t, x_0)
    mean = coef_1 * x_0 + coef_2 * x_t
    var = extract(posterior_log_variance_clipped, t, x_0)
    return mean, var

def noise_estimation_loss(model, batch_x, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps, loss_function='l2', device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    x_0 = batch_x['trajectories']
    t = batch_x['t']
    sp = batch_x['start']
    ep = batch_x['end']
    z = batch_x['z']


    batch_size = x_0.size()[0]

    # Select a random step for each example
    s = torch.randint(0, n_steps, size=(batch_size // 2 + 1,))
    s = torch.cat([s, n_steps - s - 1], dim=0)[:batch_size].long()

    e = torch.randn_like(x_0)

    x = q_x(x_0, s, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, noise=e)

    # fix start and end point to 0

    endpoint_idx = np.equal(t[:,-1], batch_x['traj_len'].view(t[:,-1].shape)).bool()
    batch_x['trajectories'][endpoint_idx,-1] = ep[endpoint_idx]

    startpoint_idx = np.equal(t[:,0], np.zeros(t[:,0].shape)).bool()
    batch_x['trajectories'][startpoint_idx,0] = sp[startpoint_idx]

    t = t / 100

    # context
    x = x.to(device)
    s = s.to(device)
    t = t.to(device)
    sp = sp.to(device)
    ep = ep.to(device)
    z = z.to(device)

    output = model(x, s, t, sp, ep, z)
    e = e.to(device)

    del x
    del s
    del t
    del sp
    del ep
    del z

    center = e.shape[1] // 2

    if loss_function == 'l1':
        # L1 loss
        return abs(e - output).sum()

    if loss_function == 'l2':
        # L2 loss
        return (e - output).square().sum()

    if loss_function == 'l4':
        # L4 loss
        return (e - output).square().square().sum()

    if loss_function == 'log_loss':
        # Log Loss
        # maybe still buggy?
        return (-1 / torch.log(abs(e - output) / 3)).sum()

    if loss_function == 'LDA':
        # https://www.nature.com/articles/s41598-021-99609-x
        length_diff = torch.sqrt(torch.square(abs(e) - abs(output))) / abs(e) + abs(output)
        angle_diff = 0.5 * (1 - e * output / abs(e) * abs(output))

        return (0.5 * length_diff + 0.5 * angle_diff).sum()

def p_sample(model, x, s, t, sp, ep, z, alphas,betas,one_minus_alphas_bar_sqrt, start_point, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):

    s = torch.tensor([s])

    # Factor to the model output
    eps_factor = ((1 - extract(alphas, s, x)) / extract(one_minus_alphas_bar_sqrt, s, x))

    # Create context
    # Model output
    x = x.to(device)
    s = s.to(device)
    t = t.to(device)
    sp = sp.to(device)
    ep = ep.to(device)
    z = z.to(device)

    eps_theta = model(x, s, t, sp, ep, z)

    center = x.shape[0] // 2
    x = x[center]
    # Final values
    s = s.cpu()
    x = x.cpu()
    eps_theta = eps_theta.cpu()
    mean = (1 / extract(alphas, s, x).sqrt()) * (x - (eps_factor * eps_theta))
    # Generate z
    z = torch.randn_like(x)

    if s == 0:
        z = 0

    # Fixed sigma
    sigma_t = extract(betas, s, x).sqrt()
    sample = mean + sigma_t * z


    del x
    del s
    del t
    del sp
    del ep
    del z

    return (sample)
