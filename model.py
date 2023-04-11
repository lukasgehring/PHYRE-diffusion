import torch.nn.functional as F
from torch import nn
import torch

class EMA(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        module_copy = type(module)(module.config).to(module.config.device)
        module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv1(x))


class Encoder(nn.Module):
    def __init__(self, chs=(3, 8, 16, 32, 64, 64)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([EncoderBlock(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(4096, 4096)

    def forward(self, x):
        for block in self.enc_blocks:
            x = block(x)
            x = self.pool(x)
        x = x.view(-1, 4096)

        return self.linear(x)


class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)

    def forward(self, x, s, t, z, sp=None, ep=None):

        if sp is None:
            x = torch.cat((x, s, t, z), axis=1)
        else:
            x = torch.cat((x, s, t, sp, ep, z), axis=1)

        x = self.lin(x)

        return x


class ConditionalModel(nn.Module):
    def __init__(self, sliding_window, use_start_end_points=True, use_sin_cos_t=False):
        super(ConditionalModel, self).__init__()

        # parameters
        self.use_start_end_points = use_start_end_points
        self.use_sin_cos_t = use_sin_cos_t
        self.sliding_window = sliding_window * 2

        # calculate number of parameters: t + s + z
        conditional_parameter_size = sliding_window + 1 + 64

        if use_sin_cos_t:
            # t is inserted 2x (sin and cos)
            conditional_parameter_size += sliding_window

        if use_start_end_points:
            # add 4 points of start and end points are used
            conditional_parameter_size += 4

        self.lin1 = ConditionalLinear(self.sliding_window + conditional_parameter_size, 1024)
        self.lin2 = ConditionalLinear(1024 + conditional_parameter_size, 1024)
        self.lin3 = ConditionalLinear(1024 + conditional_parameter_size, 512)
        self.lin4 = ConditionalLinear(512 + conditional_parameter_size, 128)

        self.out = nn.Linear(128, self.sliding_window)

    def forward(self, x, s, t, sp, ep, z):

        # x: trajectory points of the sliding window
        # s: diffusion step
        # t: trajectory step
        # sp: start point of the trajectory
        # ep: start point of the trajectory
        # z: latent space representation of the environment

        x = torch.reshape(x, (-1, self.sliding_window))
        s = s.reshape(-1, 1)

        if self.use_sin_cos_t:
            t_sin = torch.sin(t)
            t_cos = torch.cos(t)

            t = torch.cat((t_sin, t_cos), axis=1)

        if self.use_start_end_points:

            x = F.softplus(self.lin1(x, s, t, z, sp, ep))
            x = F.softplus(self.lin2(x, s, t, z, sp, ep))
            x = F.softplus(self.lin3(x, s, t, z, sp, ep))
            x = F.softplus(self.lin4(x, s, t, z, sp, ep))

        else:
            x = F.softplus(self.lin1(x, s, t, z))
            x = F.softplus(self.lin2(x, s, t, z))
            x = F.softplus(self.lin3(x, s, t, z))
            x = F.softplus(self.lin4(x, s, t, z))

        return self.out(x).reshape(-1, self.sliding_window // 2, 2)
