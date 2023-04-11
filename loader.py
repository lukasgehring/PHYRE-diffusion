import numpy as np
import h5py
import os
import torch
from torchvision import transforms

from torch.utils.data import Dataset


def lantent_representation(model, sample, device):
    with torch.no_grad():
        sample = sample.to(device)
        _, z, _ = model(sample)

        return z.view(sample.shape[0], -1)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        traj, t, sp, ep, env, z, traj_len = sample['trajectories'], sample['t'], sample['start'], sample['end'], sample[
            'env'], sample['z'], sample['traj_len']

        return {'trajectories': torch.from_numpy(traj).float(), 't': torch.Tensor(t).long(), 'start': torch.Tensor(sp),
                'end': torch.Tensor(ep), 'env': env, 'z': z, 'traj_len': torch.Tensor([traj_len]).long()}


class PHYREDataset(Dataset):
    """Load PHYRE dataset"""

    def __init__(self, env_encoder, transform=None, sliding_window=1, templates=[], redball=False,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):

        # Environment encoder
        self.env_encoder = env_encoder

        # Parameter
        self.sliding_window = sliding_window
        self.templates = templates
        self.redball = redball
        self.device = device

        self.data = self.load_data()
        self.transform = transform

    def load_data(self):
        dataset = []

        # Iterate all templates
        for template_num, template in enumerate(os.listdir('./PHYRE-dataset')):

            if len(self.templates) != 0:
                if int(template[4:]) not in self.templates:
                    continue
                else:
                    print("Template", int(template[4:]), "loaded")

            # Iterate all tasks
            for task_num, task in enumerate(os.listdir(f'./PHYRE-dataset/{template}')):

                # Iterate all folds
                for fold_num, fold in enumerate(os.listdir(f'./PHYRE-dataset/{template}/{task}')):

                    # Iterate all files
                    with h5py.File(f'./PHYRE-dataset/{template}/{task}/{fold}/data.h5', "r") as f:

                        # extract positions
                        red_ball_id = np.argwhere(np.array(f['features'])[0, :, 8] == 1)[0, 0]
                        green_ball_id = np.argwhere(np.array(f['features'])[0, :, 9] == 1)[0, 0]
                        target_id = np.argwhere(np.array(f['features'])[0, :, 11] == 1)[0, 0]
                        obstacle_id = np.argwhere(np.array(f['features'])[0, :, 13] == 1)[0, 0]

                        # load trajectory and resize to 256*256 image
                        trajectory = np.array(f['features'])[:, green_ball_id, :2] * 256

                        # generate environment
                        env_image = np.array(f['env'])
                        img_channels = 4 if self.redball else 3
                        env = np.zeros((img_channels, 256, 256), np.float32)

                        # green ball
                        env[0, env_image == 2] = 1
                        # target
                        env[1, env_image == 4] = 1
                        # obstacle
                        env[2, env_image == 6] = 1
                        # red ball
                        if self.redball:
                            env[3, env_image == 1] = 1

                        # resize env
                        env_data = transforms.Resize((64, 64))(torch.from_numpy(env)).view(-1, img_channels, 64, 64)

                        # latent vector from Conv-VAE
                        z = lantent_representation(self.env_encoder, env_data, self.device).view(64)

                        # resize trajectory
                        trajectory -= 128
                        trajectory /= 50

                        # calculate trajectory length
                        traj_len = len(trajectory) - self.sliding_window + 1

                        # add one sample for each trajectory point with the sliding window
                        for t, sw in enumerate(
                                np.lib.stride_tricks.sliding_window_view(trajectory, (self.sliding_window, 2)).reshape(
                                    -1, self.sliding_window, 2)):
                            if t == traj_len:
                                break

                            dataset.append(
                                [sw, np.arange(t, self.sliding_window + t), trajectory[0], trajectory[-1], env, z,
                                 traj_len])

        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'trajectories': self.data[idx][0], 't': self.data[idx][1], 'start': self.data[idx][2],
                  'end': self.data[idx][3], 'env': self.data[idx][4], 'z': self.data[idx][5],
                  'traj_len': self.data[idx][6]}

        if self.transform:
            sample = self.transform(sample)

        return sample
