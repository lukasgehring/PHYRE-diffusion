import numpy as np
import h5py
import os
import torch

from torch.utils.data import Dataset


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        env = sample[0]

        return torch.from_numpy(env)


class EnvironmentDataset(Dataset):
    """PHYRE dataset."""

    def __init__(self, transform=None, sliding_window=1, templates=[], redball=False):

        self.sliding_window = sliding_window
        self.templates = templates
        self.redball = redball

        self.data = self.load_data()
        self.transform = transform

    def load_data(self):
        dataset = []

        # template
        for template_num, template in enumerate(os.listdir('./PHYRE-dataset')):

            if len(self.templates) != 0:
                if template_num not in self.templates:
                    continue

            # task
            for task_num, task in enumerate(os.listdir(f'./PHYRE-dataset/{template}')):

                # fold
                for fold_num, fold in enumerate(os.listdir(f'./PHYRE-dataset/{template}/{task}')):

                    # file
                    with h5py.File(f'./PHYRE-dataset/{template}/{task}/{fold}/data.h5', "r") as f:

                        env_image = np.array(f['env'])

                        # generate env
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

                        # normal orientation
                        dataset.append(env)

                        # vertical flip
                        dataset.append(np.flip(env, 1).copy())

                        # horizonal flip
                        dataset.append(np.flip(env, 2).copy())

                        # horizonal, vertical flip
                        dataset.append(np.flip(env, (1, 2)).copy())

                break

        print(f"{len(dataset)} samples loaded!")

        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = np.array([self.data[idx]])

        if self.transform:
            sample = self.transform(sample)

        return sample
