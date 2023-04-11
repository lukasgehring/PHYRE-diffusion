import numpy as np
import matplotlib.pyplot as plt
import torch


def visualize_env(env, redball: bool = False, shape: int = 256, traj=None):
    np_env = np.array(env)

    # Note: Use inverted color!
    green_color = np.ones((shape, shape, 3))
    green_color[:, :, 0] = np_env[0]
    green_color[:, :, 1] = .2 * np_env[0]
    green_color[:, :, 2] = np_env[0]

    purple_color = np.ones((shape, shape, 3))
    purple_color[:, :, 0] = .7 * np_env[1]
    purple_color[:, :, 1] = np_env[1]
    purple_color[:, :, 2] = .2 * np_env[1]

    black_color = np.ones((shape, shape, 3))
    black_color[:, :, 0] = np_env[2]
    black_color[:, :, 1] = np_env[2]
    black_color[:, :, 2] = np_env[2]

    if redball:
        red_color = np.ones((shape, shape, 3))
        red_color[:, :, 0] = .2 * np_env[3]
        red_color[:, :, 1] = np_env[3]
        red_color[:, :, 2] = np_env[3]

    img = green_color + purple_color + black_color + red_color if redball else green_color + purple_color + black_color

    if traj is not None:
        plt.scatter(traj[:, 0], traj[:, 1], color='white', edgecolor='red', s=5)

    img = 1 - img

    plt.imshow(img)
    plt.gca().invert_yaxis()
    plt.show()


def visualize_traj(traj, start, end):
    fig, axes = plt.subplots(1, figsize=(5, 5))

    f_start = start * 50 + 128
    f_end = end * 50 + 128

    axes.scatter(traj[:, 0], traj[:, 1], alpha=0.8, color='white', edgecolor='blue', s=5)
    axes.scatter(f_start[0], f_start[1], alpha=1, color='white', edgecolor='red', s=5)
    axes.scatter(f_end[0], f_end[1], alpha=1, color='white', edgecolor='red', s=5)
    axes.set_xlim([0, 256])
    axes.set_ylim([0, 256])

    plt.show()


