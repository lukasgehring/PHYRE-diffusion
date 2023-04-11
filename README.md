# PHYRE-Diffusion

This Repo contains the code for a Diffusion-Model that predicts trajectories for the PHYRE dataset.

## Setup the project

First you need to download the PHYRE-dataset from https://uni-bielefeld.sciebo.de/s/Izuy8U1dwDjhKYm and move it into the main directory of this project.

You also need to create a folder `trained_models` in the `ConVAE` directory and a `models` folder in the main directory.

Now you can install the environment by running
```
conda env create -f environment.yml
```

## Train the VAE
If you want to train the model, you first need to train the `VAE` by running `trainVAE.py`.
You can set the following parameters
* `redball`: If redball is `True`, the model is trained including the red ball, otherwise the red ball is not included.
* `lr`: Learning rate
* `epochs`: Number of epochs, the model is trained
* `batch_size`: Batch Size

The model is then saved in `/ConVAE/trained-models`.

## Train the Diffusion Model
If you want to train the diffusion model, you need to run `train.py`.
You can set the following parameters:

* `VAE_name`: Name of the VAE
* `num_steps`: = Number of diffusion steps
* `sliding_window`: Size of the sliding window (e.g. 1, 3, 5, 9)
* `batch_size`: Batch Size
* `training_templates`: Templates used for training
* `redball`: If redball is `True`, the model is trained including the red ball, otherwise the red ball is not included.
* `use_start_end_points`: If the start and endpoint of the trajectory should be inserted into the model
* `use_sin_cos_t`: Whether t should be calcuated by means of sine and cosine
* `learning_rate`: Learning rate
* `loss_function`: The used loss function. You can choose between `l1`, `l2` and `l4`
* `epochs`: Number of epochs, the model is trained

The trained model is then saved in `/models`.

## Run the model
If the model is trained, you can predict trajectories by running `run.py`. The model is then loaded from `/models` and for each
task from defined templates, a loss is computed. You can also visualize the prediction of the first task.
You can set the following parameters:
* `num_steps`: = Number of diffusion steps
* `sliding_window`: Size of the sliding window (e.g. 1, 3, 5, 9)
* `templates`: Templates used for validation
* `redball`: If redball is `True`, the model is trained including the red ball, otherwise the red ball is not included.
* `visualize`: If you want to visualize the true and predicted trajectory.

## Modifying the project

### Diffusion
You can modify the diffusion strength by updating the `start` and `end` parameters of the `make_beta_schedule()` function in `train.py` and `run.py`

### Loss Functions
You can add more loss functions to the `noise_estimation_loss()` function in `diffusion.py`.
