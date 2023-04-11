import torch
import torch.nn.functional as F
from torchsummary import summary
from torchvision import transforms, utils
from torch.utils.data import DataLoader

from ConVAE.ConVAE import ConvVAE
from ConVAE.loader import EnvironmentDataset, ToTensor

# -------------------------------------
# PARAMETERS
# -------------------------------------

# load the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# set environment paramerts
redball = False
if redball:
    input_channels = 4
else:
    input_channels = 3

model = ConvVAE(image_channels=input_channels).to(device)

# set learning parameters
lr = 0.001
epochs = 1
batch_size = 16
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# show summary of the model
summary(model, (input_channels, 64, 64))

# resize input from 256x256 to 64x64
transform = transforms.Compose([
    ToTensor(),
    transforms.Resize((64, 64), antialias=True),
])

# load dataset
dataset = EnvironmentDataset(transform=transform, redball=redball)
trainloader = DataLoader(
    dataset, batch_size=batch_size, shuffle=True
)


# train the model for one epoch
def train(model, dataloader, device, optimizer):
    # set model for training
    model.train()

    running_loss = 0.0
    counter = 0

    for i, data in enumerate(dataloader):
        counter += 1
        data = data.to(device)

        reconstruction, mu, logVar = model(data)

        # The loss is the BCE loss combined with the KL divergence to ensure the distribution is learnt
        kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
        loss = F.binary_cross_entropy(reconstruction, data, size_average=False) + kl_divergence

        optimizer.zero_grad()
        loss.backward()
        running_loss += loss.item()
        optimizer.step()

    train_loss = running_loss / counter

    return train_loss


# can be used to validate the model
def validate(model, dataloader, dataset, device):
    # set model for evalulation
    model.eval()

    running_loss = 0.0
    counter = 0

    with torch.no_grad():

        for i, data in enumerate(dataloader):

            counter += 1
            data = data.to(device)

            reconstruction, mu, logVar = model(data)

            kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
            loss = F.binary_cross_entropy(reconstruction, data, size_average=False) + kl_divergence

            running_loss += loss.item()

            # save the last batch input and output of every epoch
            if i == int(len(dataset) / dataloader.batch_size) - 1:
                recon_images = reconstruction
                true_image = data

    val_loss = running_loss / counter

    return val_loss, recon_images, true_image


# train the model
for epoch in range(epochs):

    train_epoch_loss = train(model, trainloader, device, optimizer)

    if epoch % 10 == 0:
        print(f"Epoch {epoch + 1} of {epochs}, Train Loss: {train_epoch_loss:.4f}")

# save trained model to disk
if redball:
    torch.save(model, f'ConVAE/trained-models/conv_vae64_redball_{epochs}_epochs.pt')
else:
    torch.save(model, f'ConVAE/trained-models/conv_vae64_{epochs}_epochs.pt')