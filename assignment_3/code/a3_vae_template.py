import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

from datasets.bmnist import bmnist
from scipy.stats import norm
import os


class Encoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20, input_dim=784):
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mean = nn.Linear(hidden_dim, z_dim)
        self.std = nn.Linear(hidden_dim, z_dim)

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """
        mean, std = None, None
        hidden = self.relu(self.hidden(input))
        mean = self.mean(hidden)
        std = self.relu(self.std(hidden))
        return mean, std


class Decoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20, input_dim=784):
        super().__init__()
        self.hidden = nn.Linear(z_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mean = nn.Linear(hidden_dim, input_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        """
        Perform forward pass of decoder.

        Returns mean with shape [batch_size, 784].
        """
        mean = None
        hidden = self.relu(self.hidden(input))
        mean = self.sigmoid(self.mean(hidden))
        return mean


class VAE(nn.Module):

    def __init__(self, device, hidden_dim=500, z_dim=20):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.encoder = Encoder(hidden_dim, z_dim)
        self.decoder = Decoder(hidden_dim, z_dim)
        self.device = device

    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """
        average_negative_elbo = None
        mean, std = self.encoder(input)
        # print(std)
        # print(torch.log(torch.prod(std**2, dim=-1)))
        # print(torch.sum(std, dim=-1))
        # print(torch.diag(mean @ mean.t()))
        # print(mean.shape[-1])

        # reg_loss = 0.5 * (-torch.log(torch.prod(std**2, dim=-1)) + torch.sum(std, dim=-1) + torch.diag(mean @ mean.t()) - mean.shape[-1]).mean(dim=0)
        epsilon_slack = 1e-12
        reg_loss = 0.5 * (-torch.log(std**2 + epsilon_slack) + std**2 + mean**2 - 1).sum(dim=-1).mean(dim=0)
        eps = torch.randn(mean.shape, device=self.device)
        z = mean + eps * std
        dec_mean = self.decoder(z)
        # recon_loss = torch.log(torch.bernoulli(dec_mean)).sum(dim=-1).mean(dim=0)
        # recon_loss = nn.functional.binary_cross_entropy(dec_mean, input)
        recon_loss = -((input * torch.log(dec_mean + epsilon_slack) + (1 - input) * torch.log(1 - dec_mean)).sum(dim=-1).mean(dim=0))
        average_negative_elbo = reg_loss + recon_loss
        return average_negative_elbo

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """
        sampled_ims, im_means = None, None
        with torch.no_grad():
            z = torch.randn(n_samples, self.z_dim, device=self.device)
            im_means = self.decoder(z)
            sampled_ims = torch.bernoulli(im_means)
        return sampled_ims, im_means


def epoch_iter(model, data, optimizer, ARGS):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """
    average_epoch_elbo = torch.Tensor().new_empty(len(data.dataset) // data.batch_size + 1)
    average_epoch_elbo = average_epoch_elbo.to(ARGS.device)
    if model.training:
        for step, imgs in enumerate(data):
            imgs = imgs.to(ARGS.device)
            imgs = imgs.view(imgs.shape[0], -1)
            optimizer.zero_grad()
            average_negative_elbo = model(imgs)
            average_negative_elbo.backward()
            optimizer.step()
            average_epoch_elbo[step] = average_negative_elbo.item()
    else:
        for step, imgs in enumerate(data):
            imgs = imgs.to(ARGS.device)
            imgs = imgs.view(imgs.shape[0], -1)
            average_epoch_elbo[step] = model(imgs).item()
    average_epoch_elbo = average_epoch_elbo.mean().item()
    return average_epoch_elbo


def run_epoch(model, data, optimizer, ARGS):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data

    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer, ARGS)

    model.eval()
    val_elbo = epoch_iter(model, valdata, optimizer, ARGS)
    return train_elbo, val_elbo


def save_elbo_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train negative elbo')
    plt.plot(val_curve, label='validation negative elbo')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('Negative ELBO')
    plt.tight_layout()
    plt.savefig(filename)


def main(ARGS):
    os.makedirs('results/vae', exist_ok=True)
    prefix = 'epochs{}_zdim{}_'.format(ARGS.epochs, ARGS.zdim)
    ARGS.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = bmnist()[:2]  # ignore test split
    model = VAE(ARGS.device, z_dim=ARGS.zdim)
    model.to(ARGS.device)
    optimizer = torch.optim.Adam(model.parameters())

    train_curve, val_curve = [], []

    for epoch in range(ARGS.epochs):
        elbos = run_epoch(model, data, optimizer, ARGS)
        train_elbo, val_elbo = elbos
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        print(f"[Epoch {epoch}] train elbo: {train_elbo} val_elbo: {val_elbo}")

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functioanlity that is already imported.
        # --------------------------------------------------------------------
        if epoch in (0, int(ARGS.epochs//2), ARGS.epochs - 1):
            sample_imgs, sample_means = model.sample(100)
            sample_imgs = sample_imgs.view(-1, 1, 28, 28)
            sample_means = sample_means.view(-1, 1, 28, 28)
            imgs_grid = make_grid(sample_imgs, nrow=10, normalize=True).cpu().numpy()
            plt.imsave('results/vae/' + prefix + 'sample_imgs_epoch{}.png'.format(epoch), np.transpose(imgs_grid, (1, 2, 0)))
            means_grid = make_grid(sample_means, nrow=10, normalize=True).cpu().numpy()
            plt.imsave('results/vae/' + prefix + 'sample_means_epoch{}.png'.format(epoch), np.transpose(means_grid, (1, 2, 0)))

    # --------------------------------------------------------------------
    #  Add functionality to plot plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.
    # --------------------------------------------------------------------
    if ARGS.zdim == 2:
        model.eval()
        resolution = 20
        z1, z2 = np.linspace(0.001, 0.999, resolution), np.linspace(0.001, 0.999, resolution)
        z1, z2 = np.meshgrid(norm.ppf(z1), norm.ppf(z2))
        with torch.no_grad():
            manifold = torch.stack((torch.from_numpy(z1.astype(np.float32)), torch.from_numpy(
                z2.astype(np.float32)))).permute((1, 2, 0)).view(-1, 2).to(device=ARGS.device)
            means = model.decoder(manifold)
            imgs = torch.bernoulli(means)
        means = means.view(-1, 1, 28, 28)
        imgs = imgs.view(-1, 1, 28, 28)
        imgs_grid = make_grid(imgs, nrow=resolution, normalize=True).cpu().numpy()
        plt.imsave('results/vae/' + prefix + 'manifold_imgs.png', np.transpose(imgs_grid, (1, 2, 0)))
        means_grid = make_grid(means, nrow=resolution, normalize=True).cpu().numpy()
        plt.imsave('results/vae/' + prefix + 'manifold_means.png', np.transpose(means_grid, (1, 2, 0)))

    save_elbo_plot(train_curve, val_curve, 'results/vae/' + prefix + 'elbo.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')
    ARGS = parser.parse_args()

    main(ARGS)
