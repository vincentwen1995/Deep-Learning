import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from datasets.mnist import mnist
import os
from torchvision.utils import make_grid


def log_prior(x):
    """
    Compute the elementwise log probability of a standard Gaussian, i.e.
    N(x | mu=0, sigma=1).
    """
    logp = -0.5 * np.log(2 * np.pi) - 0.5 * x**2
    return logp


def sample_prior(size):
    """
    Sample from a standard Gaussian.
    """
    sample = torch.randn(size)
    if torch.cuda.is_available():
        sample = sample.cuda()

    return sample


def get_mask():
    mask = np.zeros((28, 28), dtype='float32')
    for i in range(28):
        for j in range(28):
            if (i + j) % 2 == 0:
                mask[i, j] = 1

    mask = mask.reshape(1, 28*28)
    mask = torch.from_numpy(mask)
    # if torch.cuda.is_available():
    # mask = mask.cuda()
    return mask


class Coupling(torch.nn.Module):
    def __init__(self, c_in, mask, n_hidden=1024):
        super().__init__()
        self.n_hidden = n_hidden
        self.c_in = c_in

        # Assigns mask to self.mask and creates reference for pytorch.
        self.register_buffer('mask', mask)

        # Create shared architecture to generate both the translation and
        # scale variables.
        # Suggestion: Linear ReLU Linear ReLU Linear.
        self.nn = torch.nn.Sequential(
            nn.Linear(c_in, self.n_hidden // 2),
            nn.ReLU(),
            nn.Linear(self.n_hidden // 2, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, c_in * 2),
        )

        # The nn should be initialized such that the weights of the last layer
        # is zero, so that its initial transform is identity.
        self.nn[-1].weight.data.zero_()
        self.nn[-1].bias.data.zero_()

    def forward(self, z, ldj, reverse=False):
        # Implement the forward and inverse for an affine coupling layer. Split
        # the input using the mask in self.mask. Transform one part with
        # Make sure to account for the log Jacobian determinant (ldj).
        # For reference, check: Density estimation using RealNVP.

        # NOTE: For stability, it is advised to model the scale via:
        # log_scale = tanh(h), where h is the scale-output
        # from the NN.
        combined = self.nn(self.mask * z)
        log_scale = torch.tanh(combined[:, :self.c_in])
        translation = combined[:, self.c_in:]

        if not reverse:
            z = self.mask * z + (1 - self.mask) * (z * torch.exp(log_scale) + translation)
            ldj += (log_scale * (1 - self.mask)).sum(-1)
        else:
            z = self.mask * z + (1 - self.mask) * (z - translation) * torch.exp(-log_scale)
        return z, ldj


class Flow(nn.Module):
    def __init__(self, shape, n_flows=4):
        super().__init__()
        channels, = shape

        mask = get_mask()

        self.layers = torch.nn.ModuleList()

        for i in range(n_flows):
            self.layers.append(Coupling(c_in=channels, mask=mask))
            self.layers.append(Coupling(c_in=channels, mask=1-mask))

        self.z_shape = (channels,)

    def forward(self, z, logdet, reverse=False):
        if not reverse:
            for layer in self.layers:
                z, logdet = layer(z, logdet)
        else:
            for layer in reversed(self.layers):
                z, logdet = layer(z, logdet, reverse=True)

        return z, logdet


class Model(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.flow = Flow(shape)

    def dequantize(self, z):
        return z + torch.rand_like(z)

    def logit_normalize(self, z, logdet, reverse=False):
        """
        Inverse sigmoid normalization.
        """
        alpha = 1e-5

        if not reverse:
            # Divide by 256 and update ldj.
            z = z / 256.
            logdet -= np.log(256) * np.prod(z.size()[1:])

            # Logit normalize
            z = z*(1-alpha) + alpha*0.5
            logdet += torch.sum(-torch.log(z) - torch.log(1-z), dim=1)
            z = torch.log(z) - torch.log(1-z)

        else:
            # Inverse normalize
            logdet += torch.sum(torch.log(z) + torch.log(1-z), dim=1)
            z = torch.sigmoid(z)

            # Multiply by 256.
            z = z * 256.
            logdet += np.log(256) * np.prod(z.size()[1:])

        return z, logdet

    def forward(self, input):
        """
        Given input, encode the input to z space. Also keep track of ldj.
        """
        z = input
        ldj = torch.zeros(z.size(0), device=z.device)

        z = self.dequantize(z)
        z, ldj = self.logit_normalize(z, ldj)

        z, ldj = self.flow(z, ldj)

        # Compute log_pz and log_px per example
        # z_0 = sample_prior(input.size())
        log_px = log_prior(z).sum(-1) + ldj
        return log_px

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Sample from prior and create ldj.
        Then invert the flow and invert the logit_normalize.
        """
        with torch.no_grad():
            z = sample_prior((n_samples,) + self.flow.z_shape)
            ldj = torch.zeros(z.size(0), device=z.device)

            z, ldj = self.flow(z, ldj, reverse=True)
            z, ldj = self.logit_normalize(z, ldj, reverse=True)

        return z


def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average bpd ("bits per dimension" which is the negative
    log_2 likelihood per dimension) averaged over the complete epoch.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    avg_bpd = None
    avg_bpd = torch.Tensor().new_empty(len(data.dataset) // data.batch_size + 1)
    avg_bpd = avg_bpd.to(device)
    if model.training:
        for step, (imgs, _) in enumerate(data):
            imgs = imgs.to(device)
            # imgs = imgs.view(-1, 28, 28)
            optimizer.zero_grad()
            log_px = model(imgs)
            avg_bpd_batch = -log_px.mean()
            avg_bpd_batch.backward()
            optimizer.step()
            avg_bpd[step] = avg_bpd_batch.item() / (np.log(2) * 784)
    else:
        for step, (imgs, _) in enumerate(data):
            imgs = imgs.to(device)
            # imgs = imgs.view(-1, 28, 28)
            log_px = model(imgs)
            avg_bpd_batch = -log_px.mean()
            avg_bpd[step] = avg_bpd_batch.item() / (np.log(2) * 784)
    avg_bpd = avg_bpd.mean().item()
    return avg_bpd


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average bpd for each.
    """
    traindata, valdata = data

    model.train()
    train_bpd = epoch_iter(model, traindata, optimizer)

    model.eval()
    val_bpd = epoch_iter(model, valdata, optimizer)

    return train_bpd, val_bpd


def save_bpd_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train bpd')
    plt.plot(val_curve, label='validation bpd')
    plt.plot([1.85 for _ in np.arange(len(train_curve))], '-', label='baseline bpd')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('bpd')
    plt.tight_layout()
    plt.savefig(filename)


def main(ARGS):
    data = mnist()[:2]  # ignore test split

    model = Model(shape=[784])

    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    os.makedirs('results/nf/', exist_ok=True)

    train_curve, val_curve = [], []
    for epoch in range(ARGS.epochs):
        bpds = run_epoch(model, data, optimizer)
        train_bpd, val_bpd = bpds
        train_curve.append(train_bpd)
        val_curve.append(val_bpd)
        print("[Epoch {epoch}] train bpd: {train_bpd} val_bpd: {val_bpd}".format(
            epoch=epoch, train_bpd=train_bpd, val_bpd=val_bpd))

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functionality that is already imported.
        #  Save grid to images_nfs/
        # --------------------------------------------------------------------
        if epoch in (0, int(ARGS.epochs//2), ARGS.epochs - 1):
            sample_imgs = model.sample(100)
            sample_imgs = sample_imgs.view(-1, 1, 28, 28)
            imgs_grid = make_grid(sample_imgs, nrow=10, normalize=True).cpu().numpy()
            plt.imsave('results/nf/sample_imgs_epoch{}.png'.format(epoch), np.transpose(imgs_grid, (1, 2, 0)))

    save_bpd_plot(train_curve, val_curve, 'results/nf/nfs_bpd.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')

    ARGS = parser.parse_args()

    main(ARGS)
