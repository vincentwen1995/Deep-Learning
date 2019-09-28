import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets
import numpy as np
from a3_gan_template import Generator
import matplotlib.pyplot as plt


def save_loss_plot(generator_curve, discriminator_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(generator_curve, label='Generator Loss')
    plt.plot(discriminator_curve, label='Discriminator Loss')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig(filename)


def main(args):
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    steps = 9
    latent_dim = int(args.checkpoint.split('_')[3][:3])
    generator = Generator(latent_dim)
    generator.load_state_dict(torch.load(args.checkpoint))
    generator.to(args.device)
    z1 = torch.randn(latent_dim, device=args.device)
    z2 = torch.randn(latent_dim, device=args.device)
    z = torch.zeros(steps, latent_dim, device=args.device)
    for d in np.arange(latent_dim):
        z[:, d] = torch.linspace(z1[d], z2[d], steps=steps, device=args.device)

    g_z = generator(z)
    save_image(g_z.view(-1, 1, 28, 28), 'results/gan/interpolated.png', nrow=steps, normalize=True)

    npzfile = np.load('losses.npz')
    losses_g = npzfile['losses_g']
    losses_d = npzfile['losses_d']
    losses_g = losses_g.mean(-1)
    losses_d = losses_d.mean(-1)
    save_loss_plot(losses_g, losses_d, 'results/gan/losses.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='mnist_generator_latent_100.pt',
                        help='saved checkpoint to load from')

    args = parser.parse_args()
    main(args)
