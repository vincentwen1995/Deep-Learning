import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets
import numpy as np


class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim=784):
        super(Generator, self).__init__()

        # Construct generator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear args.latent_dim -> 128
        #   LeakyReLU(0.2)
        #   Linear 128 -> 256
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 256 -> 512
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 512 -> 1024
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 1024 -> 768
        #   Output non-linearity
        self.layers = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, output_dim),
            nn.Tanh(),
        )

    def forward(self, z):
        # Generate images from z
        return self.layers(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim=784):
        super(Discriminator, self).__init__()

        # Construct distriminator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear 784 -> 512
        #   LeakyReLU(0.2)
        #   Linear 512 -> 256
        #   LeakyReLU(0.2)
        #   Linear 256 -> 1
        #   Output non-linearity
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        # return discriminator score for img
        return self.layers(img)


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D, args):
    g_loss = nn.BCELoss()
    d_loss = nn.BCELoss()
    losses_g = np.zeros((args.n_epochs, len(dataloader.dataset) // args.batch_size + 1))
    losses_d = np.zeros((args.n_epochs, len(dataloader.dataset) // args.batch_size + 1))
    for epoch in range(args.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            batch_size = imgs.shape[0]
            imgs = imgs.view(batch_size, -1).to(args.device)
            # Train Generator
            # ---------------
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, args.latent_dim).to(args.device)
            g_z = generator(z)
            prediction = discriminator(g_z)
            ones = torch.ones_like(prediction, device=args.device)
            loss_g = g_loss(prediction, ones)
            loss_g.backward()
            optimizer_G.step()
            losses_g[epoch, i] = loss_g.item()
            # Train Discriminator
            # -------------------
            optimizer_D.zero_grad()
            # Train on real data.
            prediction = discriminator(imgs)
            loss_d_real = d_loss(prediction, ones)
            loss_d_real.backward()
            # Train on fake data.
            prediction = discriminator(g_z.detach())
            zeros = torch.zeros_like(prediction, device=args.device)
            loss_d_fake = d_loss(prediction, zeros)
            loss_d_fake.backward()
            loss_d = loss_d_real + loss_d_fake
            optimizer_D.step()
            losses_d[epoch, i] = loss_d.item()
            # Save Images
            # -----------
            batches_done = epoch * len(dataloader) + i
            if batches_done % args.save_interval == 0:
                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:
                # save_image(gen_imgs[:25],
                #            'images/{}.png'.format(batches_done),
                #            nrow=5, normalize=True)
                save_image(g_z[:25, :].view(-1, 1, 28, 28), 'results/gan/{}.png'.format(batches_done), nrow=5, normalize=True)
        print('Epoch{}, Generator loss: {:.3f}, Discriminator loss: {:.3f}'.format(epoch, loss_g.item(), loss_d.item()))
    np.savez('losses.npz', losses_g=losses_g, losses_d=losses_d)


def main(args):
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create output image directory
    os.makedirs('results/gan', exist_ok=True)
    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,),
                                                (0.5,))])),
        batch_size=args.batch_size, shuffle=True)

    # Initialize models and optimizers
    generator = Generator(latent_dim=args.latent_dim)
    generator.to(args.device)
    discriminator = Discriminator()
    discriminator.to(args.device)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D, args)

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    torch.save(generator.state_dict(), "mnist_generator_latent_{}.pt".format(args.latent_dim))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')

    args = parser.parse_args()

    main(args)
