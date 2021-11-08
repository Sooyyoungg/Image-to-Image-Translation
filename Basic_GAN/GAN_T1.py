### Basic
import argparse
import os
import numpy as np
import pandas as pd

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from DataSplit import DataSplit
from model import Generator
from model import Discriminator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

os.makedirs("T1_T2_generated_images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")

opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)
#print(img_shape)  # (1, 28, 28)

### Data Loader
"""os.makedirs("../../data/mnist", exist_ok=True)

dataloader = DataLoader(
    datasets.MNIST("../../data/mnist",train=True,download=True,transform=transforms.Compose([transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]),),
    batch_size=opt.batch_size,
    shuffle=True)"""

#data_path = pd.read_csv('/scratch/GANBERT/sooyounng/abcd_t1_t2_diffusion_info.csv')
data_path = '/scratch/GANBERT/sooyoung/abcd_t1_t2_info.csv'

total_data = pd.read_csv(data_path)
total_N = len(total_data)
N_train_dataset = int(total_N * 0.6)
N_val_dataset = int(total_N * 0.2)
N_test_dataset = total_N - N_train_dataset - N_val_dataset
#print(total_N, N_train_dataset, N_val_dataset, N_test_dataset)  #8921 5352 1784 1785

# split
data_loader_train, data_loader_val, data_loader_test = DataSplit(csv_file=data_path, n_train=N_train_dataset, n_val=N_val_dataset, n_test=N_test_dataset, transform=False)

# load
data_loader_train = torch.utils.data.DataLoader(data_loader_train, batch_size=opt.batch_size, shuffle=False, num_workers=2, pin_memory=False)
data_loader_val = torch.utils.data.DataLoader(data_loader_val, batch_size=opt.batch_size, shuffle=False, num_workers=2, pin_memory=False)
data_loader_test = torch.utils.data.DataLoader(data_loader_test, batch_size=opt.batch_size, shuffle=False, num_workers=2, pin_memory=False)

print(data_loader_train.shape, data_loader_val.shape, data_loader_test.shape)


### Generator & Discriminator
# Initialize generator and discriminator
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Loss function
adversarial_loss = torch.nn.BCELoss().to(device)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


### Training
Tensor = torch.cuda.FloatTensor if device else torch.FloatTensor

for epoch in range(opt.n_epochs):
    """ Training """
    for i, (imgs, _) in enumerate(data_loader_train):
        # print(imgs.shape)  # torch.Size([64, 1, 28, 28])

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        """ Generator """
        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        """ Discriminator """
        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
              % (epoch, opt.n_epochs, i, len(data_loader_train), d_loss.item(), g_loss.item()))

        batches_done = epoch * len(data_loader_train) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "T1_T2_generated_images/%d.png" % batches_done, nrow=5, normalize=True)

        """ Validation """
        if i % 10 == 0:
            with torch.no_grad():
                val_loss = 0.0
                for v, (v_imgs, _) in enumerate(data_loader_val):
                    val_imgs = Variable(v_imgs.type(Tensor))
                    g_val = generator(val_imgs)


torch.save({
    'epoch': opt.n_epochs,
    'batch': opt.batch_size,
    'model_G_state_dict': generator.state_dict(),
    'model_D_state_dict': discriminator.state_dict(),
    'optimizer_G_state_dict': optimizer_G.state_dict(),
    'optimizer_D_state_dict': optimizer_D.state_dict()}, 'model_weights.pth')

### Testing
# with torch.no_grad():
