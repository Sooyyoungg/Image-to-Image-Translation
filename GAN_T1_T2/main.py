### Basic
import argparse
import gc
import os
import numpy as np
import pandas as pd
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

from DataSplit import DataSplit
from models.basic_GAN import Generator, Discriminator
from utils import GANLoss
from utils import set_requires_grad

gc.collect()

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
parser.add_argument("--img_shape", type=int, default=img_shape, help="img_shape")
#print(img_shape)  # (1, 28, 28)

### Data Loader
data_path = '/home/connectome/conmaster/GANBERT/abcd_t1_t2_diffusion_info.csv'

total_data = pd.read_csv(data_path)
total_N = len(total_data)
N_train_dataset = int(total_N * 0.6)
N_val_dataset = int(total_N * 0.2)
N_test_dataset = total_N - N_train_dataset - N_val_dataset
#print(total_N, N_train_dataset, N_val_dataset, N_test_dataset)  #8921 5352 1784 1785

# split
DataSplit = DataSplit()
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
criterionGAN = GANLoss(opt.gan_mode).to(device)
criterionL1 = torch.nn.L1Loss()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

### Training
Tensor = torch.cuda.FloatTensor if device else torch.FloatTensor

for epoch in range(opt.n_epochs):
    generator.train()
    discriminator.train()

    """ Training """
    for i, (T1, T2) in enumerate(data_loader_train):
        # print(imgs.shape)  # torch.Size([64, 1, 28, 28])

        # Configure input
        real_T1 = Variable(T1.type(Tensor))
        real_T2 = Variable(T2.type(Tensor))

        """ Discriminator """
        set_requires_grad(discriminator, True)
        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        loss_D_real = criterionGAN(discriminator(real_T2.detach()), True)
        loss_D_fake = criterionGAN(discriminator(fake_T2.detach()), False)
        loss_D = loss_D_real + loss_D_fake

        loss_D.backward()
        optimizer_D.step()

        """ Generator """
        set_requires_grad(discriminator, False)
        optimizer_G.zero_grad()

        # Sample noise as generator input
        real_T1_noise = Variable(Tensor(np.random.normal(0, 1, (real_T1.shape[0], opt.latent_dim))))

        # Generate a batch of images
        fake_T2 = generator(real_T1_noise)

        # Loss measures generator's ability to fool the discriminator
        loss_G_GAN = criterionGAN(discriminator(fake_T2), True)
        loss_G_L1 = criterionL1(fake_T2, real_T2)
        loss_G = loss_G_GAN + loss_G_L1

        loss_G.backward()
        optimizer_G.step()

        print("Training: [Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
              % (epoch, opt.n_epochs, i, len(data_loader_train), loss_D.item(), loss_G.item()))

        batches_done = epoch * len(data_loader_train) + i
        if batches_done % opt.sample_interval == 0:
            save_image(fake_T2.data[:25], "T1_T2_generated_images/%d.png" % batches_done, nrow=5, normalize=True)

    """ Validation """
    if epoch % 10 == 0:
        generator.eval()
        discriminator.eval()
        with torch.no_grad():
            val_loss_D = []
            val_loss_G = []

            for v, (v_T1, v_T2) in enumerate(data_loader_val):
                v_real_T1 = Variable(v_T1.type(Tensor))
                v_real_T2 = Variable(v_T2.type(Tensor))
                v_fake_T2 = generator(v_real_T1)

                """ Discriminator Loss """
                v_loss_D_fake = criterionGAN(discriminator(v_fake_T2), False)
                v_loss_D_real = criterionGAN(discriminator(v_real_T2), True)
                v_loss_D = v_loss_D_real + v_loss_D_fake
                val_loss_D.append(v_loss_D)

                """ Generator Loss """
                v_loss_G_GAN = criterionGAN(discriminator(v_fake_T2), True)
                v_loss_G_L1 = criterionL1(v_fake_T2, v_real_T2)
                v_loss_G = v_loss_G_GAN + v_loss_G_L1
                val_loss_G.append(v_loss_G)

            val_loss_D = np.array(val_loss_D)
            val_loss_G = np.array(val_loss_G)
            print("Valindation: [Epoch %d/%d] [Batch %d/%d] [Avg D loss: %f] [Avg G loss: %f]"
                    % (epoch, opt.n_epochs, i, len(data_loader_train), np.mean(val_loss_D), np.mean(val_loss_G.item)))
            print("====================================================================")

torch.save({
    'epoch': opt.n_epochs,
    'batch': opt.batch_size,
    'model_G_state_dict': generator.state_dict(),
    'model_D_state_dict': discriminator.state_dict(),
    'optimizer_G_state_dict': optimizer_G.state_dict(),
    'optimizer_D_state_dict': optimizer_D.state_dict()}, 'model_weights.pth')

### Testing
generator.eval()
discriminator.eval()

with torch.no_grad():
    test_loss_D = []
    test_loss_G = []

    for t, (t_T1, t_T2) in enumerate(data_loader_test):
        t_real_T1 = Variable(t_T1.type(Tensor))
        t_real_T2 = Variable(t_T2.type(Tensor))
        t_fake_T2 = generator(t_real_T1)

        """ Discriminator Loss """
        t_loss_D_fake = criterionGAN(discriminator(t_fake_T2), False)
        t_loss_D_real = criterionGAN(discriminator(t_real_T2), True)
        t_loss_D = t_loss_D_real + t_loss_D_fake
        test_loss_D.append(t_loss_D)

        """ Generator Loss """
        t_loss_G_GAN = criterionGAN(discriminator(t_fake_T2), True)
        t_loss_G_L1 = criterionL1(t_fake_T2, t_real_T2)
        t_loss_G = t_loss_G_GAN + t_loss_G_L1
        test_loss_G.append(t_loss_G)

        if t%100 == 0:
            save_image(t_fake_T2.data[:25], "T1_T2_test_generated_images/%d.png" % t/100, nrow=5, normalize=True)

    test_loss_D = np.array(test_loss_D)
    test_loss_G = np.array(test_loss_G)
    print("Test: [Avg D loss: %f] [Avg G loss: %f]" % (np.mean(test_loss_D), np.mean(test_loss_G.item)))

gc.collect()