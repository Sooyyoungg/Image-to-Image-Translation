import torch
import torch.nn as nn

class Pix2Pix(nn.Module):
    def __init__(self, opt):
        Pix2Pix.__init__(self, opt)

        self.losses = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        self.visual_names = ['real_T1', 'fake_T2', 'real_T2']   # T1->T2 translation

        if not self.isTrain:  # Testing
            self.model_names = ['G']
        else:                 # Training
            self.model_names = ['G', 'D']
            # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()

            # define optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer.append(self.optimizer_G)
            self.optimizer.append(self.optimizer_D)

    def set_input(self, input):
        T1toT2 = self.opt.direction == 'T1toT2'
        self.real_T1 = input['T1' if T1toT2 else 'T2'].to(self.device)
        self.real_T2 = input['T2' if T1toT2 else 'T1'].to(self.device)
        self.image_path = input['T1_paths' if T1toT2 else 'T2_paths']

    def forward(self):
        # Generator
        self.fake_T2 = self.netG(self.real_T1)

    def backward_G(self):
        # condition으로 real_T1을 Discriminator에 넣어줌
        fake_T1T2 =torch.cat((self.real_T1, self.fake_T2), 1)
        pred_fake = self.netD(fake_T1T2)
        # Discriminator가 fake를 real이라고 판단하게 만들기 위해
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # fake_T2를 real_T2와 유사하게 만들기 위해
        self.loss_G_L1 = self.criterionL1(self.fake_T2, self.real_T2) * self.opt.lambda_L1

        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def backward_D(self):
        # condition으로 real_T1을 Discriminator에 넣어줌
        fake_T1T2 = torch.cat((self.real_T1, self.fake_T2), 1)
        pred_fake = self.netD(fake_T1T2.detach())  # detach(): Generator의 weight 값을 update하지 않기 위함
        self.loss_D_fake = self.criterionGAN(pred_fake, False)  # fake를 fake로 판단할 수 있도록

        real_T1T2 = torch.cat((self.real_T1, self.real_T2), 1)
        pred_real = self.netD(real_T1T2.detach())
        self.loss_D_real = self.criterionGAN(pred_real, True)  # real을 real로 판단할 수 있도록

        self.loss_D = self.loss_D_fake + self.loss_D_real
        self.loss_D.backward()

    def optimize_parameters(self):
        # fake_T2 생성
        self.forward()

        # update Discriminator : real->real / fake->fake로 잘 판단할 수 있도록
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # update Generator : D에서 fake->real로 판단할 수 있도록 + fake_T2가 real_T2와 비슷해질 수 있도록
        self.set_requires_grad(self.netD, False) # D에 대한 weight은 update하지 X
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

