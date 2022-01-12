import torch
from torch import nn
import torch.nn.functional as F
from Benchmark_paper.utils.torchutils import weights_init, get_scheduler
import os
from networks import smri2scalarGen

class smri2scalar_Trainer(nn.Module):
    def __init__(self, hyperparameters,input_dim, output_dim):
        super(smri2scalar_Trainer, self).__init__()
        lr = hyperparameters['lr']

        # Initiate the networks
        self.multimodal = hyperparameters['multimodal_t1'] or hyperparameters['multimodal_t2']
        self.t1 = hyperparameters['multimodal_t1'] > 0
        self.t2 = hyperparameters['multimodal_t2'] > 0
        self.dwi = hyperparameters['multimodal_dwi'] > 0
        assert 1 + hyperparameters['multimodal_t1'] + hyperparameters['multimodal_t2'] == hyperparameters['input_dim']
        self.gen_a = smri2scalarGen(input_dim, output_dim, hyperparameters['gen'])  # auto-encoder for domain a

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        gen_params = list(self.gen_a.parameters())
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # GPU setting
        gpu_ids = hyperparameters['gpu_ids']
        self.device = torch.device('cuda:{}'.format(gpu_ids)) if gpu_ids else torch.device('cpu') # get device name: CPU or GPU
        print('Deploy to {}'.format(self.device))
        self.gen_a.to(self.device)

        # Network weight initialization
        self.gen_a.apply(weights_init(hyperparameters['init']))
        print('Init generator with {}'.format(hyperparameters['init']))

        self.loss_translate = torch.zeros([]).to(self.device)
        self.l1_w = hyperparameters['l1_w']

    ### Functions
    def recon_criterion(self, input, target, brain_mask):
        pix_loss =  F.l1_loss(input, target, reduction='sum')
        pix_loss = pix_loss / (brain_mask.sum() + 1e-10)
        loss = pix_loss
        return loss

    def forward(self, struct):
        self.eval()
        g_dwi = self.gen_a.forward(struct)
        self.train()
        return g_dwi

    def prepare_data(self, data_dict):
        return_dict = {}
        if self.t1:
            in_t1 = data_dict['t1'].to(self.device).float()
            inputs = in_t1
            return_dict['in_t1'] = in_t1[0, 0].cpu().numpy()
        if self.t2:
            in_t2 = data_dict['t2'].to(self.device).float()
            inputs = torch.cat((inputs, in_t2), dim=1)
            return_dict['in_t2'] = in_t2[0, 0].cpu().numpy()

        return inputs.to(self.device).float(), return_dict

    def valid(self, data_dict, trgs):
        self.gen_a.eval()
        in_i, return_dict = self.prepare_data(data_dict)  # torch.cat((in_t1, in_t2), dim=1)
        pred_i = self.gen_a.forward(in_i)
        target = None
        for ti in range(len(trgs)):
            return_dict['trg_%s' % trgs[ti]] = data_dict[trgs[ti]][0, 0].cpu().numpy()
            if target is None:
                target = data_dict[trgs[ti]].to(self.device).float()
            else:
                target = torch.cat((target, data_dict[trgs[ti]].to(self.device).float()), dim=1)
        loss_derives = self.recon_criterion(pred_i, target, in_i.sum(dim=1)>0.)
        self.gen_a.train()
        for ti in range(len(trgs)):
            return_dict[trgs[ti]] = pred_i[0, ti].detach().cpu().numpy()
        return_dict['loss'] = loss_derives.item()
        return return_dict

    def train(self, data_dict, targets_dwi):
        self.gen_opt.zero_grad()
        self.loss_derives = torch.zeros([]).to(self.device)
        self.loss_g = torch.zeros([]).to(self.device)
        self.loss_d = torch.zeros([]).to(self.device)
        in_i, return_dict = self.prepare_data(data_dict)#torch.cat((in_b0, in_t2),dim=1)

        target_dwi = None
        for ti in range(len(targets_dwi)):
            return_dict['target_%s' % targets_dwi[ti]] = data_dict[targets_dwi[ti]][0, 0].cpu().numpy()
            if target_dwi is None:
                target_dwi = data_dict[targets_dwi[ti]].to(self.device).float()
            else:
                target_dwi = torch.cat((target_dwi, data_dict[targets_dwi[ti]].to(self.device).float()), dim=1)
        pred_i = self.gen_a.forward(in_i)

        # 생성한 fake dwi와 real dwi 비교해 loss 계산
        self.loss_derives += self.l1_w * self.recon_criterion(pred_i, target_dwi, in_i.sum(dim=1)>0.)
        total_loss = self.loss_derives
        total_loss.backward()
        self.gen_opt.step()

        for ti in range(len(targets_dwi)):
            return_dict[targets_dwi[ti]] = pred_i[0, ti].detach().cpu().numpy()
        return return_dict

    def update_learning_rate(self):
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def set_requires_grad(self, nets, requires_grad=False):
        """
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        print('Load models from %s'%checkpoint_dir)
        state_dict = torch.load(checkpoint_dir)
        self.gen_a.load_state_dict(state_dict['a'])
        # Load optimizers
        last_opt_name = checkpoint_dir.replace("gen", "opt")
        state_dict = torch.load(last_opt_name)
        self.gen_opt.load_state_dict(state_dict['gen'])
        if self.gan_w > 0:
            self.dis_opt.load_state_dict(state_dict['dis'])
            self.dis.load_state_dict(torch.load(checkpoint_dir.replace("gen","dis")))
        return


    def save(self, snapshot_dir, epoch, step=-1, gen_name=None):
        # Save generators, discriminators, and optimizers
        if gen_name is None:
            if epoch == -1:
                gen_name = os.path.join(snapshot_dir, 'gen_latest.pt')
                opt_name = os.path.join(snapshot_dir, 'opt_latest.pt')
            else:
                if step == -1:
                    gen_name = os.path.join(snapshot_dir, 'gen_epoch%d.pt' % (epoch + 1))
                    opt_name = os.path.join(snapshot_dir, 'opt_epoch%d.pt' % (epoch + 1))
                else:
                    gen_name = os.path.join(snapshot_dir, 'gen_epoch%dstep%d.pt' % (epoch + 1, step))
                    opt_name = os.path.join(snapshot_dir, 'opt_epoch%dstep%d.pt' % (epoch + 1, step))
        else:
            opt_name = gen_name.replace('gen','opt')

        if self.gan_w > 0:
            # save optimizer for both D and G
            torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)
            # save discriminator if using gan
            torch.save({'dis': self.dis.state_dict()}, gen_name.replace('gen','dis'))
        else:
            # save optimizer for generator only
            torch.save({'gen': self.gen_opt.state_dict()}, opt_name)
        # save generator
        torch.save({'a': self.gen_a.state_dict()}, gen_name)