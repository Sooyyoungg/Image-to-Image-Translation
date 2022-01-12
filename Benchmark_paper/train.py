import argparse
import os
import time
import tensorboardX
import shutil
import torch.utils.data
from data_loader import data_load
from utils.utilization import mkdirs, convert, get_config
from Benchmark_paper.models.model import smri2scalar_Trainer
from utils.visualization import tensorboard_vis

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='../configs/smri2dwi.yaml', help='Path to the config file.')
parser.add_argument('--data_root', type=str, default='', help='Path to the data, if None, get from config files')
parser.add_argument("--resume", type=int, default=0)
opts = parser.parse_args()

### Load experiment setting
config = get_config(opts.config)
n_epochs = config['n_epoch']
n_iterations = config['n_iterations']
display_size = config['display_size']
batch_size = config['batch_size']

### Setup model and data loader
model = smri2scalar_Trainer(config)
model.to(model.device)

# Load data
data_path='/scratch/connectome/GANBERT/data/sample/final/'
train_dataset = data_load(data_path+'train')
val_dataset = data_load(data_path+'test')

data_loader_train = torch.utils.data.DataLoader(dataset= train_dataset,
                                           batch_size=batch_size, shuffle=False,  # shuffle in data loader (speed up)
                                           num_workers=2,
                                           pin_memory=False)
data_loader_val = torch.utils.data.DataLoader(dataset= val_dataset,
                                           batch_size=batch_size, shuffle=False,
                                           num_workers=2,
                                           pin_memory=False)

### Setup logger and output folders
log_dir = config['log_dir']
if not os.path.exists(log_dir):
    print('* Creating log directory: ' + log_dir)
    mkdirs(log_dir)
print('* Logs will be saved under: ' + log_dir)
train_writer = tensorboardX.SummaryWriter(log_dir)
print('* Creating tensorboard summary writer ...')
if not os.path.exists(os.path.join(log_dir, 'config.yaml')):
    shutil.copy(opts.config, os.path.join(log_dir, 'config.yaml')) # copy config file to output folder

## Load model
if config['pretrained'] != '':
    model.resume(config['pretrained'])

load_epoch = int(opts.resume)
iterations = 0
if opts.resume > 0:
    # 가장 마지막으로 학습된 log 기록을 통해서 어디까지 학습됐는지 불러오기
    with open(log_dir + '/latest_log.txt', 'r') as f:
        x = f.readlines()[0]
        load_epoch, iterations = int(x.split(',')[0]), int(x.split(',')[1])
        if load_epoch == -1:
            load_epoch = int(iterations/ len(data_loader_train))
        if iterations == -1:
            iterations = load_epoch*len(data_loader_train)

    # 학습된 부분까지의 model 불러오기
    load_suffix = 'epoch%d.pt'%load_epoch
    if not os.path.exists(log_dir + '/gen_' + load_suffix):
        load_suffix = 'latest.pt'
    if not os.path.exists(log_dir + '/gen_latest.pt'):
        load_suffix = 'best.pt'
    print('* Resume training from {}'.format(load_suffix))

    state_dict = torch.load(log_dir + '/gen_'+load_suffix)
    model.gen_a.load_state_dict(state_dict['a'])

    opt_dict = torch.load(log_dir + '/opt_' + load_suffix)
    model.gen_opt.load_state_dict(opt_dict['gen'])

    if model.gan_w > 0:
        state_dict = torch.load(log_dir + '/dis_'+load_suffix)
        model.dis.load_state_dict(state_dict['dis'])
        model.dis_opt.load_state_dict(opt_dict['dis'])

## Start training
print('* Training from epoch %d'%load_epoch)
print('lambda L1: %.2f, gan: %.2f'%(model.l1_w, model.gan_w))
best_train_loss, best_val_loss = 999, 999
epoch = load_epoch
start = time()
while epoch < n_epochs or iterations < n_iterations:
    epoch += 1
    for num, data, targets in enumerate(data_loader_train):
        iterations = num + epoch * len(data_loader_train)
        # 학습 시간 출력
        start = time()
        train_dict = model.train(data, targets)
        end = time()
        update_t = end - start

        # Loss 값 계산
        ldwi = model.loss_dwi.item()
        lg, ld = model.loss_g.item(), model.loss_d.item()
        loss_print = ''
        loss_print += ' Loss_dwi: %.4f'%ldwi if model.l1_w>0 else ''
        loss_print += ' Loss_g: %.4f, Loss_d: %.4f'%(lg, ld) if model.gan_w > 0 else ''
        print('[Time %.3fs/epoch %d: %d/%d, Iter: %d (lr:%.5f)] '%(update_t, epoch, num, len(data_loader_train),
                                                             iterations, model.gen_opt.param_groups[0]['lr']) + loss_print)
        # Update learning rate
        model.update_learning_rate()

        # Dump training stats in log file
        if (iterations + 1) % config['log_iter'] == 0:
            if ldwi > 0: train_writer.add_scalar('loss_dwi', model.loss_dwi, iterations)
            if ld > 0: train_writer.add_scalar('loss_d', model.loss_d, iterations)
            if lg > 0: train_writer.add_scalar('loss_g', model.loss_g, iterations)

        # Write images
        if (iterations + 1) % config['image_display_iter'] == 0:
            with torch.no_grad():
                data_test, targets_test = next(iter(data_loader_val))
                test_ret = model.valid(data_test, targets_test)
                imgs_vis = [test_ret[k] for k in test_ret.keys() if isinstance(test_ret[k], np.ndarray)]
                imgs_titles = list(test_ret.keys())
                print(imgs_titles)
                cmaps = ['jet' if 'seg' in i else 'gist_gray' for i in imgs_titles]
                writer = tensorboard_vis(summarywriter=train_writer, step=iterations, board_name='val/',
                                         num_row=2, img_list=imgs_vis, cmaps=cmaps,
                                         titles=imgs_titles, resize=True)

                imgs_vis = [train_dict[k] for k in train_dict.keys()]
                imgs_titles = list(train_dict.keys())
                cmaps = ['jet' if 'seg' in i else 'gist_gray' for i in imgs_titles]
                writer = tensorboard_vis(summarywriter=train_writer, step=iterations, board_name='train/',
                                         num_row=2, img_list=imgs_vis, cmaps=cmaps,
                                         titles=imgs_titles, resize=True)

    # Save network weights
    if (epoch + 1) % config['snapshot_save_iter'] == 0:
        model.save(log_dir, epoch, iterations)
    if (epoch + 1) % config['latest_save_iter'] == 0:
        model.save(log_dir, -1)
        with open(log_dir + '/latest_log.txt', 'w') as f:
            f.writelines('%d, %d'%(epoch, iterations))
end = time()
print('Training finished in {}, {} epochs, {} iterations'.format(convert(end-start), epoch, iterations))