# add atmos, trans_map, refinement
# add validation
from __future__ import print_function
import argparse
import os
import sys
import random

import time
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as utils
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from new_model_final import bn_init_as_tf, weights_init_xavier, Model_exp3
from loss_final import GeneratorLoss
from data_final import getTrainLoader, getValLoader, UnNormalize
from utils import AverageMeter
# from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import compare_psnr, compare_ssim
import cv2


# Arguments
parser = argparse.ArgumentParser(description='Dehaze Network')
parser.add_argument('--model_name',                type=str,   help='model name', default='dehaze_densenet121')
parser.add_argument('--use_benchmark', dest='use_benchmark',   help='use benchmark', action='store_true', default=False)
# Training
parser.add_argument('--fix_first_conv_blocks',                 help='if set, will fix the first two conv blocks', action='store_true')
parser.add_argument('--fix_first_conv_block',                  help='if set, will fix the first conv block', action='store_true')
parser.add_argument('--bn_no_track_stats',                     help='if set, will not track running stats in batch norm layers', action='store_true')
parser.add_argument('--weight_decay',              type=float, help='weight decay factor for optimization', default=1e-2)
# parser.add_argument('--bts_size',                  type=int,   help='initial num_filters in bts', default=512)
parser.add_argument('--retrain',                               help='if used with checkpoint_path, will restart training from step zero', action='store_true')
parser.add_argument('--adam_eps',                  type=float, help='epsilon in Adam optimizer', default=1e-6)
parser.add_argument('--batch_size',                type=int,   help='training batch size', default=4)
parser.add_argument('--num_epochs',                type=int,   help='number of total epochs to run', default=150)
parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=0.00005)

# Log and save
parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='./log_model_thesis')
parser.add_argument('--checkpoint_path',           type=str,   help='path of the dehaze model checkpoint to load', default='')
parser.add_argument('--log_freq',                  type=int,   help='Logging frequency in iterations', default= 50)
parser.add_argument('--save_model_interval',       type=int,   help='Checkpoint saving frequency in epoch', default=1)


parser.add_argument('--train_dir', required=False, default='./nyu_ots_haze_uniform_train2', help='path to train dataset')
parser.add_argument('--val_dir', required=False, default='./nyu_ots_haze_uniform_val2', help='path to validation dataset')
parser.add_argument('--image_size', type=int, default=320, help='the height and width of input image to network')
parser.add_argument('--val_interval', default=2, type=int, help='validation interval(iteration)')
parser.add_argument('--val_img_interval', default=4, type=int, help='interval for saving validation images(iteration)')


opt = parser.parse_args()
print(opt)


def set_misc(model):
    if opt.bn_no_track_stats:
        print("Disabling tracking running stats in batch norm layers")
        model.apply(bn_init_as_tf)

    if opt.fix_first_conv_blocks:
        fixing_layers = ['conv0', 'denseblock1.denselayer1', 'denseblock1.denselayer2', 'norm']
        print("Fixing first two conv blocks")
    elif opt.fix_first_conv_block:
        fixing_layers = ['conv0', 'denseblock1.denselayer1', 'norm']
        print("Fixing first conv block")
    else:
        fixing_layers = ['conv0', 'norm']
        print("Fixing first conv layer")

    # for name, child in model.named_children():
    #     if not 'encoder' in name:
    #         continue
    for name, parameters in model.named_parameters():
        # print(name, name2)
        if any(x in name for x in fixing_layers):
            parameters.requires_grad = False


if not os.path.exists(opt.log_directory):
    os.makedirs(opt.log_directory)

# opt.manualSeed = random.randint(1, 10000)
# opt.manualSeed = 101
# random.seed(opt.manualSeed)
# torch.manual_seed(opt.manualSeed)
# torch.cuda.manual_seed_all(opt.manualSeed)
# print("Random Seed: ", opt.manualSeed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

if opt.use_benchmark and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    print("Using cudnn.benchmark")

# Load data, get dataloader
train_loader, train_num = getTrainLoader( opt.train_dir,opt.batch_size, opt.image_size,trans_atmos=True,crop=True, enhance= True)
val_loader, val_num = getValLoader( opt.val_dir, opt.batch_size, opt.image_size,trans_atmos=True, crop=True,  enhance= False)

# Create model
model = Model_exp3()
# model = Model_v2()
model.train()
# Initialize decoder_A weight
model.decoder_A.apply(weights_init_xavier)
model.decoder_T.apply(weights_init_xavier)
model.generate_dehaze.apply(weights_init_xavier)
# Choose whether to fix first two conv layers in encoder
set_misc(model.encoder)
# Put on gpu
model = model.to(device)
print('Dehaze Model is created.')
print('Total number of parameters:', sum(param.numel() for param in model.parameters()))
print('Total number of learning parameters:', sum(param.numel() for param in model.parameters() if param.requires_grad) )


# Training parameters: optimizer
optimizer = torch.optim.AdamW([{'params': model.encoder.parameters(), 'weight_decay': opt.weight_decay},
                                   {'params': model.decoder_T.parameters(), 'weight_decay': 0},
                                   {'params': model.decoder_A.parameters(), 'weight_decay': 0},
                                   {'params': model.generate_dehaze.parameters(), 'weight_decay': 0}],
                                  lr=opt.learning_rate, eps=opt.adam_eps)
# optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
# scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler_rule2)

# Loss
loss_criterion = GeneratorLoss(w_trans =1., w_atmos = 1., w_nonhaze_rec = 1., w_dehaze=1.5,use_weighted_mse = True)
# Initialize step number

# Loading dehaze model ( encoder + decoder_T + decoder_A)
if opt.checkpoint_path != '' and os.path.isfile(opt.checkpoint_path):
    print("Loading dehaze checkpoint '{}'".format(opt.checkpoint_path))
    checkpoint = torch.load(opt.checkpoint_path)

    # model.load_state_dict(checkpoint['model'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']+1
    iteration = checkpoint['iteration']+1
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    print('Loading dehaze checkpoint path:', opt.checkpoint_path)
    print('Continuing training at epoch %d' % start_epoch)
    # model_just_loaded = True
else:
    start_epoch = 1
    iteration = 0

# Retrain the loading model
if opt.retrain:
    start_epoch =1
    iteration=0

# Logging
writer = SummaryWriter(opt.log_directory + '/' + opt.model_name + '/summaries', flush_secs=30)

var_sum = [var.sum() for var in model.parameters() if var.requires_grad]
var_cnt = len(var_sum)
var_sum = np.sum(var_sum)

print("Initial variables' sum: {:.3f}, avg: {:.3f}".format(var_sum, var_sum/var_cnt))

# loss estimators
# time_estimator = AverageMeter()
loss_estimator = AverageMeter()

trans_estimator = AverageMeter()
trans_l2_estimator = AverageMeter()
trans_grad_estimator = AverageMeter()
trans_ssim_estimator = AverageMeter()

atmos_estimator = AverageMeter()
atmos_l2_estimator = AverageMeter()
# atmos_grad_estimator = AverageMeter()

# nonhaze_hsv_estimator = AverageMeter()
# nonhaze_hsv_l2_estimator = AverageMeter()
# nonhaze_hsv_grad_estimator = AverageMeter()
# nonhaze_hsv_ssim_estimator = AverageMeter()

# dehaze_hsv_estimator = AverageMeter()
# dehaze_hsv_l2_estimator = AverageMeter()
# dehaze_hsv_grad_estimator = AverageMeter()
# dehaze_hsv_ssim_estimator = AverageMeter()

nonhaze_estimator = AverageMeter()
nonhaze_l2_estimator = AverageMeter()
nonhaze_ssim_estimator = AverageMeter()
nonhaze_per_estimator = AverageMeter()

dehaze_estimator = AverageMeter()
dehaze_l2_estimator = AverageMeter()
dehaze_ssim_estimator = AverageMeter()
dehaze_per_estimator = AverageMeter()

batch_time_estimator = AverageMeter()

unnormalize_fun = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

losses_list = []

train_loader_len = len(train_loader)
num_total_steps = opt.num_epochs * train_loader_len

# early stopping
best_val_psnr = -1
best_val_ssim = -1
early_stop=0

# # Start training...
for epoch in range(start_epoch, opt.num_epochs+1):
# for epoch in range(start_epoch, 3):

    loss_estimator.reset()
    trans_estimator.reset()
    trans_l2_estimator.reset()
    trans_grad_estimator.reset()
    trans_ssim_estimator.reset()
    atmos_estimator.reset()
    atmos_l2_estimator.reset()
    # atmos_grad_estimator.reset()

    # nonhaze_hsv_estimator.reset()
    # nonhaze_hsv_l2_estimator.reset()
    # nonhaze_hsv_grad_estimator.reset()
    # nonhaze_hsv_ssim_estimator.reset()
    # dehaze_hsv_estimator.reset()
    # dehaze_hsv_l2_estimator.reset()
    # dehaze_hsv_grad_estimator.reset()
    # dehaze_hsv_ssim_estimator.reset()

    nonhaze_estimator.reset()
    nonhaze_l2_estimator.reset()
    nonhaze_ssim_estimator.reset()
    nonhaze_per_estimator.reset()
    dehaze_estimator.reset()
    dehaze_l2_estimator.reset()
    dehaze_ssim_estimator.reset()
    dehaze_per_estimator.reset()
    batch_time_estimator.reset()



    # Switch to train mode
    model.train()
    set_misc(model.encoder)

    start_time = time.time()
    for i, sample_batched in enumerate(train_loader,0):
        iteration += 1
        optimizer.zero_grad()

        # Prepare sample and target
        hazes = sample_batched['haze'].to(device)
        trans = sample_batched['trans'].to(device)
        # depths = sample_batched['depth'].to(device)
        atmos = sample_batched['atmos'].to(device)
        # inverse_depths = 1- sample_batched['depth']
        # inverse_depths_norm = DepthNorm(inverse_depths)
        # inverse_depths_norm = (inverse_depths_norm).to(device)
        gts = sample_batched['gt'].to(device)

        # # Predict
        trans_map, atmos_light, nonhaze, dehaze = model(hazes)

        loss = loss_criterion( nonhaze, dehaze, gts, trans_map, trans, atmos_light, atmos)

        # Loss backpropagation
        loss_estimator.update(loss.item(), hazes.size(0))

        trans_estimator.update(loss_criterion.last_trans_loss.item(), hazes.size(0))
        trans_l2_estimator.update(loss_criterion.last_trans_l2.item(), hazes.size(0))
        trans_grad_estimator.update(loss_criterion.last_trans_grad.item(), hazes.size(0))
        trans_ssim_estimator.update(loss_criterion.last_trans_ssim.item(), hazes.size(0))

        atmos_estimator.update(loss_criterion.last_atmos_loss.item(), hazes.size(0))
        atmos_l2_estimator.update(loss_criterion.last_atmos_l2.item(), hazes.size(0))
        # atmos_grad_estimator.update(loss_criterion.last_atmos_grad.item(), hazes.size(0))

        # nonhaze_hsv_estimator.update(loss_criterion.last_nonhaze_rec_hsv_loss.item(), hazes.size(0))
        # nonhaze_hsv_l2_estimator.update(loss_criterion.last_nonhaze_rec_hsv_l2.item(), hazes.size(0))
        # nonhaze_hsv_grad_estimator.update(loss_criterion.last_nonhaze_rec_hsv_grad.item(), hazes.size(0))
        # nonhaze_hsv_ssim_estimator.update(loss_criterion.last_nonhaze_rec_hsv_ssim.item(), hazes.size(0))

        # dehaze_hsv_estimator.update(loss_criterion.last_dehaze_hsv_loss.item(), hazes.size(0))
        # dehaze_hsv_l2_estimator.update(loss_criterion.last_dehaze_hsv_l2.item(), hazes.size(0))
        # dehaze_hsv_grad_estimator.update(loss_criterion.last_dehaze_hsv_grad.item(), hazes.size(0))
        # dehaze_hsv_ssim_estimator.update(loss_criterion.last_dehaze_hsv_ssim.item(), hazes.size(0))

        nonhaze_estimator.update(loss_criterion.last_nonhaze_rec_loss.item(), hazes.size(0))
        nonhaze_l2_estimator.update(loss_criterion.last_nonhaze_rec_l2.item(), hazes.size(0))
        nonhaze_ssim_estimator.update(loss_criterion.last_nonhaze_rec_ssim.item(), hazes.size(0))
        nonhaze_per_estimator.update(loss_criterion.last_nonhaze_rec_per.item(), hazes.size(0))

        dehaze_estimator.update(loss_criterion.last_dehaze_loss.item(), hazes.size(0))
        dehaze_l2_estimator.update(loss_criterion.last_dehaze_l2.item(), hazes.size(0))
        dehaze_ssim_estimator.update(loss_criterion.last_dehaze_ssim.item(), hazes.size(0))
        dehaze_per_estimator.update(loss_criterion.last_dehaze_per.item(), hazes.size(0))

        # Update optimizer
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time_estimator.update(time.time()-start_time)
        start_time = time.time()
        eta = str(datetime.timedelta(seconds=int(batch_time_estimator.val*(train_loader_len - (i+1) ))))

        if iteration % opt.log_freq == 0:
            # Print to console
            print('Epoch: [{0}][{1}/{2}]\t'
            'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
            'ETA {eta}\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            .format(epoch, i, train_loader_len, batch_time=batch_time_estimator, loss=loss_estimator, eta=eta))
            # 'Learning_Rate {lr:.6f}'
            # .format(epoch, i, train_loader_len, batch_time=batch_time_estimator, loss=loss_estimator, eta=eta, lr = current_lr))
            atmos_map = atmos_light.unsqueeze(-1).unsqueeze(-1)
            atmos_map = atmos_map.repeat(1,1,dehaze.size()[-2],dehaze.size()[-1])
            for index in range(len(dehaze)):
                writer.add_image('Train/transmission_map/{}'.format(index), trans_map[index], iteration)
                writer.add_image('Train/atmos_light/{}'.format(index), atmos_map[index], iteration)
                writer.add_image('Train/nonhaze_rec/{}'.format(index), nonhaze[index], iteration)
                writer.add_image('Train/dehaze/{}'.format(index), dehaze[index], iteration)
                # writer.add_image('Train/haze_rec/{}'.format(index), haze_rec[index], iteration)
                writer.add_image('Train/gts/{}'.format(index), gts[index], iteration)
                # writer.add_image('Train/hazes/{}'.format(index), hazes_unnormalized[index], iteration)
                # writer.add_image('Train/transmission_map/{}'.format(index), trans_map[index], iteration)

            writer.flush()
    # Log to tensorboard
    writer.add_scalar('Train param/learning_rate', scheduler.get_lr()[0], epoch)
    writer.add_scalar('Train/Loss.avg', loss_estimator.avg, epoch)
    writer.add_scalar('Train/trans_loss', trans_estimator.avg, epoch)
    writer.add_scalar('Train/trans_l2_loss', trans_l2_estimator.avg, epoch)
    writer.add_scalar('Train/trans_grad_loss', trans_grad_estimator.avg, epoch)
    writer.add_scalar('Train/trans_ssim_loss', trans_ssim_estimator.avg, epoch)
    writer.add_scalar('Train/atmos_loss', atmos_estimator.avg, epoch)
    writer.add_scalar('Train/atmos_l2_loss', atmos_l2_estimator.avg, epoch)
    # writer.add_scalar('Train/atmos_grad_loss', atmos_grad_estimator.avg, epoch)

    # writer.add_scalar('Train/nonhaze_hsv_loss', nonhaze_hsv_estimator.avg, epoch)
    # writer.add_scalar('Train/nonhaze_hsv_l2_loss', nonhaze_hsv_l2_estimator.avg, epoch)
    # writer.add_scalar('Train/nonhaze_hsv_grad_loss', nonhaze_hsv_grad_estimator.avg, epoch)
    # writer.add_scalar('Train/nonhaze_hsv_ssim_loss', nonhaze_hsv_ssim_estimator.avg, epoch)
    # writer.add_scalar('Train/dehaze_hsv_loss', dehaze_hsv_estimator.avg, epoch)
    # writer.add_scalar('Train/dehaze_hsv_l2_loss', dehaze_hsv_l2_estimator.avg, epoch)
    # writer.add_scalar('Train/dehaze_hsv_grad_loss', dehaze_hsv_grad_estimator.avg, epoch)
    # writer.add_scalar('Train/dehaze_hsv_ssim_loss', dehaze_hsv_ssim_estimator.avg, epoch)

    writer.add_scalar('Train/nonhaze_loss', nonhaze_estimator.avg, epoch)
    writer.add_scalar('Train/nonhaze_l2_loss', nonhaze_l2_estimator.avg, epoch)
    writer.add_scalar('Train/nonhaze_ssim_loss', nonhaze_ssim_estimator.avg, epoch)
    writer.add_scalar('Train/nonhaze_per_loss', nonhaze_per_estimator.avg, epoch)
    writer.add_scalar('Train/dehaze_loss', dehaze_estimator.avg, epoch)
    writer.add_scalar('Train/dehaze_l2_loss', dehaze_l2_estimator.avg, epoch)
    writer.add_scalar('Train/dehaze_ssim_loss', dehaze_ssim_estimator.avg, epoch)
    writer.add_scalar('Train/dehaze_per_loss', dehaze_per_estimator.avg, epoch)

    scheduler.step()
    print('Learning Rate:{0:.6f}'.format(scheduler.get_lr()[0]))

    losses_list.append(loss_estimator.avg)

    # save model parameters
    if epoch % opt.save_model_interval == 0:
        state_dict = {
            'epoch': epoch,
            'iteration': iteration,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        torch.save(state_dict, '%s/netG_epoch_%d.pth' % (opt.log_directory, epoch))

    # validation step
    if epoch % opt.val_interval == 0 or epoch % opt.val_img_interval == 0:
    # if iteration % opt.val_interval == 0 or iteration % opt.val_interval == 0:
        # early_stop += 1
        loss_estimator.reset()
        trans_estimator.reset()
        trans_l2_estimator.reset()
        trans_grad_estimator.reset()
        trans_ssim_estimator.reset()
        atmos_estimator.reset()
        atmos_l2_estimator.reset()
        # atmos_grad_estimator.reset()

        # nonhaze_hsv_estimator.reset()
        # nonhaze_hsv_l2_estimator.reset()
        # nonhaze_hsv_grad_estimator.reset()
        # nonhaze_hsv_ssim_estimator.reset()
        # dehaze_hsv_estimator.reset()
        # dehaze_hsv_l2_estimator.reset()
        # dehaze_hsv_grad_estimator.reset()
        # dehaze_hsv_ssim_estimator.reset()

        nonhaze_estimator.reset()
        nonhaze_l2_estimator.reset()
        nonhaze_ssim_estimator.reset()
        nonhaze_per_estimator.reset()
        dehaze_estimator.reset()
        dehaze_l2_estimator.reset()
        dehaze_ssim_estimator.reset()
        dehaze_per_estimator.reset()
        batch_time_estimator.reset()

        model.eval()

        with torch.no_grad():
            # initialize variables to estimate averages
            psnr_sum = ssim_sum= 0
            # col_loss_sum = per_loss_sum = 0
            # val_loader_len = len(val_loader)

            for i, val_batched in enumerate(val_loader,0):

                # Prepare sample and target
                hazes = val_batched['haze'].to(device)
                trans = sample_batched['trans'].to(device)
                atmos = sample_batched['atmos'].to(device)
                # depths = val_batched['depth'].to(device)
                gts = val_batched['gt'].to(device)



                # Predict, add refinement block
                trans_map, atmos_light, nonhaze, dehaze = model(hazes)

                loss = loss_criterion( nonhaze, dehaze, gts, trans_map, trans, atmos_light, atmos)
                # compute loss
                loss_estimator.update(loss.item(), hazes.size(0))

                trans_estimator.update(loss_criterion.last_trans_loss.item(), hazes.size(0))
                trans_l2_estimator.update(loss_criterion.last_trans_l2.item(), hazes.size(0))
                trans_grad_estimator.update(loss_criterion.last_trans_grad.item(), hazes.size(0))
                trans_ssim_estimator.update(loss_criterion.last_trans_ssim.item(), hazes.size(0))

                atmos_estimator.update(loss_criterion.last_atmos_loss.item(), hazes.size(0))
                atmos_l2_estimator.update(loss_criterion.last_atmos_l2.item(), hazes.size(0))
                # atmos_grad_estimator.update(loss_criterion.last_atmos_grad.item(), hazes.size(0))

                # nonhaze_hsv_estimator.update(loss_criterion.last_nonhaze_rec_hsv_loss.item(), hazes.size(0))
                # nonhaze_hsv_l2_estimator.update(loss_criterion.last_nonhaze_rec_hsv_l2.item(), hazes.size(0))
                # nonhaze_hsv_grad_estimator.update(loss_criterion.last_nonhaze_rec_hsv_grad.item(), hazes.size(0))
                # nonhaze_hsv_ssim_estimator.update(loss_criterion.last_nonhaze_rec_hsv_ssim.item(), hazes.size(0))

                # dehaze_hsv_estimator.update(loss_criterion.last_dehaze_hsv_loss.item(), hazes.size(0))
                # dehaze_hsv_l2_estimator.update(loss_criterion.last_dehaze_hsv_l2.item(), hazes.size(0))
                # dehaze_hsv_grad_estimator.update(loss_criterion.last_dehaze_hsv_grad.item(), hazes.size(0))
                # dehaze_hsv_ssim_estimator.update(loss_criterion.last_dehaze_hsv_ssim.item(), hazes.size(0))

                nonhaze_estimator.update(loss_criterion.last_nonhaze_rec_loss.item(), hazes.size(0))
                nonhaze_l2_estimator.update(loss_criterion.last_nonhaze_rec_l2.item(), hazes.size(0))
                nonhaze_ssim_estimator.update(loss_criterion.last_nonhaze_rec_ssim.item(), hazes.size(0))
                nonhaze_per_estimator.update(loss_criterion.last_nonhaze_rec_per.item(), hazes.size(0))

                dehaze_estimator.update(loss_criterion.last_dehaze_loss.item(), hazes.size(0))
                dehaze_l2_estimator.update(loss_criterion.last_dehaze_l2.item(), hazes.size(0))
                dehaze_ssim_estimator.update(loss_criterion.last_dehaze_ssim.item(), hazes.size(0))
                dehaze_per_estimator.update(loss_criterion.last_dehaze_per.item(), hazes.size(0))


                # add for saving image results
                # val_batch_output = vutils.make_grid(nonhaze_rec, nrow=8, normalize=False)
                val_batch_output = vutils.make_grid(dehaze, nrow=8, normalize=False)
                # writer.add_image('Val/non_haze_reconstruct_'+ str(epoch)+'_'+str(i+1), val_batch_output, iteration )
                # modify not saving images
                # vutils.save_image(val_batch_output, '%s/generated_epoch_%08d_%08d.png' % (opt.log_directory, epoch,i+1), normalize=False, scale_each=False)

                # add for saving image results
                if epoch % opt.val_img_interval == 0:
                    # val_batch_output = vutils.make_grid(dehaze, nrow=8, normalize=False)
                    # vutils.save_image(val_batch_output, '%s/generated_epoch_%08d_%08d.png' % (opt.log_directory, epoch,i+1), normalize=False, scale_each=False)

                    atmos_map = atmos_light.unsqueeze(-1).unsqueeze(-1)
                    atmos_map = atmos_map.repeat(1,1,dehaze.size()[-2],dehaze.size()[-1])
                    for index in range(len(dehaze)):
                        writer.add_image('Val/transmission_map/{}'.format(index), trans_map[index], iteration)
                        writer.add_image('Val/atmos_light/{}'.format(index), atmos_map[index], iteration)
                        writer.add_image('Val/nonhaze_rec/{}'.format(index), nonhaze[index], iteration)
                        writer.add_image('Val/dehaze/{}'.format(index), dehaze[index], iteration)
                        writer.add_image('Val/gts/{}'.format(index), gts[index], iteration)
                        # writer.add_image('Val/transmission_map/{}'.format(index), trans_map[index], iteration)
                        # writer.add_image('Val/haze_rec/{}'.format(index), haze_rec[index], iteration)

                    writer.flush()


                for j in range(dehaze.size(0)):
                    # (C, H, W)
                    # nonhaze_rec_img = nonhaze_rec[j].detach().cpu().clone().numpy()
                    nonhaze_rec_img = dehaze[j].detach().cpu().clone().numpy()
                    gt_img = gts[j].detach().cpu().clone().numpy()
                    # (H, W, C)
                    nonhaze_rec_img = (255*np.transpose(nonhaze_rec_img, (1,2,0))).astype(np.uint8)
                    gt_img = (255*np.transpose(gt_img,(1,2,0))).astype(np.uint8)
                    # cv2.imwrite("nonhaze.png", nonhaze_rec_img)
                    # cv2.imwrite("gt.png", gt_img)
                    # print("nonhaze, gt dtype:", nonhaze_rec_img.dtype, gt_img.dtype)
                    # print("max,min:", np.max(nonhaze_rec_img), np.min(nonhaze_rec_img), np.max(gt_img), np.min(gt_img))
                    psnr = compare_psnr(nonhaze_rec_img, gt_img, data_range=255)
                    ssim = compare_ssim(nonhaze_rec_img, gt_img,multichannel=True, data_range=255.)
                    # print("psnr:", psnr, "ssim:", ssim)
                    psnr_sum += psnr
                    ssim_sum += ssim
                    # psnr_sum += compare_psnr(nonhaze_rec_img, gt_img)
                    # ssim_sum += compare_ssim(nonhaze_rec_img, gt_img,multichannel=True)


            print("Average PSNR :", psnr_sum/val_num,"Average SSIM:", ssim_sum/val_num)

            writer.add_scalar('Val/psnr', psnr_sum /val_num , epoch)
            writer.add_scalar('Val/ssim', ssim_sum /val_num , epoch)

            writer.add_scalar('Val/Loss.avg', loss_estimator.avg, epoch)
            writer.add_scalar('Val/trans_loss', trans_estimator.avg, epoch)
            writer.add_scalar('Val/trans_l2_loss', trans_l2_estimator.avg, epoch)
            writer.add_scalar('Val/trans_grad_loss', trans_grad_estimator.avg, epoch)
            writer.add_scalar('Val/trans_ssim_loss', trans_ssim_estimator.avg, epoch)
            writer.add_scalar('Val/atmos_loss', atmos_estimator.avg, epoch)
            writer.add_scalar('Val/atmos_l2_loss', atmos_l2_estimator.avg, epoch)
            # writer.add_scalar('Val/atmos_grad_loss', atmos_grad_estimator.avg, epoch)

            # writer.add_scalar('Val/nonhaze_hsv_loss', nonhaze_hsv_estimator.avg, epoch)
            # writer.add_scalar('Val/nonhaze_hsv_l2_loss', nonhaze_hsv_l2_estimator.avg, epoch)
            # writer.add_scalar('Val/nonhaze_hsv_grad_loss', nonhaze_hsv_grad_estimator.avg, epoch)
            # writer.add_scalar('Val/nonhaze_hsv_ssim_loss', nonhaze_hsv_ssim_estimator.avg, epoch)
            # writer.add_scalar('Val/dehaze_hsv_loss', dehaze_hsv_estimator.avg, epoch)
            # writer.add_scalar('Val/dehaze_hsv_l2_loss', dehaze_hsv_l2_estimator.avg, epoch)
            # writer.add_scalar('Val/dehaze_hsv_grad_loss', dehaze_hsv_grad_estimator.avg, epoch)
            # writer.add_scalar('Val/dehaze_hsv_ssim_loss', dehaze_hsv_ssim_estimator.avg, epoch)
            writer.add_scalar('Val/nonhaze_loss', nonhaze_estimator.avg, epoch)
            writer.add_scalar('Val/nonhaze_l2_loss', nonhaze_l2_estimator.avg, epoch)
            writer.add_scalar('Val/nonhaze_ssim_loss', nonhaze_ssim_estimator.avg, epoch)
            writer.add_scalar('Val/nonhaze_per_loss', nonhaze_per_estimator.avg, epoch)
            writer.add_scalar('Val/dehaze_loss', dehaze_estimator.avg, epoch)
            writer.add_scalar('Val/dehaze_l2_loss', dehaze_l2_estimator.avg, epoch)
            writer.add_scalar('Val/dehaze_ssim_loss', dehaze_ssim_estimator.avg, epoch)
            writer.add_scalar('Val/dehaze_per_loss', dehaze_per_estimator.avg, epoch)


plt.figure(figsize=(10,5))
plt.title("Loss During Training")
plt.plot(losses_list)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


# tensorboard --logdir=


