import os
import numpy as np
import time, datetime
import argparse
import copy
from thop import profile
from collections import OrderedDict

import math
import torch
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
from models.resnet_cifar10 import resnet_56,resnet_110

from data import cifar10
import utils

parser = argparse.ArgumentParser("CIFAR-10 training")

parser.add_argument(
    '--data_dir',
    type=str,
    default='./data',
    help='path to dataset')

parser.add_argument(
    '--arch',
    type=str,
    default='resnet_56',
    choices=('vgg_16_bn','resnet_56','resnet_110','resnet_50'),
    help='architecture to calculate feature maps')

parser.add_argument(
    '--lr_type',
    type=str,
    default='cos',
    help='lr type')

parser.add_argument(
    '--result_dir',
    type=str,
    default='./result',
    help='results path for saving models and loggers')

parser.add_argument(
    '--batch_size',
    type=int,
    default=256,
    help='batch size')

parser.add_argument(
    '--epochs',
    type=int,
    default=200,
    help='num of training epochs')

parser.add_argument(
    '--label_smooth',
    type=float,
    default=0,
    help='label smoothing')

parser.add_argument(
    '--learning_rate',
    type=float,
    default=0.01,
    help='init learning rate')

parser.add_argument(
    '--lr_decay_step',
    default='50,100',
    type=str,
    help='learning rate')

parser.add_argument(
    '--momentum',
    type=float,
    default=0.9,
    help='momentum')

parser.add_argument(
    '--weight_decay',
    type=float,
    default=0.005,
    help='weight decay')

parser.add_argument(
    '--pretrain_dir',
    type=str,
    default='',
    help='pretrain model path')

parser.add_argument(
    '--ci_dir',
    type=str,
    default='',
    help='ci path')

parser.add_argument(
    '--sparsity',
    type=str,
    default=None,
    help='sparsity of each conv layer')

parser.add_argument(
    '--gpu',
    type=str,
    default='0',
    help='gpu id')

args = parser.parse_args()
CLASSES = 10

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
CLASSES = 10
print_freq = (256*50)//args.batch_size

if not os.path.isdir(args.result_dir):
    os.makedirs(args.result_dir)

utils.record_config(args)
now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
logger = utils.get_logger(os.path.join(args.result_dir, 'logger'+now+'.log'))

#use for loading pretrain model
if len(args.gpu)>1:
    name_base='module.'
else:
    name_base=''

def load_vgg_model(model, oristate_dict):
    state_dict = model.state_dict()
    last_select_index = None #Conv index selected in the previous layer

    cnt=0
    prefix = args.ci_dir+'/ci_conv'
    subfix = ".npy"
    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, nn.Conv2d):

            cnt+=1
            oriweight = oristate_dict[name + '.weight']
            curweight = state_dict[name_base+name + '.weight']
            orifilter_num = oriweight.size(0)
            currentfilter_num = curweight.size(0)

            if orifilter_num != currentfilter_num:

                cov_id = cnt
                logger.info('loading ci from: ' + prefix + str(cov_id) + subfix)
                ci = np.load(prefix + str(cov_id) + subfix)
                select_index = np.argsort(ci)[orifilter_num-currentfilter_num:]  # preserved filter id
                select_index.sort()

                if last_select_index is not None:
                    for index_i, i in enumerate(select_index):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name_base+name + '.weight'][index_i][index_j] = \
                                oristate_dict[name + '.weight'][i][j]
                else:
                    for index_i, i in enumerate(select_index):
                       state_dict[name_base+name + '.weight'][index_i] = \
                            oristate_dict[name + '.weight'][i]

                last_select_index = select_index

            elif last_select_index is not None:
                for i in range(orifilter_num):
                    for index_j, j in enumerate(last_select_index):
                        state_dict[name_base+name + '.weight'][i][index_j] = \
                            oristate_dict[name + '.weight'][i][j]
            else:
                state_dict[name_base+name + '.weight'] = oriweight
                last_select_index = None

    model.load_state_dict(state_dict)

def load_resnet_model(model, oristate_dict, layer):
    cfg = {
        56: [9, 9, 9],
        110: [18, 18, 18],
    }

    state_dict = model.state_dict()

    current_cfg = cfg[layer]
    last_select_index = None

    all_conv_weight = []

    prefix = args.ci_dir+'/ci_conv'
    subfix = ".npy"

    cnt=1
    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'
        for k in range(num):
            for l in range(2):

                cnt+=1
                cov_id=cnt

                conv_name = layer_name + str(k) + '.conv' + str(l + 1)
                conv_weight_name = conv_name + '.weight'
                all_conv_weight.append(conv_weight_name)
                oriweight = oristate_dict[conv_weight_name]
                curweight =state_dict[name_base+conv_weight_name]
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                if orifilter_num != currentfilter_num:
                    logger.info('loading ci from: ' + prefix + str(cov_id) + subfix)
                    ci = np.load(prefix + str(cov_id) + subfix)
                    select_index = np.argsort(ci)[orifilter_num - currentfilter_num:]  # preserved filter id
                    select_index.sort()

                    if last_select_index is not None:
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[name_base+conv_weight_name][index_i] = \
                                oristate_dict[conv_weight_name][i]

                    last_select_index = select_index

                elif last_select_index is not None:
                    for index_i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                oristate_dict[conv_weight_name][index_i][j]
                    last_select_index = None

                else:
                    state_dict[name_base+conv_weight_name] = oriweight
                    last_select_index = None

    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if 'shortcut' in name:
                continue
            if conv_name not in all_conv_weight:
                state_dict[name_base+conv_name] = oristate_dict[conv_name]

        elif isinstance(module, nn.Linear):
            state_dict[name_base+name + '.weight'] = oristate_dict[name + '.weight']
            state_dict[name_base+name + '.bias'] = oristate_dict[name + '.bias']

    model.load_state_dict(state_dict)

def adjust_learning_rate(optimizer, epoch, step, len_iter):

    if args.lr_type == 'step':
        factor = epoch // 125
        # if epoch >= 80:
        #     factor = factor + 1
        lr = args.learning_rate * (0.1 ** factor)

    elif args.lr_type == 'step_5':
        factor = epoch // 10
        if epoch >= 80:
            factor = factor + 1
        lr = args.learning_rate * (0.5 ** factor)

    elif args.lr_type == 'cos':  # cos without warm-up
        lr = 0.5 * args.learning_rate * (1 + math.cos(math.pi * (epoch - 5) / (args.epochs - 5)))

    elif args.lr_type == 'exp':
        step = 1
        decay = 0.96
        lr = args.learning_rate * (decay ** (epoch // step))

    elif args.lr_type == 'fixed':
        lr = args.learning_rate
    else:
        raise NotImplementedError

    #Warmup
    if epoch < 5:
        lr = lr * float(1 + step + epoch * len_iter) / (5. * len_iter)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if step == 0:
        logger.info('learning_rate: ' + str(lr))

def main():

    cudnn.benchmark = True
    cudnn.enabled=True
    logger.info("args = %s", args)

    if args.sparsity:
        import re
        cprate_str = args.sparsity
        cprate_str_list = cprate_str.split('+')
        pat_cprate = re.compile(r'\d+\.\d*')
        pat_num = re.compile(r'\*\d+')
        cprate = []
        for x in cprate_str_list:
            num = 1
            find_num = re.findall(pat_num, x)
            if find_num:
                assert len(find_num) == 1
                num = int(find_num[0].replace('*', ''))
            find_cprate = re.findall(pat_cprate, x)
            assert len(find_cprate) == 1
            cprate += [float(find_cprate[0])] * num

        sparsity = cprate

    # load model
    logger.info('sparsity:' + str(sparsity))
    logger.info('==> Building model..')
    model = eval(args.arch)(sparsity=sparsity).cuda()
    logger.info(model)

    #calculate model size
    input_image_size=32
    input_image = torch.randn(1, 3, input_image_size, input_image_size).cuda()
    flops, params = profile(model, inputs=(input_image,))
    logger.info('Params: %.2f' % (params))
    logger.info('Flops: %.2f' % (flops))

    # load training data
    train_loader, val_loader = cifar10.load_cifar_data(args)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    criterion_smooth = utils.CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
    criterion_smooth = criterion_smooth.cuda()

    if len(args.gpu) > 1:
        device_id = []
        for i in range((len(args.gpu) + 1) // 2):
            device_id.append(i)
        model = nn.DataParallel(model, device_ids=device_id).cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_decay_step = list(map(int, args.lr_decay_step.split(',')))
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)

    start_epoch = 0
    best_top1_acc= 0

    # load the checkpoint if it exists
    checkpoint_dir = os.path.join(args.result_dir, 'checkpoint.pth.tar')

    logger.info('resuming from pretrain model')
    origin_model = eval(args.arch)(sparsity=[0.] * 100).cuda()
    ckpt = torch.load(args.pretrain_dir, map_location='cuda:0')

    if args.arch == 'resnet_110':
        new_state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            new_state_dict[k.replace('module.', '')] = v
        origin_model.load_state_dict(new_state_dict)
    else:
        origin_model.load_state_dict(ckpt['state_dict'])

    oristate_dict = origin_model.state_dict()

    if args.arch == 'vgg_16_bn':
        load_vgg_model(model, oristate_dict)
    elif args.arch == 'resnet_56':
        load_resnet_model(model, oristate_dict, 56)
    elif args.arch == 'resnet_110':
        load_resnet_model(model, oristate_dict, 110)
    else:
        raise

    # adjust the learning rate according to the checkpoint
    # for epoch in range(start_epoch):
    #     scheduler.step()

    # train the model
    epoch = start_epoch
    while epoch < args.epochs:
        if args.label_smooth > 0:
            train_obj, train_top1_acc, train_top5_acc = train(epoch, train_loader, model, criterion_smooth,
                                                              optimizer)  # , scheduler)
        else:
            train_obj, train_top1_acc,  train_top5_acc = train(epoch,  train_loader, model, criterion, optimizer)#, scheduler)
        valid_obj, valid_top1_acc, valid_top5_acc = validate(epoch, val_loader, model, criterion, args)

        is_best = False
        if valid_top1_acc > best_top1_acc:
            best_top1_acc = valid_top1_acc
            is_best = True

        utils.save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_top1_acc': best_top1_acc,
            'optimizer' : optimizer.state_dict(),
            }, is_best, args.result_dir)

        epoch += 1
        logger.info("=>Best accuracy {:.3f}".format(best_top1_acc))#


def train(epoch, train_loader, model, criterion, optimizer, scheduler = None):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')

    model.train()
    end = time.time()

    for param_group in optimizer.param_groups:
        cur_lr = param_group['lr']
    logger.info('learning_rate: ' + str(cur_lr))

    num_iter = len(train_loader)
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.cuda()
        target = target.cuda()

        adjust_learning_rate(optimizer, epoch, i, num_iter)

        # compute outputy
        logits = model(images)
        loss = criterion(logits, target)

        # measure accuracy and record loss
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = images.size(0)
        losses.update(loss.item(), n)   #accumulated loss
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            logger.info(
                'Epoch[{0}]({1}/{2}): '
                'Loss {loss.avg:.4f} '
                'Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f}'.format(
                    epoch, i, num_iter, loss=losses,
                    top1=top1, top5=top5))

    # scheduler.step()

    return losses.avg, top1.avg, top5.avg

def validate(epoch, val_loader, model, criterion, args):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')

    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()

            # compute output
            logits = model(images)
            loss = criterion(logits, target)

            # measure accuracy and record loss
            pred1, pred5 = utils.accuracy(logits, target, topk=(1, 5))
            n = images.size(0)
            losses.update(loss.item(), n)
            top1.update(pred1[0], n)
            top5.update(pred5[0], n)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        logger.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


if __name__ == '__main__':
  main()
