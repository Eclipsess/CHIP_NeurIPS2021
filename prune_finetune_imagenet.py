import os
import numpy as np
import time, datetime
import torch
import argparse
import math
import shutil
from collections import OrderedDict
from thop import profile

import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
from models.resnet_imagenet import resnet_50

from data import imagenet
import utils
from torch.cuda.amp import autocast, GradScaler

parser = argparse.ArgumentParser("ImageNet training")

parser.add_argument(
    '--data_dir',
    type=str,
    default='',
    help='path to dataset')

parser.add_argument(
    '--arch',
    type=str,
    default='resnet_56',
    help='architecture')

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
    '--learning_rate',
    type=float,
    default=0.1,
    help='init learning rate')

'''parser.add_argument(
    '--lr_decay_step',
    default='30,60',
    type=str,
    help='learning rate decay step')'''

parser.add_argument(
    '--lr_type',
    default='cos',
    type=str,
    help='learning rate decay schedule')

parser.add_argument(
    '--momentum',
    type=float,
    default=0.9,
    help='momentum')

parser.add_argument(
    '--weight_decay',
    type=float,
    default=1e-4,
    help='weight decay')

parser.add_argument(
    '--label_smooth',
    type=float,
    default=0.1,
    help='label smoothing')

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
    help='compress rate of each conv')

parser.add_argument(
    '--gpu',
    type=str,
    default='0',
    help='gpu id')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

CLASSES = 1000
print_freq = 128000//args.batch_size

if not os.path.isdir(args.result_dir):
    os.makedirs(args.result_dir)

#save old training file
now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
cp_file_dir = os.path.join(args.result_dir, 'cp_file/' + now)
if os.path.exists(args.result_dir+'/model_best.pth.tar'):
    if not os.path.isdir(cp_file_dir):
        os.makedirs(cp_file_dir)
    shutil.copy(args.result_dir+'/config.txt', cp_file_dir)
    shutil.copy(args.result_dir+'/logger.log', cp_file_dir)
    shutil.copy(args.result_dir+'/model_best.pth.tar', cp_file_dir)
    shutil.copy(args.result_dir + '/checkpoint.pth.tar', cp_file_dir)

utils.record_config(args)
logger = utils.get_logger(os.path.join(args.result_dir, 'logger.log'))

#use for loading pretrain model
if len(args.gpu)>1:
    name_base='module.'
else:
    name_base=''

def load_resnet_model(model, oristate_dict):
    cfg = {'resnet_50': [3, 4, 6, 3],}

    state_dict = model.state_dict()

    current_cfg = cfg[args.arch]
    last_select_index = None

    all_honey_conv_weight = []

    bn_part_name=['.weight','.bias','.running_mean','.running_var']#,'.num_batches_tracked']
    prefix = args.ci_dir+'/ci_conv'
    subfix = ".npy"
    cnt=1

    conv_weight_name = 'conv1.weight'
    all_honey_conv_weight.append(conv_weight_name)
    oriweight = oristate_dict[conv_weight_name]
    curweight = state_dict[name_base+conv_weight_name]
    orifilter_num = oriweight.size(0)
    currentfilter_num = curweight.size(0)

    if orifilter_num != currentfilter_num:
        logger.info('loading ci from: ' + prefix + str(cnt) + subfix)
        ci = np.load(prefix + str(cnt) + subfix)
        select_index = np.argsort(ci)[orifilter_num - currentfilter_num:]  # preserved filter id
        select_index.sort()

        for index_i, i in enumerate(select_index):
            state_dict[name_base+conv_weight_name][index_i] = \
                oristate_dict[conv_weight_name][i]
            for bn_part in bn_part_name:
                state_dict[name_base + 'bn1' + bn_part][index_i] = \
                    oristate_dict['bn1' + bn_part][i]

        last_select_index = select_index
    else:
        state_dict[name_base + conv_weight_name] = oriweight
        for bn_part in bn_part_name:
            state_dict[name_base + 'bn1' + bn_part] = oristate_dict['bn1'+bn_part]

    state_dict[name_base + 'bn1' + '.num_batches_tracked'] = oristate_dict['bn1' + '.num_batches_tracked']

    cnt+=1
    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'

        for k in range(num):
            iter = 3
            if k==0:
                iter +=1
            for l in range(iter):
                record_last=True
                if k==0 and l==2:
                    conv_name = layer_name + str(k) + '.downsample.0'
                    bn_name = layer_name + str(k) + '.downsample.1'
                    record_last=False
                elif k==0 and l==3:
                    conv_name = layer_name + str(k) + '.conv' + str(l)
                    bn_name = layer_name + str(k) + '.bn' + str(l)
                else:
                    conv_name = layer_name + str(k) + '.conv' + str(l + 1)
                    bn_name = layer_name + str(k) + '.bn' + str(l + 1)

                conv_weight_name = conv_name + '.weight'
                all_honey_conv_weight.append(conv_weight_name)
                oriweight = oristate_dict[conv_weight_name]
                curweight = state_dict[name_base+conv_weight_name]
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                if orifilter_num != currentfilter_num:
                    logger.info('loading ci from: ' + prefix + str(cnt) + subfix)
                    ci = np.load(prefix + str(cnt) + subfix)
                    select_index = np.argsort(ci)[orifilter_num - currentfilter_num:]  # preserved filter id
                    select_index.sort()

                    if last_select_index is not None:
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]


                            for bn_part in bn_part_name:
                                state_dict[name_base + bn_name + bn_part][index_i] = \
                                    oristate_dict[bn_name + bn_part][i]

                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[name_base+conv_weight_name][index_i] = \
                                oristate_dict[conv_weight_name][i]

                            for bn_part in bn_part_name:
                                state_dict[name_base + bn_name + bn_part][index_i] = \
                                    oristate_dict[bn_name + bn_part][i]

                    if record_last:
                        last_select_index = select_index

                elif last_select_index is not None:
                    for index_i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                oristate_dict[conv_weight_name][index_i][j]

                    for bn_part in bn_part_name:
                        state_dict[name_base + bn_name + bn_part] = \
                            oristate_dict[bn_name + bn_part]

                    if record_last:
                        last_select_index = None

                else:
                    state_dict[name_base+conv_weight_name] = oriweight
                    for bn_part in bn_part_name:
                        state_dict[name_base + bn_name + bn_part] = \
                            oristate_dict[bn_name + bn_part]
                    if record_last:
                        last_select_index = None

                state_dict[name_base + bn_name + '.num_batches_tracked'] = oristate_dict[bn_name + '.num_batches_tracked']
                cnt+=1

    for name, module in model.named_modules():
        name = name.replace('module.', '')
        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if conv_name not in all_honey_conv_weight:
                state_dict[name_base+conv_name] = oristate_dict[conv_name]

        elif isinstance(module, nn.Linear):
            state_dict[name_base+name + '.weight'] = oristate_dict[name + '.weight']
            state_dict[name_base+name + '.bias'] = oristate_dict[name + '.bias']

    model.load_state_dict(state_dict)

def adjust_learning_rate(optimizer, epoch, step, len_iter):

    if args.lr_type == 'step':
        factor = epoch // 30
        if epoch >= 80:
            factor = factor + 1
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

    start_t = time.time()

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

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    criterion_smooth = utils.CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
    criterion_smooth = criterion_smooth.cuda()

    # load training data
    print('==> Preparing data..')

    data_tmp = imagenet.Data(args)
    train_loader = data_tmp.train_loader
    val_loader = data_tmp.test_loader

    # calculate model size
    input_image_size = 224
    input_image = torch.randn(1, 3, input_image_size, input_image_size).cuda()
    flops, params = profile(model, inputs=(input_image,))
    logger.info('Params: %.2f' % (params))
    logger.info('Flops: %.2f' % (flops))

    if len(args.gpu) > 1:
        device_id = []
        for i in range((len(args.gpu) + 1) // 2):
            device_id.append(i)
        model = nn.DataParallel(model, device_ids=device_id).cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    '''# define the learning rate scheduler
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step : (1.0-step/args.epochs), last_epoch=-1)
    if args.lr_type=='multi_step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.epochs//4, args.epochs//2, args.epochs//4*3], gamma=0.1)
    elif args.lr_type=='cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=100, eta_min=0.0004)#'''
    start_epoch = 0
    best_top1_acc= 0
    best_top5_acc= 0

    # load the checkpoint if it exists
    checkpoint_dir = os.path.join(args.result_dir, 'checkpoint.pth.tar')

    logger.info('resuming from pretrain model')
    origin_model = eval(args.arch)(sparsity=[0.] * 100).cuda()
    ckpt = torch.load(args.pretrain_dir)
    origin_model.load_state_dict(ckpt)
    oristate_dict = origin_model.state_dict()
    if args.arch == 'resnet_50':
        load_resnet_model(model, oristate_dict)
    else:
        raise

    # train the model
    scaler = GradScaler()
    epoch = start_epoch
    while epoch < args.epochs:

        train_obj, train_top1_acc,  train_top5_acc = train(epoch,  train_loader, model, criterion_smooth, optimizer, scaler)
        valid_obj, valid_top1_acc, valid_top5_acc = validate(epoch, val_loader, model, criterion, args)

        is_best = False
        if valid_top1_acc > best_top1_acc:
            best_top1_acc = valid_top1_acc
            best_top5_acc = valid_top5_acc
            is_best = True

        utils.save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_top1_acc': best_top1_acc,
            'best_top5_acc': best_top5_acc,
            'optimizer' : optimizer.state_dict(),
            }, is_best, args.result_dir)

        epoch += 1
        logger.info("=>Best accuracy Top1: {:.3f}, Top5: {:.3f}".format(best_top1_acc, best_top5_acc))

    training_time = (time.time() - start_t) / 36000
    logger.info('total training time = {} hours'.format(training_time))


def train(epoch, train_loader, model, criterion, optimizer, scaler = None):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')

    model.train()
    end = time.time()
    #scheduler.step()

    num_iter = len(train_loader)

    print_freq = num_iter // 10

    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.cuda()
        targets = targets.cuda()
        data_time.update(time.time() - end)

        adjust_learning_rate(optimizer, epoch, batch_idx, num_iter)

        # compute output
        logits = model(images)
        loss = criterion(logits, targets)

        # measure accuracy and record loss
        prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
        n = images.size(0)
        losses.update(loss.item(), n)  # accumulated loss
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % print_freq == 0 and batch_idx != 0:
            logger.info(
                'Epoch[{0}]({1}/{2}): '
                'Loss {loss.avg:.4f} '
                'Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f}'.format(
                    epoch, batch_idx, num_iter, loss=losses,
                    top1=top1, top5=top5))
    return losses.avg, top1.avg, top5.avg

def validate(epoch, val_loader, model, criterion, args):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')

    model.eval()
    with torch.no_grad():
        end = time.time()
        for batch_idx, (images, targets) in enumerate(val_loader):
            images = images.cuda()
            targets = targets.cuda()

            # compute output
            logits = model(images)
            loss = criterion(logits, targets)

            # measure accuracy and record loss
            pred1, pred5 = utils.accuracy(logits, targets, topk=(1, 5))
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
