import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from data import cifar10, imagenet
import time
from models.resnet_cifar10 import resnet_56,resnet_110
from models.resnet_imagenet import resnet_50


parser = argparse.ArgumentParser(description='Calculate Feature Maps')

parser.add_argument(
    '--arch',
    type=str,
    default='resnet_56',
    choices=('vgg_16_bn','resnet_56','resnet_110','resnet_50'),
    help='architecture to calculate feature maps')

parser.add_argument(
    '--dataset',
    type=str,
    default='cifar10',
    choices=('cifar10','imagenet'),
    help='cifar10 or imagenet')

parser.add_argument(
    '--data_dir',
    type=str,
    default='./data',
    help='dataset path')

parser.add_argument(
    '--pretrain_dir',
    type=str,
    default=None,
    help='dir for the pretriained model to calculate feature maps')

parser.add_argument(
    '--batch_size',
    type=int,
    default=128,
    help='batch size for one batch.')

parser.add_argument(
    '--repeat',
    type=int,
    default=5,
    help='the number of different batches for calculating feature maps.')

parser.add_argument(
    '--gpu',
    type=str,
    default='0',
    help='gpu id')

args = parser.parse_args()

# gpu setting
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# prepare data
if args.dataset=='cifar10':
    train_loader, _ = cifar10.load_cifar_data(args)
elif args.dataset=='imagenet':
    data_tmp = imagenet.Data(args)
    train_loader = data_tmp.train_loader

# Model
model = eval(args.arch)(sparsity=[0.]*100).to(device)

# Load pretrained model.
print('Loading Pretrained Model...')
if args.arch=='vgg_16_bn' or args.arch=='resnet_56':
    checkpoint = torch.load(args.pretrain_dir, map_location='cuda:'+args.gpu)
else:
    checkpoint = torch.load(args.pretrain_dir)
if args.arch=='resnet_50':
    model.load_state_dict(checkpoint)
else:
    model.load_state_dict(checkpoint['state_dict'])

conv_index = torch.tensor(1)

def get_feature_hook(self, input, output):
    global conv_index

    if not os.path.isdir('conv_feature_map/' + args.arch + '_repeat%d' % (args.repeat)):
        os.makedirs('conv_feature_map/' + args.arch + '_repeat%d' % (args.repeat))
    np.save('conv_feature_map/' + args.arch + '_repeat%d' % (args.repeat) + '/conv_feature_map_'+ str(conv_index) + '.npy',
            output.cpu().numpy())
    conv_index += 1

def inference():
    model.eval()
    repeat = args.repeat
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            #use 5 batches to get feature maps.
            if batch_idx >= repeat:
               break

            inputs, targets = inputs.to(device), targets.to(device)

            model(inputs)

if args.arch=='vgg_16_bn':

    if len(args.gpu) > 1:
        relucfg = model.module.relucfg
    else:
        relucfg = model.relucfg
    start = time.time()
    for i, cov_id in enumerate(relucfg):
        cov_layer = model.features[cov_id]
        handler = cov_layer.register_forward_hook(get_feature_hook)
        inference()
        handler.remove()

elif args.arch=='resnet_56':

    cov_layer = eval('model.relu')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    # ResNet56 per block
    cnt=1
    for i in range(3):
        block = eval('model.layer%d' % (i + 1))
        for j in range(9):
            cov_layer = block[j].relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt+=1

            cov_layer = block[j].relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt += 1

elif args.arch=='resnet_110':

    cov_layer = eval('model.relu')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    cnt = 1
    # ResNet110 per block
    for i in range(3):
        block = eval('model.layer%d' % (i + 1))
        for j in range(18):
            cov_layer = block[j].relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt += 1
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)

            cov_layer = block[j].relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            cnt += 1

elif args.arch=='resnet_50':
    cov_layer = eval('model.maxpool')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    # ResNet50 per bottleneck
    for i in range(4):
        block = eval('model.layer%d' % (i + 1))
        for j in range(model.num_blocks[i]):
            cov_layer = block[j].relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()

            cov_layer = block[j].relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()

            cov_layer = block[j].relu3
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()

            if j==0:
                cov_layer = block[j].relu3
                handler = cov_layer.register_forward_hook(get_feature_hook)
                inference()
                handler.remove()
