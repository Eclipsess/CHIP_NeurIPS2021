# CHIP: CHannel Independence-based Pruning for Compact Neural Networks

## Code for our NeurIPS 2021 paper: ([CHIP: CHannel Independence-based Pruning for Compact Neural Networks](https://arxiv.org/abs/2110.13981))![visitors](https://visitor-badge.glitch.me/badge?page_id=yangsui.chip&left_color=green&right_color=red).

<p align="center">
<img src="fig/algorithm.png" width="800">
</p>

## Usage

## Citation
```
@article{sui2021chip,
  title={CHIP: CHannel Independence-based Pruning for Compact Neural Networks},
  author={Sui, Yang and Yin, Miao and Xie, Yi and Phan, Huy and Aliari Zonouz, Saman and Yuan, Bo},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```

### Generate Feature Maps.

##### 1. CIFAR-10
```shell
python calculate_feature_maps.py \
--arch resnet_56 \
--dataset cifar10 \
--data_dir ./data \
--pretrain_dir ./pretrained_models/resnet_56.pt \
--gpu 0
```
##### 2. ImageNet
```shell
python calculate_feature_maps.py \
--arch resnet_50 \
--dataset imagenet \
--data_dir /raid/data/imagenet \
--pretrain_dir ./pretrained_models/resnet50.pth \
--gpu 0
```
### Generate Channel Independence (CI).

This procedure is time-consuming, please be patient.

##### 1. CIFAR-10
```shell
python calculate_ci.py \
--arch resnet_56 \
--repeat 5 \
--num_layers 55
```
##### 2. ImageNet
```shell
python calculate_ci.py \
--arch resnet_50 \
--repeat 5 \
--num_layers 53
```

### Prune and Fine-tune Models.

##### 1. CIFAR-10
```shell
python prune_finetune_cifar.py \
--data_dir ./data \
--result_dir ./result/resnet_56/1 \
--arch resnet_56 \
--ci_dir ./CI_resnet_56 \
--batch_size 256 \
--epochs 200 \
--lr_type cos \
--learning_rate 0.01 \
--momentum 0.99 \
--weight_decay 0.001 \
--pretrain_dir ./pretrained_models/resnet_56.pt \
--sparsity [0.]+[0.4]*2+[0.5]*9+[0.6]*9+[0.7]*9 \
--gpu 0 
```
##### 2. ImageNet
```shell
python prune_finetune_imagenet.py \
--data_dir /raid/data/imagenet \
--result_dir ./result/resnet_50/1 \
--arch resnet_50 \
--ci_dir ./CI_resnet_50 \
--batch_size 256 \
--epochs 200 \
--lr_type cos \
--learning_rate 0.01 \
--momentum 0.99 \
--label_smooth 0.1 \
--weight_decay 0.0001 \
--pretrain_dir ./pretrained_models/resnet50.pth \
--sparsity [0.]+[0.1]*3+[0.35]*16 \
--gpu 0
```

## Pre-trained Models

- [Pre-trained Models](https://drive.google.com/drive/folders/1b--dZlvKUUu0rXqMYAtIr0ynHQHuEWDI?usp=sharing)
   - CIFAR-10: VGG-16_BN, ResNet-56, ResNet-110.
   - ImageNet: ResNet-50.
   
### Results

We release our training logs of ResNet-56/110 model on CIFAR-10 for more epochs which can achieve better results than paper. 
We release our training logs of ResNet-50 model on ImageNet. 
Training logs can be found at [link](https://drive.google.com/drive/folders/1qhHxu2AIayUBejbuHuEoGt1-i32DTJRv?usp=sharing).
Some results are better than papers.

##### CIFAR-10

| Model <img width=60/>  | # of Params (Reduction)      | Flops  (Reduction)        |  Top-1 Accuracy | Sparsity Setting                                           |
|:-------------------:|:---:|:--------------:|:--------:|:------------------------------------------------------------:|
| ResNet-56   | 0.85M(0.0%) | 125.49M(0.0%) |  93.26%   |  N/A |
| ResNet-56   | 0.48M(42.8%) | 65.94M(47.4%) | 94.16%   | [0.]+[0.15]*2+[0.4]*27 |
| ResNet-56   | 0.24M(70.0%) | 34.78M(74.1%) | 92.43%  | [0.]+[0.4]*2+[0.5]*9+[0.6]*9+[0.7]*9 |
| ResNet-110   | 1.72M(0.0%) | 252.89M(0.0%) |  93.50%   | N/A |
| ResNet-110   | 1.04M(39.1%) | 140.54M(44.4%) |  94.50%  | [0.]+[0.2]*2+[0.3]*18+[0.35]*36 |
| ResNet-110   | 0.89M(48.3%) | 121.09M(52.1%) |  94.44%  | [0.]+[0.22]*2+[0.35]*18+[0.45]*36 |
| ResNet-110   | 0.54M(68.3%) | 71.69M(71.6%) |  93.23%  | [0.]+[0.4]*2+[0.5]*18+[0.65]*36 |
| VGG-16-BN    | 14.98M(0.0%) | 313.73M(0.0%) | 93.96%   | N/A |
| VGG-16-BN      | 2.76M(81.6%) | 131.17M(58.1%) | 93.86%   | [0.21]*7+[0.75]*5 |
| VGG-16-BN      | 2.50M(83.3%) | 104.78M(66.6%) | 93.72%   | [0.3]*7+[0.75]*5 |
| VGG-16-BN      | 1.90M(87.3%) | 66.95M(78.6%) | 93.18%    | [0.45]*7+[0.78]*5 |


##### ImageNet 
| Model <img width=60/> | # of Params  (Reduction)      | Flops (Reduction)     | Top-1 Accuracy | Top-5 Accuracy | Sparsity Setting |
|:----------:|:-------------:|:--------------:|:------------------:|:----------------------------:|:---:|
| ResNet-50  |       25.55M(0.0%)          |      4.11B(0.0%)      |   76.15%      |       92.87%         | N/A |
| ResNet-50  |       15.09M(40.8%)          |      2.26B(44.8%)       |  76.41%      |       93.06%         | [0.]+[0.1]*3+[0.35]*16 |
| ResNet-50  |       14.28M(44.2%)          |      2.19B(48.7%)       |   76.35%      |       93.05%         | [0.]+[0.12]*3+[0.38]*16 |
| ResNet-50  |       11.05M(56.7%)          |      1.52B(62.8%)      |   75.26%    | 92.53% | [0.]+[0.25]*3+[0.5]*16 |
| ResNet-50  |       8.02M(68.6%)          |      0.95B(76.7%)       | 73.30%      |      91.48%         | [0.]+[0.5]*3+[0.6]*16|


## Others
Codes are based on [link](https://github.com/lmbxmu/HRankPlus).

Since I rearranged my original codes for simplicity, please feel free to open an issue if something wrong happens when you run the codes. (Please forgive me for the late response and wait for me to respond to your problems in several days.)

