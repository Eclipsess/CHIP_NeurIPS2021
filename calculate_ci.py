import numpy as np
import torch
import os
import argparse

parser = argparse.ArgumentParser(description='Calculate CI')

parser.add_argument(
    '--arch',
    type=str,
    default='resnet_56',
    choices=('vgg_16_bn','resnet_56','resnet_110','resnet_50'),
    help='architecture to calculate feature maps')

parser.add_argument(
    '--repeat',
    type=int,
    default=5,
    help='repeat times')

parser.add_argument(
    '--num_layers',
    type=int,
    default=55,
    help='conv layers in the model')

parser.add_argument(
    '--feature_map_dir',
    type=str,
    default='./conv_feature_map',
    help='feature maps dir')

args = parser.parse_args()

def reduced_1_row_norm(input, row_index, data_index):
    input[data_index, row_index, :] = torch.zeros(input.shape[-1])
    m = torch.norm(input[data_index, :, :], p = 'nuc').item()
    return m

def ci_score(path_conv):
    conv_output = torch.tensor(np.round(np.load(path_conv), 4))
    conv_reshape = conv_output.reshape(conv_output.shape[0], conv_output.shape[1], -1)

    r1_norm = torch.zeros([conv_reshape.shape[0], conv_reshape.shape[1]])
    for i in range(conv_reshape.shape[0]):
        for j in range(conv_reshape.shape[1]):
            r1_norm[i, j] = reduced_1_row_norm(conv_reshape.clone(), j, data_index = i)

    ci = np.zeros_like(r1_norm)

    for i in range(r1_norm.shape[0]):
        original_norm = torch.norm(torch.tensor(conv_reshape[i, :, :]), p='nuc').item()
        ci[i] = original_norm - r1_norm[i]

    # return shape: [batch_size, filter_number]
    return ci

def mean_repeat_ci(repeat, num_layers):
    layer_ci_mean_total = []
    for j in range(num_layers):
        repeat_ci_mean = []
        for i in range(repeat):
            index = j * repeat + i + 1
            # add
            path_conv = "./conv_feature_map/{0}_repeat5/conv_feature_map_tensor({1}).npy".format(str(args.arch), str(index))
            # path_nuc = "./feature_conv_nuc/resnet_56_repeat5/feature_conv_nuctensor({0}).npy".format(str(index))
            # batch_ci = ci_score(path_conv, path_nuc)
            batch_ci = ci_score(path_conv)
            single_repeat_ci_mean = np.mean(batch_ci, axis=0)
            repeat_ci_mean.append(single_repeat_ci_mean)

        layer_ci_mean = np.mean(repeat_ci_mean, axis=0)
        layer_ci_mean_total.append(layer_ci_mean)

    return np.array(layer_ci_mean_total)

def main():
    repeat = args.repeat
    num_layers = args.num_layers
    save_path = 'CI_' + args.arch
    ci = mean_repeat_ci(repeat, num_layers)
    if args.arch == 'resnet_50':
        num_layers = 53
    for i in range(num_layers):
        print(i)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        np.save(save_path + "/ci_conv{0}.npy".format(str(i + 1)), ci[i])

if __name__ == '__main__':
    main()



