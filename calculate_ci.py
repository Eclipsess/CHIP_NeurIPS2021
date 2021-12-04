import numpy as np
import torch
import os

def nuc_score(path_conv, path_nuc):
    conv_output = np.round(np.load(path_conv), 4)
    conv_reshape = conv_output.reshape(conv_output.shape[0], conv_output.shape[1], -1)
    # original_norm = torch.norm(torch.tensor(conv_reshape[0, :, :]), p='nuc').item()

    nuc = np.round(np.load(path_nuc), 4)
    # s = np.zeros_like(nuc)
    nuc_importance = np.zeros_like(nuc)

    for i in range(nuc.shape[0]):
        original_norm = torch.norm(torch.tensor(conv_reshape[i, :, :]), p='nuc').item()
        nuc_importance[i] = original_norm - nuc[i]

    # shape [batch_size, filter_number]
    return nuc_importance

def mean_repeat_nuc_score(repeat, num_layers, base_path_conv=None, base_path_nuc=None):
    # nuc_score_mean = []
    layer_mean_total = []
    for j in range(num_layers):
        repeat_mean = []
        for i in range(repeat):
            index = j * repeat + i + 1
            # add
            path_conv = "./feature_conv/resnet_56_repeat5/feature_convtensor({0}).npy".format(str(index))
            path_nuc = "./feature_conv_nuc/resnet_56_repeat5/feature_conv_nuctensor({0}).npy".format(str(index))
            nuc_s = nuc_score(path_conv, path_nuc)
            nuc_s_m = np.mean(nuc_s, axis=0)
            repeat_mean.append(nuc_s_m)

        layer_mean = np.mean(repeat_mean, axis=0)
        layer_mean_total.append(layer_mean)

    return np.array(layer_mean_total)

def main():
    repeat = 5
    num_layers = 55
    save_path = './nuc_rank_resnet56'
    ci = mean_repeat_nuc_score(repeat, num_layers)
    for i in range(num_layers):
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        np.save(save_path + "/nuc_conv{0}.npy".format(str(i + 1)), ci[i])

if __name__ == '__main__':
  main()



