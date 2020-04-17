from __future__ import print_function
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data.predict_data import SignalPrediction
from ResNet_1D import build_SSD
from eval_signals import eval_signals

def draw_pred(net, cuda, testdataset, labelmap, threshold):
    num_seqs = len(testdataset)
    for idx in range(num_seqs):
        print('plot for seq.{}'.format(idx))
        seq = testdataset.pull_seq(idx)
        # seq_id, annos = testdataset.pull_anno(idx)

        x = torch.from_numpy(seq).type(torch.FloatTensor)
        x = Variable(x.unsqueeze(0))

        if cuda:
            x = x.cuda()

        y = net(x)
        detection = y.data
        scale = torch.Tensor([1.0261e+05, 1.0261e+05])
        coords_1 = []
        coords_2 = []
        score_all = []
        for i in range(detection.size(1)):
            if i == 1:
                j = 0
                while detection[0, i, j, 0] >= threshold:
                    score = detection[0, i, j, 0].numpy()
                    score_all.append(score)
                    pt = (detection[0, i, j, 1:] * scale).cpu().numpy()
                    coordination = (pt[0], pt[1])
                    coords_1.append(coordination)
                    j += 1
            elif i == 2:
                j = 0
                while detection[0, i, j, 0] >= threshold:
                    score = detection[0, i, j, 0].numpy()
                    pt = (detection[0, i, j, 1:] * scale).cpu().numpy()
                    coordination = (pt[0], pt[1])
                    coords_2.append(coordination)
                    j += 1
        plot_spec(seq, coords_1, coords_2, score_all, idx)

def plot_spec(seqs, coords_1, coords_2, score_all, idx):
    scale = 1.0261e+05
#    start = scale / 2
    start = 0
    seqs = seqs.squeeze(0)
    # scale = 1e5
    gap = scale / len(seqs)
    plt.figure()
    x = np.linspace(scale * idx + start, scale * (idx + 1)+ start, len(seqs + 1))
#    x = np.linspace(scale * idx, scale * (idx + 1) , len(seqs + 1))
    plt.plot(x, seqs, c='blue', linewidth=0.5)
    for i in range(len(coords_1)):
        pt0 = int(coords_1[i][0] / gap)
        pt1 = int(coords_1[i][1] / gap)
#        if pt1 - pt0 > 1200 or pt1 - pt0 < 100:
#            continue
        value  = (np.mean(seqs[pt0 : pt1])) * np.ones(pt1 - pt0 + 1)
        x1 = np.array([ i * gap + scale * idx + start for i in range(pt0, pt1 + 1)])
#        x1 = np.array([i * gap + scale * idx for i in range(pt0, pt1 + 1)])
        plt.plot(x1, value, color='red', linewidth=1)

        plt.annotate(str(score_all[i]), xy=(pt0 * gap + scale * idx + start, max(value)), color='green', fontsize=15)
    for i in range(len(coords_2)):
        pt0 = int(coords_2[i][0] / gap)
        pt1 = int(coords_2[i][1] / gap)
#        if pt1 - pt0 > 1200 or pt1 - pt0 < 100:
#            continue
        value_true  = np.mean(seqs[pt0 : pt1]) * np.ones(pt1 - pt0 + 1)
        x2 = np.array([ i * gap + scale * idx + start for i in range(pt0, pt1 + 1)])
#        x2 = np.array([i * gap + scale * idx for i in range(pt0, pt1 + 1)])
        plt.plot(x2, value_true, color='green', linewidth=1)
    plt.xlabel('Hz')
    plt.ylabel('Spec')
    plt.title('prediction for seq {}'.format(idx))
    # plt.xlim(scale * idx + start, scale * (idx + 1) + start)
    plt.xlim(scale * idx, scale * (idx + 1))
    plt.ylim(0, 1)
    plt.savefig('./fig_realdata/seq_{}.png'.format(idx), format='png', transparent=True, dpi=300, pad_inches=0)
    # plt.show()

def test_net(save_folder, txtname, net, cuda, testset, labelmap, threshold):
    filename = save_folder + txtname
    num_seqs = len(testset)
    for idx in tqdm(range(num_seqs)):
        print('Testing seqs: {:d}/{:d} ... '.format(idx+1, num_seqs))
        seq = testset.pull_seq(idx)

        x = torch.from_numpy(seq).type(torch.FloatTensor)
        x = Variable(x.unsqueeze(0))

        with open(filename, mode='a') as f:
            f.write('\nPredict for index ' + str(idx) + '\n')
        if cuda:
            x = x.cuda()
        # forward pass
        y = net(x)
        detection = y.data
        # scale each detection back up to the seq
        scale = torch.Tensor([1.0261e+05, 1.0261e+05])

        pre_num = 0
        for i in range(detection.size(1)):
            j = 0
            while  detection[0, i, j, 0] >= threshold:
                if pre_num == 0:
                    with open(filename, mode='a') as f:
                        f.write('Prediction: ' + '\n')
                score = detection[0, i, j, 0].numpy()
                label_name = labelmap[i - 1]
                pt = (detection[0, i, j, 1:] * scale + scale * idx).cpu().numpy()
                coordination = (pt[0], pt[1])
                pre_num += 1
                with open(filename, mode='a') as f:
                    print(score)
                    f.writelines(str(pre_num) + ' label: ' + label_name + ' scores: ' + str(score) + ' ' + '||'.join(str(c) for c in coordination) + '\n')
                j += 1

def test_signals(cfg):
    num_classes = 2
    net = build_SSD('test', 8192, num_classes, cfg)
    net.load_state_dict(torch.load(cfg['trained_model'], map_location='cpu'))
    net.eval()
    print('model loaded!')
    #load data
    testset = SignalPrediction(cfg['test_data_root'])
    if cfg['using_gpu']:
        net.cuda()

    #evaluation
#    test_net(cfg['saved_folder'], cfg['txt_name'], net, cfg['using_gpu'], testset,
#             cfg['labelmap'], cfg['visual_threshold'])
    #plot
    draw_pred(net, cfg['using_gpu'], testset, cfg['labelmap'], cfg['visual_threshold'])

if __name__ == '__main__':

    settings = {
        'feature_maps': [256, 128, 64, 32, 16],
        'min_dim': 8192,
        'steps': [32, 64, 128, 256, 512],
        # 'num_scales': [5, 5, 5, 15, 15],
        'num_scales': [6, 9, 12, 15, 15],
        'min_size': [164, 328, 656, 820, 2130],
        'max_size': [328, 656, 820, 2130, 3441],
        'variance': [0.1, 0.2],
        'clip': True,
        'name': 'Signals',
        'trained_model': './weights/ResNet_simulated_8192_20190706_10000.pth',
        'using_gpu': False,
        'test_data_root': './data/realdata.mat',
        'saved_folder': './test/',
        'txt_name': 'test_realdata_10000.txt',
        'labelmap': ['AM', 'DSB-SC'],
        'visual_threshold': 0.02, 
    }
    test_signals(settings)
    # eval_signals(settings['saved_folder'] + settings['txt_name'])