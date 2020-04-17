import os
import torch
import time
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as init
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from data import *
from data import detection_collate
from data.SignalDetection import SignalDetection
from layers.modules import MultiBoxLoss
from ResNet_1D import build_SSD


def train(settings, save_dir, resume=False, using_gpu=True, using_vim=True):
    if torch.cuda.is_available():
        if using_gpu:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            print('GPU detected, why not using it? It`s way more faster.')
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists('./weights'):
        os.mkdir('./weights')

    dataset = SignalDetection('train', settings['data_dir'])

    ssd_net = build_SSD('train', cfg['min_dim'], cfg['num_classes'], cfg)
    net = ssd_net

    if using_gpu:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True
        net = net.cuda()
    if resume:
        print('resume weights...')
        ssd_net.load_weights(resume)
    else:
        print('Start training, initializing weights...')
        ssd_net.res.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    if using_vim:
        import visdom
        global viz
        viz = visdom.Visdom()

    # optimizer = optim.SGD(net.parameters(), lr=settings['lr'],
    #                       momentum=settings['momentum'], weight_decay=settings['weight_decay'])
    optimizer = optim.Adam(net.parameters(), lr=settings['lr'], betas=[0.9, 0.99], eps=1e-8,
                           weight_decay=settings['weight_decay'])
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5, False, using_gpu)

    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0

    print('Loading dataset.')

    step_index = 0
    if using_vim:
        vis_title = 'Signal Detection'
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

    data_loader = data.DataLoader(dataset, settings['batch_size'],
                                  num_workers=0, shuffle=True,
                                  collate_fn=detection_collate, pin_memory=True)

    epoch_size = len(data_loader) // settings['batch_size']
    batch_iterator = iter(data_loader)
    for iteration in range(settings['start_iter'], cfg['max_iter']):
        if using_vim and iteration != 0 and (iteration % epoch_size == 0):
            update_vis_plot(epoch, loc_loss, conf_loss, epoch_plot, None,
                            'append', epoch_size)
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
            epoch += 1

        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, settings['lr'], settings['gamma'], step_index)

        # load train data
        # images, targets = next(batch_iterator)
        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)
        except Exception as e:
            print("Loading data Exception:", e)

        if using_gpu:
            images = Variable(images.cuda())
            with torch.no_grad():
                targets = [Variable(ann.cuda()) for ann in targets]
        else:
            images = Variable(images)
            with torch.no_grad():
                targets = [Variable(ann) for ann in targets]
        # forward
        t0 = time.time()
        out = net(images)
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()
        loc_loss += loss_l.item()
        conf_loss += loss_c.item()

        if iteration % 10 == 0:
            print('timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.item()), end=' ')

        if using_vim:
            epoch += 1
            update_vis_plot(iteration, loss_l.data[0], loss_c.data[0],
                            iter_plot, epoch_plot, 'append')

        if iteration != 0 and iteration % 1000 == 0:
            localtime = time.localtime(time.time())
            track = str(localtime[0]) + str(localtime[1]).zfill(2) + str(localtime[2]).zfill(2)
            print('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(), 'weights/ResNet_simulated_8192_' +
                       track + '_'  + repr(iteration) + '.pth')
    torch.save(ssd_net.state_dict(),
               save_dir + '/' + 'ResNet_8192_max.pth')


def kaiming(param):
    init.kaiming_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        kaiming(m.weight.data)
        m.bias.data.zero_()


def adjust_learning_rate(optimizer, init_lr, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = init_lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend,
        )
    )


def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )


if __name__ == '__main__':
    settings = {
        'batch_size': 64,
        'lr': 4e-05,
        'momentum': 0.9,
        'weight_decay': 4e-05,
        'gamma': 0.5,
        'data_dir': './data/simulated_data_0706.mat',
        'start_iter': 0,
    }
    # Using vim, run python -m visdom.server must be first
    # train_weights = './weights/ResNet_8192_20190412_5000.pth'
    train_weights = False
    train(settings, './checkpoints', train_weights, False, False)
    # ssd_net = build_SSD('train', 8192, 3, cfg)
    # net = ssd_net
    # # net.cuda()
    # net.train()
    # images = torch.randn(10, 1, 8192)
    # out = net(images)
