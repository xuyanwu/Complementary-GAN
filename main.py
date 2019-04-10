
import argparse
import os
import random
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.utils as utils
import torch.nn.functional as F
from models import G_CIFAR10,G_MNIST,D_CIFAR10,D_MNIST,DPN92,ResNet18
from tensorboard_logger import configure, log_value

from data import generate_c_data, CIFAR10_Complementary

from train_test import train_g,train_data_g,test,test_acc,train_c



from torchvision import datasets, transforms

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enbaled = True




def args():
    FLAG = argparse.ArgumentParser(description='ACGAN Implement With Pytorch.')
    FLAG.add_argument('--dataset', default='MNIST', help='CIFAR10 | MNIST')
    FLAG.add_argument('--savingroot', default='../result', help='path to saving.')
    FLAG.add_argument('--dataroot', default='data', help='path to dataset.')
    FLAG.add_argument('--manual_seed', default=42, help='manual seed.')
    FLAG.add_argument('--p1', default=1.0, type=float, help='p1')
    FLAG.add_argument('--p2', default=0.05, type=float, help='p2')
    FLAG.add_argument('--image_size', default=32, help='image size.')
    FLAG.add_argument('--batch_size', default=128, help='batch size.')
    FLAG.add_argument('--num_workers', default=2, help='num workers.')
    FLAG.add_argument('--num_epoches', default=40, type=int, help='num workers.')
    FLAG.add_argument('--nc', default=1, type=int, help='channel of input image.')
    FLAG.add_argument('--nz', default=64, help='length of noize.')
    FLAG.add_argument('--ndf', default=64, help='number of filters.')
    FLAG.add_argument('--ngf', default=64, help='number of filters.')
    arguments = FLAG.parse_args()
    return arguments


##############################################################################





assert torch.cuda.is_available(), '[!] CUDA required!'





def embed_z(opt):
    fixed = Variable(torch.Tensor(100, opt.nz).normal_(0, 1)).cuda()
    return fixed



def train_gan(opt):

    os.makedirs(os.path.join(opt.savingroot,opt.dataset,str(opt.p1 * 100) + '%complementary/' + str(opt.p1) + '_images'), exist_ok=True)
    os.makedirs(os.path.join(opt.savingroot,opt.dataset,str(opt.p1 * 100) + '%complementary/' + str(opt.p1) +  '_chkpts'), exist_ok=True)
    if not os.path.exists(os.path.join(opt.savingroot,opt.data_r,'data','processed/training'+str(opt.p1)+str(opt.p2)+'.pt')):
        generate_c_data(opt)
    else:
        print('data exits')

    #Build networ
    if opt.data_r == 'MNIST':
        netd_g = D_MNIST(opt.ndf, opt.nc, num_classes=10).cuda()
        netd_c = D_MNIST(opt.ndf, opt.nc, num_classes=10).cuda()
        netg = G_MNIST( opt.nz, opt.ngf, opt.nc).cuda()
    elif opt.data_r == 'CIFAR10':
        netd_g = D_CIFAR10(opt.ndf, opt.nc, num_classes=10).cuda()
        netd_c = ResNet18().cuda()#DPN92().cuda()#D_CIFAR10(opt.ndf, opt.nc, num_classes=10).cuda()
        netg = G_CIFAR10(opt.nz, opt.ngf, opt.nc).cuda()



    optd_g = optim.Adam(netd_g.parameters(), lr=0.0002,
                      betas=(0.5, 0.999))  # optim.SGD(netd.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)#
    optd_c = optim.SGD(netd_c.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)#optim.Adam(netd_c.parameters(), lr=0.0002, betas=(0.5, 0.999),weight_decay=5e-4)  #
    optg = optim.Adam(netg.parameters(), lr=0.0002,
                      betas=(0.5, 0.999))  # optim.SGD(netg.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)#

    print('training_start')
    step = 0
    acc = []

    # configure(os.path.join(opt.savingroot,opt.dataset,str(opt.p1 * 100) + '%complementary/' +  str(opt.p1) + '/logs'), flush_secs=5)
    test_loader = torch.utils.data.DataLoader(
        CIFAR10_Complementary(os.path.join(opt.savingroot,opt.data_r,'data'), train=False, transform=transforms.Compose([
            transforms.Resize(opt.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])), batch_size=128, num_workers=2)

    fixed = embed_z(opt)
    dataset = CIFAR10_Complementary(os.path.join(opt.savingroot, opt.data_r, 'data'), transform=tsfm, p1=opt.p1,
                                    p2=opt.p2)
    loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=2,
                                         worker_init_fn=np.random.seed)

    for epoch in range(opt.num_epoches):
        print(f'Epoch {epoch:03d}.')

        if epoch % int(opt.num_epoches / 3) == 0 and epoch != 0:
            for param_group in optd_c.param_groups:
                param_group['lr'] = param_group['lr'] / 10
                print(param_group['lr'])
        step = train_c(epoch,netd_c, optd_c, loader, step,opt)
        acc.append(test_acc(netd_c, test_loader))

    netd_c.load_state_dict(
        torch.load(os.path.join(opt.savingroot, opt.dataset, str(opt.p1 * 100) + '%complementary/' + str(
            opt.p1) + f'_chkpts/d_{(opt.num_epoches-1):03d}.pth')))

    step = 0

    for epoch in range(opt.num_epoches):
        print(f'Epoch {epoch:03d}.')
        if opt.data_r  == 'MNIST':
            dataset = dset.MNIST(root=opt.dataroot, download=True, transform=tsfm)
        elif opt.data_r  == 'CIFAR10':
            dataset = dset.CIFAR10(root=opt.dataroot, download=True, transform=tsfm)

        loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=2,worker_init_fn = np.random.seed)
        step = train_g(netd_g,netd_c.eval(),netg,optd_g,optg,loader,epoch,step,opt)
        test(netg, fixed, epoch, opt)

    f = open(os.path.join(opt.savingroot,opt.dataset,str(opt.p1 * 100) + '%complementary/'  + 'acc.txt'), 'w')
    for cont in acc:
        f.writelines(str(cont) + '\n')

    f.close()

def train_f_data(opt):
    os.makedirs(os.path.join(opt.savingroot, opt.dataset, str(opt.p1 * 100) + '%complementary/' + str(opt.p1) + '_chkpts_fake_data'),exist_ok=True)


    if opt.data_r == 'MNIST':
        netd = D_MNIST(opt.ndf, opt.nc, num_classes=10).cuda()
        netg = G_MNIST( opt.nz, opt.ngf, opt.nc).cuda()
    elif opt.data_r == 'CIFAR10':
        netd  = ResNet18().cuda()#DPN92().cuda()#D_CIFAR10(opt.ndf, opt.nc, num_classes=10).cuda()
        netg = G_CIFAR10(opt.nz, opt.ngf, opt.nc).cuda()
    print(os.path.join(opt.savingroot, opt.dataset, str(opt.p1 * 100) + '%complementary/' + str(
            opt.p1) + f'_chkpts/g_{(opt.num_epoches-1):03d}.pth'))
    netg.load_state_dict(torch.load(os.path.join(opt.savingroot, opt.dataset, str(opt.p1 * 100) + '%complementary/' + str(
            opt.p1) + f'_chkpts/g_{(opt.num_epoches-1):03d}.pth')))
    netg.eval()



    # netd = MobileNetV2().cuda()
    # netd = VGG('VGG19').cuda()
    # netd = ResNet18().cuda()




    # netg.load_state_dict(model.state_dict())

    optd = optim.SGD(netd.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)#optim.Adam(netd.parameters(), lr=0.0002, betas=(0.5, 0.999))  #

    print('training_start')
    step = 0
    acc = []


    test_loader = torch.utils.data.DataLoader(
        CIFAR10_Complementary(os.path.join(opt.savingroot,opt.data_r,'data'), train=False, transform=transforms.Compose([
            transforms.Resize(opt.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])), batch_size=128, num_workers=2)

    fixed = embed_z(opt)
    # dataset = dset.CIFAR10(root=opt.dataroot, download=True, transform=tsfm)

    for epoch in range(opt.num_epoches):
        dataset = CIFAR10_Complementary(os.path.join(opt.savingroot, opt.data_r, 'data'), transform=tsfm, p1=opt.p1,
                                        p2=opt.p2)

        loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=2,
                                             worker_init_fn=np.random.seed)
        print(f'Epoch {epoch:03d}.')
        if epoch % int(opt.num_epoches/3) == 0 and epoch != 0:
            for param_group in optd.param_groups:
                param_group['lr'] = param_group['lr'] / 10
                print(param_group['lr'])
        step = train_data_g(netd,netg,optd,epoch,step,opt,loader)

        acc.append(test_acc(netd, test_loader))

    f = open(os.path.join(opt.savingroot,opt.dataset,str(opt.p1 * 100) + '%complementary/' + 'acc_f_train.txt'), 'w')
    for cont in acc:
        f.writelines(str(cont) + '\n')

    f.close()



if __name__ == '__main__':

    opt = args()
    opt.data_r = opt.dataset



    if opt.data_r == 'MNIST':
        tsfm = transforms.Compose([
            transforms.Resize(opt.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    else:
        tsfm = transforms.Compose([
            transforms.Resize(opt.image_size),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])






    torch.cuda.manual_seed(opt.manual_seed)
    opt.dataset = os.path.join(opt.dataset, opt.dataset + '_' + str(opt.p2))

    configure(os.path.join(opt.savingroot, opt.dataset, str(opt.p1 * 100) + '%complementary/' +'/logs'),
              flush_secs=5)

    train_gan(opt)
    train_f_data(opt)

