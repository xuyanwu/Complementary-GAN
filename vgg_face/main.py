
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
from models import G_CIFAR10,G_MNIST,D_CIFAR10,D_MNIST,ResNet18,D_TINY_IMAGENET,G_TINY_IMAGENET,ResNet18_64,ResNet34_64,ResNet34,ResNet34_128
from model_resnet import Discriminator,Generator
from model_resnet32 import Discriminator32,Generator32
from tensorboard_logger import configure, log_value


from data import generate_c_data, CIFAR10_Complementary,TINY_IMAGENET_Complementary_g

from train_test import train_g,train_data_g,train_data_gc,test,test_acc,train_c,test_acc_f



from torchvision import datasets, transforms

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enbaled = True




def args():
    FLAG = argparse.ArgumentParser(description='ACGAN Implement With Pytorch.')
    FLAG.add_argument('--dataset', default='CIFAR10', help='CIFAR10 | MNIST | IMAGENET100 | VGG-FACE')
    FLAG.add_argument('--savingroot', default='../result', help='path to saving.')
    FLAG.add_argument('--dataroot', default='data', help='path to dataset.')
    FLAG.add_argument('--manual_seed', default=42, help='manual seed.')
    FLAG.add_argument('--p1', default=1.0, type=float, help='p1')
    FLAG.add_argument('--p2', default=0.2, type=float, help='p2')

    FLAG.add_argument('--image_size', default=64, help='image size.')
    FLAG.add_argument('--batch_size', default=64, help='batch size.')
    FLAG.add_argument('--num_workers', default=2, help='num workers.')
    FLAG.add_argument('--num_epoches', default=80, type=int, help='num workers.')
    FLAG.add_argument('--nc', default=3, type=int, help='channel of input image.')
    FLAG.add_argument('--nz', default=100, help='length of noize.')
    FLAG.add_argument('--ndf', default=64, help='number of filters.')
    FLAG.add_argument('--ngf', default=64, help='number of filters.')
    FLAG.add_argument('--num_class',type=int ,default=10, help='number of classes.')
    FLAG.add_argument('--num_label',type=int, default=None, help='number of classes.')
    FLAG.add_argument('--lr', type=float,default=0.01, help='learning rate sgd')
    FLAG.add_argument('--wd', type=float,default=5e-4, help='weight decay sgd')

    ##########
    FLAG.add_argument(
        '--n_d',
        default=2,
        type=int,
        help=('number of discriminator update ' 'per 1 generator update'),
    )


    FLAG.add_argument("--saving_model", default=4000, type=int,
                        help="every iter to save model")
    FLAG.add_argument('--iter', default=100000, type=int, help='maximum iterations')
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
    # if not os.path.exists(os.path.join(opt.savingroot,opt.data_r,'data','processed/training'+str(opt.p1)+str(opt.p2)+'.pt')):
    Q = generate_c_data(opt)
    # #Build networ
    if opt.data_r == 'MNIST':
        netd_g = D_MNIST(opt.ndf, opt.nc, num_classes=opt.num_class).cuda()
        netd_c = D_MNIST(opt.ndf, opt.nc, num_classes=opt.num_class).cuda()
        netg = G_MNIST( opt.nz, opt.ngf, opt.nc).cuda()
    elif opt.data_r == 'CIFAR10':
        netd_c = ResNet18(opt.num_class).cuda()
        netd_g = Discriminator32(n_class=opt.num_class, size=opt.image_size,SN=True).cuda()
        netg = Generator32(n_class=opt.num_class, size=opt.image_size, SN=True, code_dim=opt.nz).cuda()
    else:
        netd_g = Discriminator(n_class=opt.num_class,size=opt.image_size,SN=True).cuda()#D_TINY_IMAGENET(opt.ndf, opt.nc, num_classes=opt.num_class,image_size=opt.image_size).cuda()
        netg = Generator(n_class=opt.num_class,size=opt.image_size,SN=True,code_dim=opt.nz).cuda()#G_TINY_IMAGENET(opt.nz, opt.ngf, opt.nc,num_class=opt.num_class,image_size=opt.image_size).cuda()

        if opt.image_size == 32:
            netd_c = ResNet34(opt.num_class).cuda()
        elif opt.image_size == 64:
            netd_c = ResNet34_64(opt.num_class).cuda()
        elif opt.image_size == 128:
            netd_c = ResNet34_128(opt.num_class).cuda()



    optd_g = optim.Adam(netd_g.parameters(), lr=0.0004
                        ,
                      betas=(0.0, 0.9))  # optim.SGD(netd.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)#
    optd_c = optim.SGD(netd_c.parameters(), lr=opt.lr, momentum=0.9, weight_decay=opt.wd)#optim.Adam(netd_c.parameters(), lr=0.0002, betas=(0.5, 0.999),weight_decay=5e-4)  #
    optg = optim.Adam(netg.parameters(), lr=0.0001,
                      betas=(0.0, 0.9))  # optim.SGD(netg.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)#

    print('training_start')
    step = 0
    acc = []

    test_loader = torch.utils.data.DataLoader(
        CIFAR10_Complementary(os.path.join(opt.savingroot,opt.data_r,'data'), train=False, transform=transforms.Compose([
            transforms.Resize(opt.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])), batch_size=128, num_workers=2)

    dataset = CIFAR10_Complementary(os.path.join(opt.savingroot, opt.data_r, 'data'), transform=tsfm, p1=opt.p1,
                                    p2=opt.p2)
    loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=2,
                                         worker_init_fn=np.random.seed)
    #
    # for epoch in range(opt.num_epoches):
    #     print(f'Epoch {epoch:03d}.')
    #
    #     if epoch % int(opt.num_epoches/3) == 0 and epoch != 0:
    #         for param_group in optd_c.param_groups:
    #             param_group['lr'] = param_group['lr'] / 10
    #             print(param_group['lr'])
    #     step = train_c(epoch,netd_c, optd_c, loader, step,opt,Q)
    #     acc.append(test_acc(netd_c, test_loader))
    # f = open(os.path.join(opt.savingroot, opt.dataset, str(opt.p1 * 100) + '%complementary/' + 'acc.txt'), 'w')
    # for cont in acc:
    #     f.writelines(str(cont) + '\n')
    # f.close()


    netd_c.load_state_dict(
        torch.load(os.path.join(opt.savingroot, opt.dataset, str(opt.p1 * 100) + '%complementary/' + str(
            opt.p1) + f'_chkpts/d_{(opt.num_epoches-1):03d}.pth')))



    step = 0

    if opt.data_r == 'MNIST':
        dataset = dset.MNIST(root=opt.dataroot, download=True, transform=tsfm)
    elif opt.data_r == 'CIFAR10':
        dataset = dset.CIFAR10(root=opt.dataroot, download=True, transform=tsfm)
    elif opt.data_r == 'IMAGENET100':
        dataset = TINY_IMAGENET_Complementary_g(os.path.join(opt.savingroot, opt.data_r, 'data'), transform=tsfm,
                                                p1=opt.p1, p2=opt.p2)
    elif opt.data_r == 'VGG-FACE':
        dataset = TINY_IMAGENET_Complementary_g(os.path.join(opt.savingroot, opt.data_r, 'data'), transform=tsfm,
                                                p1=opt.p1, p2=opt.p2)

    train_g(netd_g, netd_c.eval(), netg, optd_g, optg, dataset, opt)




    return Q




def train_f_data_gc(opt,Q):
    os.makedirs(os.path.join(opt.savingroot, opt.dataset, str(opt.p1 * 100) + '%complementary/' + str(opt.p1) + '_chkpts_fake_data'),exist_ok=True)


    if opt.data_r == 'MNIST':
        netd = D_MNIST(opt.ndf, opt.nc, num_classes=opt.num_class).cuda()
        netg = G_MNIST( opt.nz, opt.ngf, opt.nc).cuda()
    elif opt.data_r == 'CIFAR10':
        netd  = ResNet18(opt.num_class).cuda()#DPN92().cuda()#D_CIFAR10(opt.ndf, opt.nc, num_classes=10).cuda()
        netg = Generator32(n_class=opt.num_class, size=opt.image_size, SN=True, code_dim=opt.nz).cuda()
    else:
        netg = Generator(n_class=opt.num_class,size=opt.image_size,SN=True,code_dim=opt.nz).cuda()#G_TINY_IMAGENET(opt.nz, opt.ngf, opt.nc, image_size=opt.image_size).cuda()
        if opt.image_size == 32:
            netd = ResNet34(opt.num_class).cuda()
        elif opt.image_size == 64:
            netd = ResNet34_64(opt.num_class).cuda()
        elif opt.image_size == 128:
            netd = ResNet34_128(opt.num_class).cuda()
    print(os.path.join(opt.savingroot, opt.dataset, str(opt.p1 * 100) + '%complementary/' + str(
            opt.p1) + f'_chkpts/g_{opt.iter:03d}.pth'))
    netg.load_state_dict(torch.load(os.path.join(opt.savingroot, opt.dataset, str(opt.p1 * 100) + '%complementary/' + str(
            opt.p1) + f'_chkpts/g_{opt.iter:03d}.pth')))
    netg.eval()

    netd = nn.DataParallel(netd)
    netg = nn.DataParallel(netg)


    optd = optim.SGD(netd.module.parameters(), lr=opt.lr, momentum=0.9, weight_decay=opt.wd)#optim.Adam(netd.parameters(), lr=0.0002, betas=(0.5, 0.999))  #

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

        loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=2,pin_memory=True,
                                             worker_init_fn=np.random.seed)
        print(f'Epoch {epoch:03d}.')
        if epoch % int(opt.num_epoches/3) == 0 and epoch != 0:
            for param_group in optd.param_groups:
                param_group['lr'] = param_group['lr'] / 10
                print(param_group['lr'])
        step = train_data_gc(netd,netg,optd,epoch,step,opt,loader,Q)

        acc.append(test_acc(netd, test_loader))

    f = open(os.path.join(opt.savingroot,opt.dataset,str(opt.p1 * 100) + '%complementary/' + 'acc_f_train.txt'), 'w')
    for cont in acc:
        f.writelines(str(cont) + '\n')

    f.close()

def train_f_data_g(opt,Q):
    os.makedirs(os.path.join(opt.savingroot, opt.dataset, str(opt.p1 * 100) + '%complementary/' + str(opt.p1) + '_chkpts_fake_data'),exist_ok=True)


    if opt.data_r == 'MNIST':
        netd = D_MNIST(opt.ndf, opt.nc, num_classes=opt.num_class).cuda()
        netg = G_MNIST( opt.nz, opt.ngf, opt.nc).cuda()
    elif opt.data_r == 'CIFAR10':
        netd  = ResNet18(opt.num_class).cuda()#DPN92().cuda()#D_CIFAR10(opt.ndf, opt.nc, num_classes=10).cuda()
        netg = Generator32(n_class=opt.num_class, size=opt.image_size, SN=True, code_dim=opt.nz).cuda()
    else:
        netg = Generator(n_class=opt.num_class,size=opt.image_size,SN=True,code_dim=opt.nz).cuda()#G_TINY_IMAGENET(opt.nz, opt.ngf, opt.nc, image_size=opt.image_size).cuda()
        if opt.image_size == 32:
            netd = ResNet34(opt.num_class).cuda()
        elif opt.image_size == 64:
            netd = ResNet34_64(opt.num_class).cuda()
        elif opt.image_size == 128:
            netd = ResNet34_128(opt.num_class).cuda()
    print(os.path.join(opt.savingroot, opt.dataset, str(opt.p1 * 100) + '%complementary/' + str(
            opt.p1) + f'_chkpts/g_{opt.iter:03d}.pth'))
    netg.load_state_dict(torch.load(os.path.join(opt.savingroot, opt.dataset, str(opt.p1 * 100) + '%complementary/' + str(
            opt.p1) + f'_chkpts/g_{opt.iter:03d}.pth')))
    netg.eval()

    netd = nn.DataParallel(netd)
    netg = nn.DataParallel(netg)


    optd = optim.SGD(netd.module.parameters(), lr=opt.lr, momentum=0.9, weight_decay=opt.wd)#optim.Adam(netd.parameters(), lr=0.0002, betas=(0.5, 0.999))  #

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

        loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=2,pin_memory=True,
                                             worker_init_fn=np.random.seed)
        print(f'Epoch {epoch:03d}.')
        if epoch % int(opt.num_epoches/3) == 0 and epoch != 0:
            for param_group in optd.param_groups:
                param_group['lr'] = param_group['lr'] / 10
                print(param_group['lr'])
        step = train_data_g(netd,netg,optd,epoch,step,opt,loader,Q)

        acc.append(test_acc(netd, test_loader))

    f = open(os.path.join(opt.savingroot,opt.dataset,str(opt.p1 * 100) + '%complementary/' + 'Nacc_f_train.txt'), 'w')
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
            transforms.RandomCrop(opt.image_size, padding=int(opt.image_size/8)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])






    torch.cuda.manual_seed(opt.manual_seed)
    opt.dataset = os.path.join(opt.dataset, opt.dataset + '_' + str(opt.p2))

    configure(os.path.join(opt.savingroot, opt.dataset, str(opt.p1 * 100) + '%complementary/' +'/logs'),
              flush_secs=5)

    Q = train_gan(opt)
    train_f_data_gc(opt,Q)
    train_f_data_g(opt, Q)

