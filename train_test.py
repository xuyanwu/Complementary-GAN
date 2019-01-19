
from torch.autograd import Variable
import torch
import torch.nn.functional as F
from loss import forward_loss
from tensorboard_logger import configure, log_value
import torchvision
import os
import matplotlib.pyplot as plt
import numpy as np
import time

def denorm(x):
    return (x +1)/2

def train_c(epoch,netd_c,optd_c,loader,step,opt):


    netd_c.train()
    for _, (image_g,image_c, label) in enumerate(loader):

        # plt.imshow(np.transpose((image[0].cpu().numpy()+1)/2,[1,2,0]))
        # plt.show()
        # print(image.max(),image.min())

        index = label[:, 1]
        index = Variable(index).cuda()
        label = label[:, 0]

        real_label = label.cuda()

        real_loss_c = torch.zeros(1).cuda()
        if sum(index==1)+ sum(index == 0)>0:

            real_input_c = Variable(image_c).cuda()

            _, real_cls = netd_c(real_input_c)
            if sum(index == 1) > 0:
                real_loss_c += F.cross_entropy(real_cls[index == 1], real_label[index == 1])  #
            elif sum(index == 0) > 0:
                real_loss_c += forward_loss(real_cls[index == 0], real_label[index == 0])  #

            optd_c.zero_grad()
            real_loss_c.backward()
            optd_c.step()

    torch.save(netd_c.state_dict(), os.path.join(opt.savingroot, opt.dataset,
                                                 str(opt.p1 * 100) + '%complementary/' + str(
                                                     opt.p1) + f'_chkpts/d_{epoch:03d}.pth'))

    return step

def train_g(netd_g,netd_c,netg,optd_g,optg,loader,epoch,step,opt):

    g_rate = 1
    netg.train()
    netd_g.train()
    netd_c.train()
    for _, (image_g, label) in enumerate(loader):

        # plt.imshow(np.transpose((image[0].cpu().numpy()+1)/2,[1,2,0]))
        # plt.show()
        # print(image.max(),image.min())

        label = label
        # # print(image.size())
        # # print(label.size())
        # #######################
        # # real input and label
        # #######################
        real_input_g = image_g.cuda()
        real_ = torch.ones(label.size()).cuda()
        #
        #
        #######################
        # # fake input and label
        # #######################
        noise = torch.Tensor(opt.batch_size, opt.nz).normal_(0, 1).cuda()
        fake_label =torch.LongTensor(opt.batch_size).random_(10).cuda()
        fake_ = torch.zeros(fake_label.size()).cuda()
        #
        # #######################
        # # update net d
        # #######################
        fake_input = netg(noise,fake_label)
        real_pred, _ = netd_g(real_input_g)
        fake_pred, _ = netd_g(fake_input.detach())
        real_loss_g = F.binary_cross_entropy(real_pred, real_) * g_rate
        fake_loss = F.binary_cross_entropy(fake_pred, fake_)*g_rate #
        real_loss_g = real_loss_g + fake_loss

        optd_g.zero_grad()
        real_loss_g.backward()
        optd_g.step()



        d_loss = real_loss_g

        ######################
        # update net g
        ######################
        real_ = Variable(torch.ones(fake_label.size())).cuda()
        for g_iter in range(5):

            optg.zero_grad()

            noise = Variable(torch.Tensor(opt.batch_size, opt.nz).normal_(0, 1)).cuda()

            fake_input = netg(noise, fake_label)

            fake_pred, _ = netd_g(fake_input)


            g_loss = F.binary_cross_entropy(fake_pred, real_) * g_rate

            if g_iter == 0:
                _, fake_cls = netd_c(fake_input)
                g_loss += F.cross_entropy(fake_cls, fake_label)
            g_loss.backward()
            optg.step()

        step = step + 1
        log_value('d_loss', d_loss, step)
        log_value('g_loss', g_loss, step)

    #######################
    # save image pre epoch
    #######################
    torchvision.utils.save_image(denorm(fake_input.data), os.path.join(opt.savingroot,opt.dataset,str(opt.p1 * 100) + '%complementary/' +str(opt.p1)+f'_images/fake_{epoch:03d}.jpg'))
    # utils.save_image(denorm(real_input.data), f'images/real_{epoch:03d}.jpg')

    #######################
    # save model pre epoch
    #######################
    torch.save(netg.state_dict(), os.path.join(opt.savingroot,opt.dataset,str(opt.p1 * 100) + '%complementary/' + str(opt.p1)+f'_chkpts/g_{epoch:03d}.pth'))

    return step

def train_data_g(netd,netg,optd,epoch,step,opt,loader):

    netg.eval()
    netd.train()
    for _, (image_g,image_c, label) in enumerate(loader):

        # plt.imshow(np.transpose((image[0].cpu().numpy()+1)/2,[1,2,0]))
        # plt.show()
        # print(image.max(),image.min())

        index = label[:, 1]
        index = Variable(index).cuda()
        label = label[:, 0]
        real_label = label.cuda()

        real_loss_c = torch.zeros(1).cuda()
        if sum(index == 1) + sum(index == 0) > 0:

            real_input_c = Variable(image_c).cuda()

            _, real_cls = netd(real_input_c)
            if sum(index == 1) > 0:
                real_loss_c += F.cross_entropy(real_cls[index == 1], real_label[index == 1])  #
            elif sum(index == 0) > 0:
                real_loss_c += forward_loss(real_cls[index == 0], real_label[index == 0])  #



        #######################
        # fake input and label
        #######################
        noise = Variable(torch.Tensor(opt.batch_size, opt.nz).normal_(0, 1)).cuda()
        fake_label = Variable(torch.LongTensor(opt.batch_size).random_(10)).cuda()
        optd.zero_grad()
        fake_input = netg(noise,fake_label)
        # print(fake_input.min(),fake_input.max())

        #
        # img = np.transpose((fake_input[0].cpu().detach().numpy()+1)/2,[1,2,0])

        # if img.shape[2] == 1:
        #     img = np.concatenate([img,img,img],axis=2)

        # plt.imshow(img)
        # plt.show()
        # print(fake_label[0])
        # time.sleep(1)

        fake_pred, fake_cls = netd(fake_input.detach())


        fake_loss_c = F.cross_entropy(fake_cls,fake_label) #

        # if epoch >=80:
        #     fake_loss = fake_loss + cep(fake_cls, fake_label)

        c_loss = fake_loss_c+real_loss_c
        c_loss.backward()
        optd.step()

        log_value('c_f_loss', c_loss, step)

    torch.save(netd.state_dict(), os.path.join(opt.savingroot,opt.dataset,str(opt.p1 * 100) + '%complementary/' + str(opt.p1)+f'_chkpts_fake_data/d_{epoch:03d}.pth'))
    return step


def test(netg,fixed,epoch,opt):
    netg.eval()

    fixed = Variable(torch.Tensor(100, opt.nz).normal_(0, 1)).cuda()
    label = Variable(torch.LongTensor([range(10)] * 10)).view(-1).cuda()

    fixed_input = netg(fixed,label)

    torchvision.utils.save_image(denorm(fixed_input.data), os.path.join(opt.savingroot,opt.dataset,str(opt.p1 * 100) + '%complementary/' + str(opt.p1)+f'_images/fixed_{epoch:03d}.jpg'), nrow=10)

def test_acc(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            # plt.imshow(np.transpose((data[0].cpu().numpy()+1)/2,[1,2,0]))
            # plt.show()
            output = model(data)[1]
            test_loss += F.nll_loss(output, target).sum().item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)*1.0))

    return correct / len(test_loader.dataset)*1.0