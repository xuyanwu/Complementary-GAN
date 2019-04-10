
from torch.autograd import Variable
import torch
import torch.nn.functional as F
from loss import forward_loss,clip_cross_entropy
from tensorboard_logger import configure, log_value
import torchvision
import os
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm
from torchvision import utils
from torch.utils.data import DataLoader
from torch.autograd import grad
import torch.nn as nn

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag




def denorm(x):
    return (x +1)/2

def sample_data(dataset,batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                 pin_memory=True, drop_last=True, worker_init_fn=np.random.seed,num_workers=2)
    # print(len(loader))
    loader = iter(loader)

    while True:
        try:
            yield next(loader)

        except StopIteration:
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                 pin_memory=True, drop_last=True, worker_init_fn=np.random.seed,num_workers=2)
            loader = iter(loader)
            yield next(loader)

def train_c(epoch,netd_c,optd_c,loader,step,opt,Q):


    netd_c.train()
    pbar = tqdm(enumerate(loader))
    for _, (image_c, label) in pbar:



        index = label[:, 1]
        index = Variable(index).cuda()
        label = label[:, 0]
        # print(image_c.size())
        # plt.figure(0)
        # if torch.sum(label==1)>1:
        #     plt.imshow(np.transpose((image_c[label == 1][0].cpu().numpy() + 1) / 2, [1, 2, 0]))
        #     plt.show()
        #     print(image_c.max(), image_c.min())

        real_label = label.cuda().long()

        real_loss_c = torch.zeros(1).cuda()
        if torch.sum(torch.ones(real_label.size())[index==1])+ torch.sum(torch.ones(real_label.size())[index==0])>0:

            real_input_c = Variable(image_c).cuda()

            _, real_cls = netd_c(real_input_c)
            if torch.sum(torch.ones(real_label.size())[index==1]) > 0:
                real_loss_c += clip_cross_entropy(real_cls[index == 1], real_label[index == 1])  #
                # print(real_loss_c)
            if torch.sum(torch.ones(real_label.size())[index==0]) > 0:
                real_loss_c += forward_loss(real_cls[index == 0], real_label[index == 0],Q)  #

            optd_c.zero_grad()
            real_loss_c.backward()
            optd_c.step()
        pbar.set_description(
            'Loss: {:.6f}'.format(real_loss_c[0]))

    torch.save(netd_c.state_dict(), os.path.join(opt.savingroot, opt.dataset,
                                                 str(opt.p1 * 100) + '%complementary/' + str(
                                                     opt.p1) + f'_chkpts/d_{epoch:03d}.pth'))

    return step

def train_g(discriminator,netd_c,generator,d_optimizer,g_optimizer,dataset,opt):

    dataset = sample_data(dataset,opt.batch_size)
    pbar = tqdm(range(opt.iter), dynamic_ncols=True)

    requires_grad(generator, False)
    requires_grad(discriminator, True)
    requires_grad(netd_c, False)

    disc_loss_val = 0
    gen_loss_val = 0
    entropy_loss = 0

    d_fake_loss_D = 0
    d_real_loss_D = 0
    d_fake_loss_G = 0


    for i in pbar:
        discriminator.zero_grad()
        real_image, _ = next(dataset)
        b_size = real_image.size(0)
        real_image = real_image.cuda()
        # label = torch.multinomial(
        #     torch.ones(opt.num_class), opt.batch_size, replacement=True
        # ).cuda()
        choose_label = np.random.choice(opt.num_class,int(5),replace=True)
        label = torch.from_numpy(np.random.choice(choose_label,opt.batch_size, replace=True))
        fake_image = generator(
            torch.randn(b_size, opt.nz).cuda(), label.cuda()
        )

        fake_predict = discriminator(fake_image)
        real_predict = discriminator(real_image)
        acc_d_r = torch.sum(torch.ones(real_predict.size())[real_predict>0])/opt.batch_size
        acc_d_f = torch.sum(torch.ones(fake_predict.size())[fake_predict < 0]) / opt.batch_size
        d_fake_loss_D = F.relu(1 + fake_predict).mean()
        d_real_loss_D = F.relu(1 - real_predict).mean()
        loss = d_fake_loss_D + d_real_loss_D
        disc_loss_val = loss.detach().item()
        loss.backward()
        d_optimizer.step()

        if (i + 1) % opt.n_d == 0:
            requires_grad(generator, True)
            requires_grad(discriminator, False)

            # for giter in range(2):
            generator.zero_grad()

            # input_class = torch.multinomial(
            #     torch.ones(opt.num_class), opt.batch_size, replacement=True
            # ).cuda()
            choose_label = np.random.choice(opt.num_class, int(5), replace=True)
            input_class = torch.from_numpy(np.random.choice(choose_label, opt.batch_size, replace=True)).cuda()
            fake_image = generator(
                torch.randn(opt.batch_size, opt.nz).cuda(), input_class
            )
            predict = discriminator(fake_image)
            # print(fake_image.size())

            # if giter == 0:
            _, fake_c = netd_c(fake_image)
            entropy_loss = clip_cross_entropy(fake_c, input_class)
            d_fake_loss_G = predict.mean()
            loss = -d_fake_loss_G + entropy_loss * 0.2
            gen_loss_val = loss.detach().item()
            loss.backward()
            g_optimizer.step()


            requires_grad(generator, False)
            requires_grad(discriminator, True)



        if (i + 1) % 1000 == 0:
            generator.train(False)
            input_class = torch.arange(opt.num_class).long().repeat(5).cuda()
            input_z = torch.randn(opt.num_class * 5, opt.nz).cuda()
            n = int(input_class.size()[0] / 5)

            for j in range(5):
                if j == 0:
                    fake_image = generator(input_z[n * j:n * (j + 1)], input_class[n * j:n * (j + 1)]).detach()
                else:
                    fake_image = torch.cat([fake_image, generator(input_z[n * j:n * (j + 1)], input_class[n * j:n * (j + 1)]).detach()],
                                  dim=0)
            generator.train(True)
            utils.save_image(
                fake_image.cpu().data,
                os.path.join(opt.savingroot,opt.dataset,str(opt.p1 * 100) + '%complementary/' + str(opt.p1)+ f'_images/{str(i + 1).zfill(7)}.png'),
                nrow=opt.num_class,
                normalize=True,
                range=(-1, 1),
            )

        pbar.set_description(
            (f'{i + 1}; G: {gen_loss_val:.5f};' f' D: {disc_loss_val:.5f}' f' G|D: {acc_d_f:.5f}|{acc_d_r:.5f}')
        )
        log_value('acc_d_r', acc_d_r, i)
        log_value('acc_d_f', acc_d_f, i)
        log_value('class_entropy_loss', entropy_loss, i)
        log_value('d_real_loss_D', d_real_loss_D, i)
        log_value('d_fake_loss_D', d_fake_loss_D, i)
        log_value('d_fake_loss_G', d_fake_loss_G, i)

        if (i + 1) % opt.saving_model == 0:
            torch.save(generator.state_dict(), os.path.join(opt.savingroot, opt.dataset,
                                                            str(opt.p1 * 100) + '%complementary/' + str(
                                                                opt.p1) + f'_chkpts/g_{i + 1:03d}.pth'))
            torch.save(discriminator.state_dict(), os.path.join(opt.savingroot, opt.dataset,
                                                            str(opt.p1 * 100) + '%complementary/' + str(
                                                                opt.p1) + f'_chkpts/dg_{i + 1:03d}.pth'))
            test_acc_f(netd_c, generator, opt)



def train_data_gc(netd,netg,optd,epoch,step,opt,loader,Q):

    netg.eval()
    requires_grad(netg, False)
    netd.train()
    for _, (image_c, label) in enumerate(loader):

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
                real_loss_c += forward_loss(real_cls[index == 0], real_label[index == 0],Q=Q)  #



        #######################
        # fake input and label
        #######################
        noise = torch.randn(opt.batch_size, opt.nz).cuda()
        fake_label = torch.multinomial(
                torch.ones(opt.num_class), opt.batch_size, replacement=True
            ).cuda()
        optd.zero_grad()
        fake_input = netg(noise,fake_label)
        # print(fake_input.min(),fake_input.max())

        #
        # img = np.transpose((fake_input[0].cpu().detach().numpy()+1)/2,[1,2,0])


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

def train_data_g(netd,netg,optd,epoch,step,opt,loader,Q):

    netg.eval()
    requires_grad(netg, False)
    netd.train()
    for _, (image_c, label) in enumerate(loader):

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
                real_loss_c += forward_loss(real_cls[index == 0], real_label[index == 0],Q=Q)  #



        #######################
        # fake input and label
        #######################
        noise = torch.randn(opt.batch_size, opt.nz).cuda()
        fake_label = torch.multinomial(
                torch.ones(opt.num_class), opt.batch_size, replacement=True
            ).cuda()
        optd.zero_grad()
        fake_input = netg(noise,fake_label)
        # print(fake_input.min(),fake_input.max())

        #
        # img = np.transpose((fake_input[0].cpu().detach().numpy()+1)/2,[1,2,0])


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

    torch.save(netd.state_dict(), os.path.join(opt.savingroot,opt.dataset,str(opt.p1 * 100) + '%complementary/' + str(opt.p1)+f'_chkpts_fake_data/Nd_{epoch:03d}.pth'))
    return step


def test(netg,fixed,epoch,opt):
    netg.eval()

    fixed = torch.randn(opt.num_class*10, opt.nz).cuda()
    label = Variable(torch.LongTensor(np.ndarray.tolist(np.arange(0, opt.num_class, 1)) * 10)).view(-1).cuda()

    fixed_input = netg(fixed,label)

    torchvision.utils.save_image(denorm(fixed_input.data), os.path.join(opt.savingroot,opt.dataset,str(opt.p1 * 100) + '%complementary/' + str(opt.p1)+f'_images/fixed_{epoch:03d}.jpg'), nrow=10)

def test_acc_f(model, netg,opt):
    requires_grad(netg, False)
    test_loss = 0
    correct = 0
    for _ in range(100):
            noise = torch.randn(opt.batch_size, opt.nz).cuda()
            fake_label = torch.multinomial(
                torch.ones(opt.num_class), opt.batch_size, replacement=True
            ).cuda()
            data = netg(noise,fake_label)
            target = fake_label
            # if torch.sum(target == 1) > 1:
            #     plt.figure(1)
            #     plt.imshow(np.transpose((data[target == 1][0].cpu().numpy() + 1) / 2, [1, 2, 0]))
            #     plt.show()
            output = model(data)[1]
            test_loss += F.nll_loss(output, target).sum().item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= 100*opt.batch_size
    print(correct)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, 100*opt.batch_size,
        100. * correct / (100.*opt.batch_size*1.0)))

    requires_grad(netg, True)

    return correct / 100*opt.batch_size*1.0

def test_acc(model, test_loader):
    requires_grad(model, False)
    test_loss = 0
    correct = 0
    for data, target in test_loader:
            data, target = data.cuda(), target.cuda().long()
            # if torch.sum(target == 1) > 1:
            #     plt.figure(1)
            #     plt.imshow(np.transpose((data[target == 1][0].cpu().numpy() + 1) / 2, [1, 2, 0]))
            #     plt.show()
            output = model(data)[1]
            test_loss += F.nll_loss(output, target).sum().item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)*1.0))

    requires_grad(model, True)

    return correct / len(test_loader.dataset)*1.0