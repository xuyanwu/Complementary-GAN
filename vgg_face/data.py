
import torch
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def calculate_Q(label,opt):
    label = label[:, :opt.num_label]

    total_label = label.size()[0] * label.size()[1]
    M = []
    for i in range(opt.num_class):
        M.append(torch.sum(torch.ones(label.size())[label == i]) / total_label)
    M = torch.tensor(M)
    M = M.repeat(opt.num_class, 1)
    for i in range(opt.num_class):
        M[i, i] = 0

    M = M / M.sum(1).unsqueeze(dim=1)
    return M


class CIFAR10_Complementary():

    def __init__(self, root, train=True, size=32, transform = None, p1 = 1.0, p2=1.0,opt=None):

        self.raw_folder = 'raw'
        self.processed_folder = 'processed'
        self.training_file = 'training' + str(p1)+str(p2) + '.pt'
        self.test_file = 'test.pt'

        self.root = os.path.expanduser(root)
        self.train = train  # training set or test set

        self.size = size
        self.transform = transform
        self.gray = False

        if self.train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))
            # self.Q = calculate_Q(self.train_labels,opt)
            # print(self.train_data.min(),self.train_data.max())
            if len(self.train_data.size()) <= 3:
                self.gray = True
            elif self.train_data.size()[-1]==3:
                self.train_data = torch.from_numpy((self.train_data).numpy().astype(np.uint8))
            else:
                self.train_data = torch.from_numpy((self.train_data).numpy().transpose((0, 2, 3, 1)).astype(np.uint8))

            self.train_data_c = self.train_data[self.train_labels[:,1] != -1]*1
            # print(self.train_data.size(),self.train_data_c.size())

            self.train_labels_c = self.train_labels[self.train_labels[:,1] != -1]*1

            self.train_data_g_l = self.train_data.size()[0]
            self.train_data_c_l = self.train_data_c.size()[0]
            print(self.train_data_c_l)

        else:
            self.test_data, self.test_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))
            if len(self.test_data.size()) <= 3:
                self.gray = True
            elif self.test_data.size()[-1] == 3:
                self.test_data = torch.from_numpy(self.test_data.numpy().astype(np.uint8))
            else:
                self.test_data = torch.from_numpy((self.test_data).numpy().transpose((0, 2, 3, 1)).astype(np.uint8))


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        if self.train:
            # reminder = index % (self.train_labels.size()[1]-1)
            # divisor = index // (self.train_labels.size()[1]-1)
            # print(self.train_labels_c[divisor])
            # c_index = np.asscalar(np.random.choice(self.train_data_c_l,1))
            # img_g = self.train_data[g_index]
            img_c, label_c = self.train_data_c[index], self.train_labels_c[index]#[self.train_labels_c[divisor,reminder],self.train_labels_c[divisor,-1]]
            # print(img_g.size(),img_c.size(),label_c.size())
        else:
            img, target = self.test_data[index], self.test_labels[index]

        if self.gray == True:
            if self.train:
                # img_g = Image.fromarray(img_g.numpy(), mode='L')
                img_c = Image.fromarray(img_c.numpy(), mode='L')
            else:
                img = Image.fromarray(img.numpy(), mode='L')

        else:
            if self.train:
                # img_g = Image.fromarray(img_g.numpy())
                img_c = Image.fromarray(img_c.numpy())
            else:
                img = Image.fromarray(img.numpy())

        if self.transform is not None:
            if self.train:
                # img_g = self.transform(img_g)
                img_c = self.transform(img_c)
            else:
                img = self.transform(img)


        if self.train:
            # print(label_c)
            return  img_c, label_c
        else:
            return img, target

    def __len__(self):
        if self.train:
            return self.train_data_c_l
        else:
            return len(self.test_data)

class TINY_IMAGENET_Complementary_g():

    def __init__(self, root, train=True, size=32, transform = None, p1 = 1.0, p2=1.0):

        self.raw_folder = 'raw'
        self.processed_folder = 'processed'
        self.training_file = 'training' + str(p1)+str(p2) + '.pt'
        self.test_file = 'test.pt'

        self.root = os.path.expanduser(root)
        self.train = train  # training set or test set

        self.size = size
        self.transform = transform

        self.train_data, self.train_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))
        self.test_data, self.test_labels = torch.load(
            os.path.join(self.root, self.processed_folder, self.test_file))
        self.train_data = torch.cat([self.train_data,self.test_data],dim=0)
            # print(self.train_data.min(),self.train_data.max())
        self.train_data = torch.from_numpy((self.train_data).numpy().transpose((0, 2, 3, 1)).astype(np.uint8))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img_g = self.train_data[index]
        img_g = Image.fromarray(img_g.numpy())
        img_g = self.transform(img_g)

        return img_g,index

    def __len__(self):
        return self.train_data.size()[0]

def generate_c_data(opt):
    np.random.seed(40)
    p1 = opt.p1

    data = torch.load(os.path.join(opt.savingroot,opt.data_r,'data','original.pt'))


    print(data[0].size(), data[0].max(), data[0].min())

    # img_c = data[0][data[1] == 2]
    # for img in img_c:
    #     plt.imshow(np.transpose(img, [1, 2, 0]))
    #     plt.show()



    if opt.num_label is None:
        index = []

        labels = (data[1] * 1).long()

        ncls = opt.num_class
        rho = 1.0
        Q = (rho / (ncls - 1)) * np.ones((ncls, ncls))  #
        for i in range(ncls):
            Q[i, i] = 1. - rho
        Q = torch.from_numpy(Q).float().cuda()

        i = 0
        ###############p for complementary label###############################
        p2 = opt.p2
        for label in data[1]:

            c = torch.from_numpy(np.random.choice(np.arange(0, opt.num_class, 1), 1, replace=True))
            # index.append(0)
            # if label == c:
            #     index.append(1)
            #     print('aaa')
            # else:
            p = np.random.choice([0, 1], 1, p=[p1, 1 - p1])

            if label == -1:
                index.append(-1)
            else:
                if p == 0:
                    while label == c:
                        c = torch.from_numpy(np.random.choice(np.arange(0, opt.num_class, 1), 1, replace=True))
                    index.append(0)
                    labels[i] = c
                    # print(label, c)
                else:
                    index.append(1)
                    if label!= labels[i]:
                        print(label,labels[i])

            i = i + 1
    else:
        index = []

        labels = (data[1] * 1).long()


        ###############p for complementary label###############################
        label_matrix = []
        for row in range(opt.num_class):
            c_list = []

            for j in range(opt.num_label):
                c = np.random.choice(np.arange(0, opt.num_class, 1), 1, replace=True)
                while c in c_list or c == row:
                    c = np.random.choice(np.arange(0, opt.num_class, 1), 1, replace=True)
                c_list.append(c)
            label_matrix.append(c_list)
        label_matrix = np.asarray(label_matrix)
        print(label_matrix)
        Q = np.zeros((opt.num_class,opt.num_class))

        for row in range(opt.num_class):
            for col in range(opt.num_label):
                Q[row,label_matrix[row,col]] = 1.0/opt.num_label
        Q = torch.from_numpy(Q).float().cuda()
        print(Q)
        p2 = opt.p2
        i = 0
        for label in data[1]:

            p = np.random.choice([0, 1], 1, p=[p1, 1 - p1])

            c = torch.from_numpy(np.random.choice(label_matrix[label].squeeze(), 1, replace=True))

            if label == -1:
                index_r = -1
            else:
                if p == 0:
                    while c == label:
                        c = torch.from_numpy(np.random.choice(label_matrix[label].squeeze(), 1, replace=True))
                        print('repeat')
                    index_r = 0
                    labels[i] = c

                else:
                    index_r = 1

            index.append(index_r)
            i = i + 1



    index = torch.from_numpy(np.array(index)).unsqueeze(dim=1)
    labels = torch.from_numpy(np.array(labels)).unsqueeze(dim=1)

    # img_c = data[0][labels.squeeze() == 2]
    # for img in img_c:
    #     plt.imshow(np.transpose(img, [1, 2, 0]))
    #     plt.show()


    print(index.size(),labels.size())

    labels = torch.cat([labels, index], dim=1)

    if opt.data_r == 'STL10':
        label_data = data[0][index[:,0]!=-1]
        unlabel_data = data[0][index[:,0]==-1]

        label_labels = labels[index[:,0]!=-1]
        unlabel_labels = labels[index[:,0]==-1]

        choose_unlabel = np.random.choice(unlabel_data.shape[0],int(label_data.shape[0]/p2-label_data.shape[0]),replace=False)
        unlabel_data = unlabel_data[choose_unlabel]
        unlabel_labels = unlabel_labels[choose_unlabel]

        saving_data =torch.cat([label_data,unlabel_data],dim=0)
        saving_label = torch.cat([label_labels,unlabel_labels],dim=0)
        print(saving_data.size(),saving_label.size())
        torch.save([saving_data, saving_label],
                   os.path.join(opt.savingroot, opt.data_r, 'data', 'processed/training' + str(p1) + str(p2) + '.pt'))
        print('data process finished')

    else:
        img_num = labels.shape[0]
        print(labels.shape)

        choose_img = np.random.choice(img_num, int(img_num * (1 - p2)), replace=False)
        print(choose_img)
        print(choose_img.shape)
        labels[choose_img, 1] = -1
        torch.save([data[0], labels],
                   os.path.join(opt.savingroot, opt.data_r, 'data', 'processed/training' + str(p1) + str(p2) + '.pt'))
        print('data process finished')

    return Q

