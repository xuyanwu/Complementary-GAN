
import torch
import os
import numpy as np
from PIL import Image



class CIFAR10_Complementary():

    def __init__(self, root, train=True, size=32, transform = None, p1 = 1.0, p2=1.0):

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
            # print(self.train_data.min(),self.train_data.max())
            if self.train_data.size()[1] > 3:
                self.gray = True
            else:
                self.train_data = torch.from_numpy((torch.round(self.train_data).numpy()).transpose((0, 2, 3, 1)).astype(np.uint8))

            self.train_data_c = self.train_data[self.train_labels[:,1] != -1]*1
            # print(self.train_data.size(),self.train_data_c.size())

            self.train_labels_c = self.train_labels[self.train_labels[:,1] != -1]*1

            self.train_data_g_l = self.train_data.size()[0]
            self.train_data_c_l = self.train_data_c.size()[0]

        else:
            self.test_data, self.test_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))
            if self.test_data.size()[1] > 3:
                self.gray = True

            else:
                self.test_data = torch.from_numpy((torch.round(self.test_data).numpy()).transpose((0, 2, 3, 1)).astype(np.uint8))


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            g_index = np.asscalar(np.random.choice(self.train_data_g_l,1))
            img_g = self.train_data[g_index]
            img_c, label_c = self.train_data_c[index], self.train_labels_c[index]
            # print(img_g.size(),img_c.size(),label_c.size())
        else:
            img, target = self.test_data[index], self.test_labels[index]

        if self.gray == True:
            if self.train:
                img_g = Image.fromarray(img_g.numpy(), mode='L')
                img_c = Image.fromarray(img_c.numpy(), mode='L')
            else:
                img = Image.fromarray(img.numpy(), mode='L')

        else:
            if self.train:
                img_g = Image.fromarray(img_g.numpy())
                img_c = Image.fromarray(img_c.numpy())
            else:
                img = Image.fromarray(img.numpy())

        if self.transform is not None:
            if self.train:
                img_g = self.transform(img_g)
                img_c = self.transform(img_c)
            else:
                img = self.transform(img)


        if self.train:
            return img_c, label_c
        else:
            return img, target

    def __len__(self):
        if self.train:
            return self.train_data_c_l
        else:
            return len(self.test_data)


def generate_c_data(opt):
    p1 = opt.p1

    data = torch.load(os.path.join(opt.savingroot,opt.data_r,'data','original.pt'))


    print(data[0].size(), data[0].max(), data[0].min())

    index = []

    labels = data[1] * 1
    i = 0

    ###############p for complementary label###############################

    p2 = opt.p2
    for label in data[1]:

        c = torch.from_numpy(np.random.choice(np.arange(0, 10, 1), 1, replace=True))
        # index.append(0)
        # if label == c:
        #     index.append(1)
        #     print('aaa')
        # else:
        p = np.random.choice([0, 1], 1, p=[p1, 1 - p1])

        if p == 0:
            while label == c:
                c = torch.from_numpy(np.random.choice(np.arange(0, 10, 1), 1, replace=True))
            index.append(0)
            labels[i] = c
            # print(label, c)
        else:
            index.append(1)

        i = i + 1

    index = torch.from_numpy(np.array(index)).unsqueeze(dim=1)
    labels = labels.unsqueeze(dim=1)

    labels = torch.cat([labels, index], dim=1)

    img_num = labels.shape[0]
    print(labels.shape)

    choose_img = np.random.choice(img_num, int(img_num * (1-p2)), replace=False)
    print(choose_img.shape)
    labels[choose_img,1]=-1

    torch.save([data[0], labels], os.path.join(opt.savingroot,opt.data_r,'data','processed/training'+str(p1)+str(p2)+'.pt'))
    print('data process finished')