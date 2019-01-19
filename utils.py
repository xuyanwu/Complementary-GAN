
import torchvision.datasets as dset
from torchvision import datasets, transforms
import torch
import numpy as np
import pickle

tsfm=transforms.Compose([
transforms.ToTensor()
])
dataset = dset.CIFAR10(root='./data',train=False, download=True,transform=tsfm)

data_loader = torch.utils.data.DataLoader(dataset,batch_size=2000, num_workers=16)

print(len(data_loader))

for i,data in enumerate(data_loader):

    if i==0:
        imgs,labels = torch.round(data[0]*255),data[1]
    else:
        imgs,labels = torch.cat([imgs,torch.round(data[0]*255)],dim=0),torch.cat([labels,data[1]],dim=0)


print(imgs.size(),imgs.max(),imgs.min(),labels.size())
torch.save([imgs,labels],'test.pt')
