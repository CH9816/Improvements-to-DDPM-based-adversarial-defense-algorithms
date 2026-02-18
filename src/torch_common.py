import torch 
from torch import nn, Tensor
import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

import numpy as np

cpu, gpu = torch.device("cpu"), torch.device("cuda")

en = enumerate
islist = lambda l: isinstance(l, list) or isinstance(l, tuple)

LOW_VRAM = True


def project_linf(x, orig, eps):
    diff = -(x-orig)
    return orig + torch.sign(diff) * torch.min(diff.abs(), torch.ones_like(diff) * eps)




def project_l2(x : Tensor, orig : Tensor, eps) -> Tensor:
    pass;

    diff = x - orig
    diff_norm = torch.norm(diff.view(x.shape[0], -1), p=2,dim=1)

    K = eps / (diff_norm)
    K = torch.min(K, torch.ones_like(K))

    allowed_peturb = diff * K.view(-1, 1,1,1)

    out = torch.clamp(orig + allowed_peturb, 0, 1)
    

    return out


def get_acc(out : Tensor, y : Tensor):
    out = torch.argmax(out, dim=1)
    return (out == y).float().mean()





def get_train_test_split(dataset, split, bsize = 32, doShuffle = True):
    if split is None:
        return DataLoader(dataset, bsize, doShuffle)

    size = len(dataset)
    testSize = split * size
    trainSize = size - testSize

    return [
        # the dataLOADER created from the dataSET returned by .. 
        DataLoader(dataSet, bsize, doShuffle) 
            for dataSet in 

                # .. randomly splitting the whole dataset
                torch.utils.data.random_split(
                                    dataset, 
                                    [int(trainSize), int(testSize)])
    ]





def getCIFAR(
        dir="cifar",
        targetImgSize = 32, bsize = 32,
        # download if dataset is not already downloaded 
        # in folder specified by dir
        downloadIfNotExists = False,
        shuffle = True,
        train_test_split = None
    ):

    tforms = transforms.Compose([    
        transforms.Resize(targetImgSize), transforms.ToTensor(),
        #transforms.Normalize(
        #    mean=[0.4914, 0.4822, 0.4465], 
        #    std=[0.2470, 0.2435, 0.2616]
        #)
    ]) 

    cifar10 = CIFAR10(dir, True, tforms, download=downloadIfNotExists)
    return get_train_test_split(cifar10, train_test_split, bsize, shuffle)



def topil(tensor : Tensor, rescale = True):
    #if rescale: tensor = ((tensor + 1.) / 2.)
    tensor = tensor.cpu().detach()
    #tensor = tempScale(tensor)
    return transforms.ToPILImage()(
        tensor if len(list(tensor.shape)) == 3 else
        make_grid([img for img in tensor])
    )

def showtensor(x):
    topil(x.cpu()).show()

def showtensors(*xs):
    if len(xs) > 0 and islist(xs[0]): xs=xs[0]
    out = torch.cat([x.cpu() for x in xs],dim=0)
    showtensor(out)
