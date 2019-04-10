
import torch.nn.functional as F
import torch
import numpy as np
import math

# def complementary_transition_matrix():
#     ncls = 10
#     rho = 1.0
#     M = (rho / (ncls - 1)) * np.ones((ncls, ncls))  #
#     for i in range(ncls):
#         M[i, i] = 1. - rho
#     M = torch.from_numpy(M).float().cuda()
#     return M
#
# Q = complementary_transition_matrix()

def forward_loss(x, target,Q):
    probt = F.softmax(x-torch.max(x, dim=1, keepdim=True)[0])
    # probt_max = torch.max(probt, dim=1, keepdim=True)[0]
    Q = (Q + 1e-12)/(1+Q.size()[0]*1e-12)
    prob = torch.mm(probt, Q)*10
    out = torch.log(prob)
    loss = F.nll_loss(out, target)

    return loss


def clip_cross_entropy(x, target):
    probt = F.softmax(x-torch.max(x, dim=1, keepdim=True)[0])
    probt = probt/torch.sum(probt,dim=1,keepdim=True)
    out = torch.log(probt*10)
    loss = F.nll_loss(out, target)

    return loss

