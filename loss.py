import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
import pdb


def cosine_sim(im, s):
    return im.mm(s.t())

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.sim = cosine_sim

    def forward(self, hs):
        seq_len = hs.size(1)
        tgt = hs[:,-1,:]
        src = hs[:,0:-1,:]
        tgt = tgt.unsqueeze(1)

        scores = (tgt * src).sum(dim = 2)
        # We expect that the similarity
        d = scores[:,-1]
        d = d.unsqueeze(1)
        scores = scores[:,0:-1]

        d = d.expand_as(scores)
        cost = (self.margin + scores - d).clamp(min=0)
        return cost.mean()

