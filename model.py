import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
import pdb

class MaskConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(MaskConv, self).__init__()
        assert kernel_size%2 == 1
        assert padding * 2 + 1 == kernel_size

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)

        self.conv_mask = nn.Conv3d(in_channels, 2, 1,
                                    stride, 0, dilation, groups, bias)
    def forward(self, input):
        mask = self.conv_mask(input)
        mask = F.softmax(mask, dim = 1)
        opt = self.conv(input)
        opt = opt * mask[:,0,:,:,:].unsqueeze(1)
        return opt 

class ActionNet(nn.Module):
    def __init__(self, hidden_size, class_num):
        super(ActionNet, self).__init__()
        self.features = nn.Sequential(
            MaskConv(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
	    nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2), 
            MaskConv(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
	    nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            MaskConv(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
	    nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            MaskConv(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
	    nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Dropout()
        )

        self.fea_g = nn.Sequential(
            MaskConv(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace = True),
            nn.MaxPool3d(kernel_size = (3,3,5))
        )

        #self.att = nn.Bilinear(256, 256, 1)
        self.att = nn.Bilinear(256, hidden_size, 1)

        self.semantic = nn.Linear(hidden_size, hidden_size)

        self.sf = nn.Softmax(dim = -1)
        self.hidden_size = hidden_size
        self.lstm = nn.LSTMCell(256, hidden_size)

        self.classifier_att = nn.Sequential(
                nn.Dropout(),
                nn.Linear(hidden_size + 256, hidden_size),
                nn.ReLU(inplace = True),
                nn.Linear(hidden_size, class_num)
                )

    def forward(self, data):
        batch_size = data.size(0)
        data = data.view(data.size(0) * data.size(1), 1, data.size(2), data.size(3), data.size(4))
        x_fea = self.features(data)
        x = self.fea_g(x_fea)
        x = x.view(batch_size, x.size(0) // batch_size, x.size(1))

        hs = []
        atts = []
        x_fea = x_fea.view(batch_size, data.size(0) // batch_size, x_fea.size(1), x_fea.size(2), x_fea.size(3), x_fea.size(4))
        h0 = Variable(torch.zeros(batch_size, self.hidden_size)).cuda()
        c0 = Variable(torch.zeros(batch_size, self.hidden_size)).cuda()
        h,c = h0, c0
        for i in range(data.size(0) // batch_size):
            ipt = x_fea[:,i,:].permute(0, 2, 3, 4, 1).contiguous()
            h_ = h.unsqueeze(1).expand(-1, ipt.size(1) * ipt.size(2) * ipt.size(3), -1)
            ipt = ipt.view(-1, ipt.size(-1))
            h_ = h_.contiguous()
            h_ = h_.view(-1, h_.size(2))
            att = self.att(ipt, h_)
            att = att.view(x.size(0), -1)
            att = self.sf(att)
            att = att.view(x_fea.size(0), x_fea.size(3), x_fea.size(4), x_fea.size(5), 1)
            atts.append(Variable(att.data, requires_grad=False))
            ipt = (att * ipt.view(att.size(0), att.size(1), att.size(2), att.size(3), ipt.size(1))).sum(dim = 1).sum(dim = 1).sum(dim = 1)
            if not self.training and len(ipt.size()) == 1:
                ipt = ipt.unsqueeze(0)

            h, c = self.lstm(ipt, (h,c))
            hs.append(h)
        opt = torch.stack(hs, dim = 1)
        sipt = opt.view(-1, opt.size(2))
        se = self.semantic(sipt)
        se = se.view(batch_size, se.size(0) // batch_size, se.size(1))
        se = F.normalize(se, dim = 2)
        opt = torch.cat([opt, x], dim = 2)
        opt = opt.view(-1, opt.size(2))
        opt = self.classifier_att(opt)

        return opt.view(batch_size, opt.size(0) // batch_size, opt.size(1)), se

