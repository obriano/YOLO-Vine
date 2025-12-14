import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, GhostConv
from .block import C3

class Bottle2neck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, shortcut, baseWidth=8, scale = 5):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            baseWidth: basic width of conv3x3
            scale: number of scale.
        """
        super(Bottle2neck, self).__init__()

        width = max(1, int(math.floor(planes * (baseWidth / 64.0))))
        self.conv1 = Conv(inplanes, width*scale, k=1)
        
        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
        convs = []
        for i in range(self.nums):
          convs.append(Conv(width, width, k=3))
        self.convs = nn.ModuleList(convs)

        self.conv3 = Conv(width*scale, planes * self.expansion, k=1, act=False)

        self.silu = nn.SiLU(inplace=True)
        self.scale = scale
        self.width  = width
        self.shortcut = shortcut

    def forward(self, x):
        
        if self.shortcut:
            residual = x
        out = self.conv1(x)
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0:
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        if self.scale != 1:
          out = torch.cat((out, spx[self.nums]),1)

        out = self.conv3(out)
        if self.shortcut:
            out += residual
        out = self.silu(out)
        return out


class TridentBlock(nn.Module):
    def __init__(self, c1, c2, stride=1, c=False, e=0.5, padding=[1, 2, 3], dilate=[1, 2, 3], bias=False):
        super(TridentBlock, self).__init__()
        self.stride = stride
        self.c = c
        c_ = int(c2 * e)
        self.padding = padding
        self.dilate = dilate
        self.share_weightconv1 = nn.Parameter(torch.Tensor(c_, c1, 1, 1))
        self.share_weightconv2 = nn.Parameter(torch.Tensor(c2, c_, 3, 3))

        self.bn1 = nn.BatchNorm2d(c_)
        self.bn2 = nn.BatchNorm2d(c2)

        self.act = nn.SiLU()

        nn.init.kaiming_uniform_(self.share_weightconv1, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.share_weightconv2, nonlinearity="relu")

        if bias:
            self.bias = nn.Parameter(torch.Tensor(c2))
        else:
            self.bias = None

        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward_for_small(self, x):
        residual = x
        out = nn.functional.conv2d(x, self.share_weightconv1, bias=self.bias)
        out = self.bn1(out)
        out = self.act(out)

        out = nn.functional.conv2d(out, self.share_weightconv2, bias=self.bias, stride=self.stride, padding=self.padding[0],
                                   dilation=self.dilate[0])
        out = self.bn2(out)
        out += residual
        out = self.act(out)

        return out

    def forward_for_middle(self, x):
        residual = x
        out = nn.functional.conv2d(x, self.share_weightconv1, bias=self.bias)
        out = self.bn1(out)
        out = self.act(out)

        out = nn.functional.conv2d(out, self.share_weightconv2, bias=self.bias, stride=self.stride, padding=self.padding[1],
                                   dilation=self.dilate[1])
        out = self.bn2(out)
        out += residual
        out = self.act(out)

        return out

    def forward_for_big(self, x):
        residual = x
        out = nn.functional.conv2d(x, self.share_weightconv1, bias=self.bias)
        out = self.bn1(out)
        out = self.act(out)

        out = nn.functional.conv2d(out, self.share_weightconv2, bias=self.bias, stride=self.stride, padding=self.padding[2],
                                   dilation=self.dilate[2])
        out = self.bn2(out)
        out += residual
        out = self.act(out)

        return out

    def forward(self, x):
        xm = x
        base_feat = []
        if self.c is not False:
            x1 = self.forward_for_small(x)
            x2 = self.forward_for_middle(x)
            x3 = self.forward_for_big(x)
        else:
            x1 = self.forward_for_small(xm[0])
            x2 = self.forward_for_middle(xm[1])
            x3 = self.forward_for_big(xm[2])

        base_feat.append(x1)
        base_feat.append(x2)
        base_feat.append(x3)

        return base_feat
    
class RFEM(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5, stride=1):
        super(RFEM, self).__init__()
        c = True
        layers = []
        layers.append(TridentBlock(c1, c2, stride=stride, c=c, e=e))
        c1 = c2
        for i in range(1, n):
            layers.append(TridentBlock(c1, c2))
        self.layer = nn.Sequential(*layers)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        out = self.layer(x)
        out = out[0] + out[1] + out[2] + x
        out = self.act(self.bn(out))
        return out
    
class C3_Res2Block(C3):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottle2neck(c_, c_, shortcut) for _ in range(n)))

class C3RFEM(C3):
    # C3 module with RFEM
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(RFEM(c_, c_, n=1, e=e) for _ in range(n)))

class SPPRFEM(nn.Module):
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, None, 2)
        self.cv2 = Conv(c_ * 5, c2, 1, 1, None, 2)
        # self.cv1 = Conv(c1, c_, 1, 1, None, 1) #test
        # self.cv2 = Conv(c_ * 5, c2, 1, 1, None, 1) #test
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.rfe = RFEM(c_, c_)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
    
        return self.cv2(torch.cat([x, y1, y2, self.m(y2),self.rfe(x)], 1))


class GhostSPPRFEM(nn.Module):
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = GhostConv(c1, c_, 1, 1)
        self.cv2 = GhostConv(c_ * 5, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.rfe = RFEM(c_, c_)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
    
        return self.cv2(torch.cat([x, y1, y2, self.m(y2),self.rfe(x)], 1))
