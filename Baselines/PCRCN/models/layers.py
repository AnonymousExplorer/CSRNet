import torch.nn as nn
import torch
import torch.nn.functional as F
class CALayer(nn.Module):
    # reduction；降维比例为r=16
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale a undpscale --> channel weight
        self.conv_du = nn.Sequential(
            # channel // reduction，输出降维，即论文中的1x1xC/r
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            # channel，输出升维，即论文中的1x1xC
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
        x就是 HxWxC 通道  y是权重
        y权重通过上面方式求出, 然后 和x求乘积
        使得重要通道权重更大，不重要通道权重减小
        '''
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
    
class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)


class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(*[DenseLayer(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)])

        # local feature fusion
        self.lff = nn.Conv2d(in_channels + growth_rate * num_layers, growth_rate, kernel_size=1)

    def forward(self, x):
        return x + self.lff(self.layers(x))  # local residual learning 
    
class RDCAB(nn.Module):
    def __init__(self):
        super(RDCAB, self).__init__()
        self.r = RDB(64,64,3)
        self.ca = CALayer(64)
    def forward(self, x):
        x = self.r(x)
        x = self.ca(x)
        return x
    
class RDCAN(nn.Module):
    def __init__(self):
        super(RDCAN, self).__init__()
        self.RDCABs = nn.ModuleList([RDCAB(),RDCAB(),RDCAB()])
        self.lff = nn.Conv2d(3*64,64,1)
        self.conv = nn.Conv2d(64,64,3,1,1)

    def forward(self,x):
        x1 = self.RDCABs[0](x)
        x2 = self.RDCABs[1](x1)
        x3 = self.RDCABs[2](x2)
        x4 = self.lff(torch.cat([x1,x2,x3],1))
        x5 = x + self.conv(x4)
        return x5
    
class ISR(nn.Module):
    def __init__(self):
        super(ISR,self).__init__()
        self.conv_bn = nn.Sequential(nn.Conv2d(3,64,3,1,1),nn.BatchNorm2d(64),nn.ReLU())
        self.R =RDCAN()
        self.deconv = nn.ConvTranspose2d(64,64,3,2,1,1)
    def forward(self,x):
        x = self.conv_bn(x)
        x = self.R(x)
        x = self.deconv(x)
        return x

def batchnorm(x):
    return nn.BatchNorm2d(x.size()[1])(x)

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride = 1, bn = False, relu = True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    
class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim/2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim/2))
        self.conv2 = Conv(int(out_dim/2), int(out_dim/2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim/2))
        self.conv3 = Conv(int(out_dim/2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True
        
    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out 
Pool = nn.MaxPool2d
class Hourglass(nn.Module):
    def __init__(self, n, f, bn=None, increase=0):
        super(Hourglass, self).__init__()
        nf = f + increase
        self.up1 = Residual(f, f)
        # Lower branch
        self.pool1 = Pool(2, 2)
        self.low1 = Residual(f, nf)
        self.n = n
        # Recursive hourglass
        if self.n > 1:
            self.low2 = Hourglass(n-1, nf, bn=bn)
        else:
            self.low2 = Residual(nf, nf)
        self.low3 = Residual(nf, f)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        up1  = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2  = self.up2(low3)
        return up1 + up2
    

class PDFE(nn.Module):
    def __init__(self):
        super(PDFE,self).__init__()
        self.conv_3X3_bns = nn.ModuleList([nn.Sequential(nn.Conv2d(64,64,3,1,1),nn.BatchNorm2d(64),nn.ReLU()),nn.Sequential(nn.Conv2d(64,64,3,1,1),nn.BatchNorm2d(64),nn.ReLU())])
        self.conv_1X1_bns = nn.ModuleList([nn.Sequential(nn.Conv2d(128,64,1),nn.BatchNorm2d(64),nn.ReLU()),nn.Sequential(nn.Conv2d(64,64,1),nn.BatchNorm2d(64),nn.ReLU())])
        self.conv_1X1_hourglass = nn.Sequential(nn.Conv2d(64,19,1),nn.BatchNorm2d(19),nn.ReLU())
        self.conv_1X1_fusion = nn.Sequential(nn.Conv2d(64+19,64,1),nn.BatchNorm2d(64),nn.ReLU())
        self.RDCANs = nn.ModuleList([RDCAN(),RDCAN()])
        self.hourglass = Hourglass(4, 64, False)
    
    def forward(self,x):
        x = self.conv_1X1_bns[0](x)
        face_feature = self.RDCANs[0](self.conv_3X3_bns[0](x))
        shape_feature = self.RDCANs[1](self.conv_3X3_bns[1](x))
        shape_feature = self.conv_1X1_bns[1](shape_feature)
        shape_feature = self.hourglass(shape_feature)
        parsing_maps = self.conv_1X1_hourglass(shape_feature)
        fusion = self.conv_1X1_fusion(torch.cat([face_feature,parsing_maps],1))
        return parsing_maps, fusion

class PCRCN(nn.Module):
    def __init__(self):
        super(PCRCN,self).__init__()
        self.isr = ISR()
        self.pdfe = nn.ModuleList([PDFE(),PDFE()])
        self.deconvs = nn.ModuleList([nn.ConvTranspose2d(64,64,3,2,1,1),nn.ConvTranspose2d(64,64,3,2,1,1)])
        self.convs = nn.ModuleList([nn.Conv2d(64,3,3,1,1),nn.Conv2d(64,3,3,1,1),nn.Conv2d(64,3,3,1,1)])
    
    def forward(self,x):
        _4x_sr = []
        _8x_sr = []
        _4x_par = []
        _8x_par = []
        f1 = self.isr(x)
        _2x_sr = self.convs[0](f1)
        last_in_2x = f1
        last_in_4x = None
        upsample = F.interpolate(x,(192,192))
        for i in range(4):
            parse_4x, last_in_2x  = self.pdfe[0](torch.cat([f1,last_in_2x],1))
            f2 = self.deconvs[0](last_in_2x)
            _4x_sr.append(self.convs[1](f2))
            _4x_par.append(parse_4x)

            if last_in_4x is None:
                last_in_4x = f2
            parse_8x, last_in_4x = self.pdfe[1](torch.cat([f2,last_in_4x],1))
            f3 = self.deconvs[1](last_in_4x)
            _8x_sr.append(self.convs[2](f3)+upsample)
            _8x_par.append(parse_8x)

        return _2x_sr,_4x_sr,_8x_sr,_4x_par,_8x_par

