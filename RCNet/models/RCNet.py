import torch.nn as nn
import torch.nn.functional as F
from models.layers import *
import torch

class SharedFeature(nn.Module):
    def __init__(self):
        super(SharedFeature,self).__init__()
        self.conv1 = nn.Conv2d(3,128,3,1,1)
        self.res1 = ResBlock()
        self.res2 = nn.Sequential(ResBlock(),ResBlock())
        self.res3 = nn.Sequential(ResBlock(),ResBlock(),ResBlock())

    def forward(self,x):
        x = self.conv1(x)
        s1 = x
        x = F.max_pool2d(x,3,2,1)
        x = self.res1(x)
        s2 = x
        x = F.max_pool2d(x,3,2,1)
        x = self.res2(x)
        s3 = x
        x = F.max_pool2d(x,3,2,1)
        x = self.res3(x)
        s4 = x

        w,h = list(s1.shape[-2:])
        return F.interpolate(s1,(w//8,h//8)) + 3*F.interpolate(s2,(w//8,h//8)) + 3*F.interpolate(s3,(w//8,h//8)) + s4
    

class LandmarkEs(nn.Module):
    def __init__(self):
        super(LandmarkEs,self).__init__()
        self.deconvs = nn.Sequential(
            nn.ConvTranspose2d(128,128,3,2,1,1),
            nn.ReLU(),
            nn.ConvTranspose2d(128,128,3,2,1,1),
            nn.ReLU(),
            nn.ConvTranspose2d(128,128,3,2,1,1),
            nn.ReLU()
        )

        self.hgs = nn.Sequential(
            Hourglass(4, 128, False)
        ) 

        self.conv = nn.Sequential(nn.Conv2d(128,68,3,1,1),nn.ReLU())
        
    def forward(self,x):

        x = self.deconvs(x)
        x1 = x

        x = self.hgs(x)

        x2 = x
        x = self.conv(x)

        return x
    
class ComponentAwa(nn.Module):
    def __init__(self):
        super(ComponentAwa,self).__init__()
        self.fea_convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(16,16,3,1,1),
                          nn.Conv2d(16,16,3,1,1),
                          nn.Conv2d(16,16,3,1,1)),
            nn.Sequential(nn.Conv2d(16,16,3,1,1),
                          nn.Conv2d(16,16,3,1,1),
                          nn.Conv2d(16,16,3,1,1)),
            nn.Sequential(nn.Conv2d(16,16,3,1,1),
                          nn.Conv2d(16,16,3,1,1),
                          nn.Conv2d(16,16,3,1,1)),
            nn.Sequential(nn.Conv2d(16,16,3,1,1),
                          nn.Conv2d(16,16,3,1,1),
                          nn.Conv2d(16,16,3,1,1))
                          ])
        def get_5_convs():
            return nn.Sequential(nn.Conv2d(1,1,3,1,1),
                          nn.Conv2d(1,1,3,1,1),
                          nn.Conv2d(1,1,3,1,1),
                          nn.Conv2d(1,1,3,1,1),
                          nn.Conv2d(1,1,3,1,1))
        self.lm_convs = nn.ModuleList([get_5_convs() for i in range(68)])

    def forward(self,landmarks,features):
        # the region of the 68 points should be noted
        land_fea = [self.lm_convs[i](landmarks[:,i,:,:].unsqueeze(1)) for i in range(68)]
        # group eye,node,mouth,jaw
        eye_fea = torch.stack(land_fea[17:27],dim=1).sum(axis = 1) + torch.stack(land_fea[36:48],axis=1).sum(axis = 1)
        nose_fea = torch.stack(land_fea[27:36],dim=1).sum(axis = 1)
        jaw_fea = torch.stack(land_fea[:17],dim=1).sum(axis = 1)
        mouth_fea = torch.stack(land_fea[48:68],dim=1).sum(axis = 1)
        # fuse
        eye_fea = self.fea_convs[0](features) * eye_fea.repeat(1,16,1,1)
        nose_fea = self.fea_convs[1](features) * nose_fea.repeat(1,16,1,1)
        jaw_fea = self.fea_convs[2](features) * jaw_fea.repeat(1,16,1,1)
        mouth_fea = self.fea_convs[3](features) * mouth_fea.repeat(1,16,1,1)

        refine_feature = eye_fea+nose_fea+jaw_fea+mouth_fea
    
        return refine_feature
    

class RCNet(nn.Module):
    def __init__(self):
        super(RCNet,self).__init__()
        self.ups = nn.Sequential(
            nn.Conv2d(128,128,3,1,1),
            nn.ReLU(),
            nn.PixelShuffle(2),
            nn.Conv2d(32,64,3,1,1), # the paper atch migh be wrong. pixelshuffle c->c//4
            ResBlock(64),
            ResBlock(64),
            nn.PixelShuffle(2),
            nn.Conv2d(16,32,3,1,1), #
            ResBlock(32),
            ResBlock(32),
            nn.PixelShuffle(2),
            nn.Conv2d(8,16,3,1,1), #
            ResBlock(16),
            ResBlock(16),
        )
        self.share = SharedFeature()
        self.landes = LandmarkEs()
        self.getcoarse = nn.Sequential(
            nn.Conv2d(16,3,3,1,1),
            nn.ReLU()
        )
        self.getrefined = nn.Sequential(
            nn.Conv2d(16,3,3,1,1),
            nn.ReLU()
        )
        self.num_steps=4
        self.com_fusion = ComponentAwa()

    def forward(self,x):
        refine_sr_outs = []
        coarse_sr_outs = []
        heatmap_outs = []
        refined = None
        for step in range(self.num_steps):
            if refined is None:
                x = self.share(x)
            else:
                x = self.share(refined)
            # print(x.shape)
            ldmarks = self.landes(x)
            features = self.ups(x)
            heatmap_outs.append(ldmarks)
            coarse_sr_outs.append(self.getcoarse(features))
            fused_feature = self.com_fusion(ldmarks,features)
            refined =self.getrefined(fused_feature)
            refine_sr_outs.append(refined)

        
        return coarse_sr_outs,refine_sr_outs,heatmap_outs

class getheatmap(nn.Module):
    def __init__(self):
        super(getheatmap,self).__init__()
        
        self.share = SharedFeature()
        self.landes = LandmarkEs()
       
    def forward(self,x):

        x = self.share(x)
        # print(x.shape)
        de,hou,ldmarks = self.landes(x)
        
        return ldmarks
    

class getheatmap_hou(nn.Module):
    def __init__(self):
        super(getheatmap_hou,self).__init__()
        
        # self.share = SharedFeature()
        self.landes = LandmarkEs()
       
    def forward(self,x):

        # x = self.share(x)
        # print(x.shape)
        de,hou,ldmarks = self.landes(x)
        
        return x,ldmarks,de,hou