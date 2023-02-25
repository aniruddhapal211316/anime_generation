import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict


def Normalize_2D(x, eps=1e-8):
    return x * (x.square().mean(-1) + eps).rsqrt().view(-1, 1)


def PixelNorm(img, eps=1e-8):
    assert len(img.shape) == 4
    img = img - torch.mean(img, (2, 3), True)
    tmp = torch.mul(img, img)  # or x ** 2
    tmp = torch.rsqrt(torch.mean(tmp, (2, 3), True) + eps)
    return img * tmp


def Generate_map_channels(out_res, start_res=4, max_channels=512):
    base_channels = 16 * 1024
    map_channels = dict()
    k = start_res
    while k <= out_res:
        map_channels[k] = min(base_channels // k, max_channels)
        k *= 2
    return map_channels 


class Mapping(nn.Module):
    def __init__(self,
                 z_dim: int,           
                 deep_mapping=8,       
                 normalize=True,       
                 eps=1e-8,             
                 ):
        super().__init__()
        self.dim = z_dim
        self.deep = deep_mapping
        self.normalize = normalize
        self.eps = eps
        self.blocks = []

        for i in range(self.deep):
            self.blocks.append(nn.Sequential(
                nn.Linear(self.dim, self.dim),
                nn.LeakyReLU(0.2),
            ))

            nn.init.xavier_normal_(self.blocks[-1][0].weight.data)
            nn.init.zeros_(self.blocks[-1][0].bias.data)

        self.blocks = nn.ModuleList(self.blocks)

    def forward(self, z):
        if self.normalize:
            z = Normalize_2D(z, self.eps)
        for block in self.blocks:
            z = block(z)
        return z


class Conv2Demod(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 z_dim,
                 kernel_size,
                 stride=1,
                 demod=True,
                 eps=1e-8
                ):
        super().__init__()
        assert stride == 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.demod = demod
        self.z_dim = z_dim
        self.eps = eps
        
        self.weight = nn.Parameter(torch.randn((out_channels, in_channels, kernel_size, kernel_size)))
        self.A = nn.Linear(z_dim, in_channels)
        self.B = nn.Parameter(torch.zeros(out_channels))
        
        nn.init.xavier_normal_(self.weight)


    def forward(self, img, ws):
        b, c, kh, kw = img.shape
        styles = self.A(ws)
        
        w = self.weight.unsqueeze(0)
        w = w * styles.reshape(b, 1, -1, 1, 1)
        dcoefs = (w.square().sum(dim=[2,3,4]) + self.eps).rsqrt()
        w = w * dcoefs.reshape(b, -1, 1, 1, 1)
        
        img = img.reshape(1, -1, kh, kw)
        w = w.reshape(-1, self.in_channels, self.kernel_size, self.kernel_size)
        img = F.conv2d(img, weight=w, stride=self.stride, padding='same', groups=b)
        img = img.reshape(b, -1, kh, kw)
        
        noise = torch.randn(b, 1, kh, kw, device=img.device, dtype=img.dtype)
        img.add_(self.B.view(1, -1, 1, 1) * noise)
        
        return img


class BlockG(nn.Module):
    def __init__(self,
                 res_in: int,          
                 res_out: int,         
                 in_channels: int,     
                 out_channels: int,    
                 rgb_channels: int,    
                 latent_size: int,     
                 is_last=False,        
                 ):
        super().__init__()
        assert res_out == res_in or res_out == 2 * res_in
        self.res_in = res_in
        self.res_out = res_out
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_size = latent_size
        self.is_last = is_last

        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.Conv1 = Conv2Demod(in_channels, in_channels, latent_size, 3, 1)
        self.Conv2 = Conv2Demod(in_channels, out_channels, latent_size, 3, 1)
        self.Act = nn.LeakyReLU(0.2)
        if is_last:
            self.tRGB = nn.Conv2d(out_channels, rgb_channels, 1, 1)
            nn.init.xavier_normal_(self.tRGB.weight.data)
            nn.init.zeros_(self.tRGB.bias.data)
        else:
            self.tRGB = Conv2Demod(out_channels, rgb_channels, latent_size, 1, 1)


    def forward(self, x, rgb, w):
        assert len(x.shape) == 4

        if self.res_out == 2 * self.res_in:
            x = self.up_sample(x)
            rgb = self.up_sample(rgb)

        x = self.Conv1(x, w)
        x = self.Act(x)
        x = self.Conv2(x, w)
        x = self.Act(x)

        if self.is_last:
            rgb.add_(self.Act(self.tRGB(x)))
        else:
            rgb.add_(self.Act(self.tRGB(x, w)))
        return x, rgb


class Generator(nn.Module):
    def __init__(self,
                 res: int,             
                 RGB=True,             
                 deep_mapping=8,       
                 start_res=4,          
                 max_channels=512,     
                 latent_size=512,      
                 normalize=True,       
                 eps=1e-8,             
                 ):
        super().__init__()
        assert 2 ** round(np.log2(res)) == res and res >= 4 and res <= 1024
        self.res = res
        self.out_channels = 3 if RGB else 1
        self.deep_mapping = deep_mapping
        self.latent_size = latent_size
        self.start_res = start_res
        self.eps = eps

        self.map_channels = Generate_map_channels(res, start_res, max_channels)

        self.mapping = Mapping(latent_size, deep_mapping, normalize, eps)
        self.const = nn.Parameter(torch.ones(max_channels, start_res, start_res))
        self.blocks = OrderedDict([
            (f'res {start_res}', BlockG(start_res, start_res, max_channels, self.map_channels[start_res], self.out_channels, latent_size)),
        ])

        to_res = 2 * start_res
        while to_res <= res:
            cur_res = to_res // 2
            in_channels = self.map_channels[cur_res]
            out_channels = self.map_channels[to_res]
            is_last = to_res == res
            self.blocks[f'res {to_res}'] = BlockG(cur_res, to_res, in_channels, out_channels, self.out_channels, latent_size, is_last)
            to_res *= 2

        self.blocks = nn.ModuleDict(self.blocks)

    def forward(self, z):
        w = self.mapping(z)
        img = self.const.expand(w.size(0), -1, -1, -1)
        rgb = torch.zeros((w.size(0), self.out_channels, self.start_res, self.start_res), device=w.device)
        for block in self.blocks.values():
            img, rgb = block(img, rgb, w)
        return rgb

