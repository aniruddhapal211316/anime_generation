import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict


def Generate_map_channels(out_res, start_res=4, max_channels=512):
    base_channels = 16 * 1024
    map_channels = dict()
    k = start_res
    while k <= out_res:
        map_channels[k] = min(base_channels // k, max_channels)
        k *= 2
    return map_channels 


class BlockD(nn.Module):
    def __init__(self,
                 res_in: int,         
                 res_out: int,        
                 in_channels: int,    
                 out_channels: int,   
                 ):
        super().__init__()
        self.res_in = res_in
        self.res_out = res_out
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.Conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.Conv2 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        self.Skip = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.Act = nn.LeakyReLU(0.2)
        
        nn.init.xavier_normal_(self.Conv1.weight.data)
        nn.init.zeros_(self.Conv1.bias.data)
        nn.init.xavier_normal_(self.Conv2.weight.data)
        nn.init.zeros_(self.Conv2.bias.data)
        nn.init.xavier_normal_(self.Skip.weight.data)
        nn.init.zeros_(self.Skip.bias.data)

    def forward(self, x):
        y = self.Skip(x)
        y = self.Act(y)
        
        x = self.Conv1(x)
        x = self.Act(x)
        
        x = self.Conv2(x)
        x = self.Act(x)
        return x + y


class Discriminator(nn.Module):
    def __init__(self,
                 res,                  
                 RGB=True,             
                 last_res=4,           
                 max_channels=512,     
                 ):
        super().__init__()
        assert 2 ** round(np.log2(res)) == res
        self.res = res
        self.in_channels = 3 if RGB else 1
        self.blocks = OrderedDict()

        self.map_channels = Generate_map_channels(res, last_res, max_channels)

        to_res = res // 2
        while to_res >= last_res:
            cur_res = 2 * to_res
            in_channels = self.map_channels[cur_res]
            out_channels = self.map_channels[to_res]
            self.blocks[f'res {cur_res}'] = BlockD(cur_res, to_res, in_channels, out_channels)
            to_res //= 2

        self.fRGB = nn.Conv2d(self.in_channels, self.map_channels[res], 1, 1)
        self.Linear = nn.Linear(self.map_channels[last_res] * last_res ** 2, 1)

        nn.init.xavier_normal_(self.Linear.weight.data)
        nn.init.zeros_(self.Linear.bias.data)
        nn.init.xavier_normal_(self.fRGB.weight.data)
        nn.init.zeros_(self.fRGB.bias.data)

        self.blocks = nn.ModuleDict(self.blocks)

    def forward(self, img):
        assert img.shape[1: ] == (self.in_channels, self.res, self.res)
        
        img = self.fRGB(img)
        for block in self.blocks.values():
            img = block(img)
        return self.Linear(img.view(img.size(0), -1))
