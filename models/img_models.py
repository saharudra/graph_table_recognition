import torch
import torch.nn as nn
import torch.nn.functional as F 

from misc.args import *


class ConvBaseGFTE(nn.Module):
    def __init__(self, params):
        super(ConvBase, self).__init__()
        self.params = params
        self.inc = self.params.inc
        self.nif = self.params.nif
        self.ks = self.params.ks
        self.ps = self.params.ps
        self.ss = self.params.ss

    def conv_layer(self):
        self.conv_base = nn.Sequential(
            nn.Conv2d(self.inc, )
        )


if __name__ == "__main__":
    params = img_model_params()
    print(params)
    
