import torch
import torch.nn as nn
import torch.nn.functional as F 

from misc.args import *


class ConvBaseGFTE(nn.Module):
    def __init__(self, params):
        super(ConvBaseGFTE, self).__init__()
        self.params = params
        self.inc = self.params.inc
        self.nif = self.params.nif
        self.ks = self.params.ks
        self.ps = self.params.ps
        self.ss = self.params.ss
        self.sp = self.params.sp

        self.cnn = self.conv_layer()

    def conv_layer(self):
        conv_base_gfte = nn.Sequential(
            nn.Conv2d(self.inc, self.nif[0], self.ks[0], self.ss[0], self.ps[0]),
            nn.ReLU(True), 
            nn.MaxPool2d(self.sp[0], self.sp[0]),
            nn.Conv2d(self.nif[0], self.nif[1], self.ks[1], self.ss[1], self.ps[1]),
            nn.ReLU(True),
            nn.MaxPool2d(self.sp[1], self.sp[1]),
            nn.Conv2d(self.nif[1], self.nif[2], self.ks[2], self.ss[2], self.ps[2]),
            nn.BatchNorm2d(self.nif[2]),
            nn.ReLU(True),
            nn.MaxPool2d(self.sp[2], self.sp[2])
        )

        return conv_base_gfte

    def forward(self, img):
        out = self.cnn(img)
        return out


if __name__ == "__main__":
    params = img_model_params()
    print(params)
    
