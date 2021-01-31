import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

encoder_layers = TransformerEncoderLayer(8, 4)
encoder = TransformerEncoder(encoder_layers, 2)

inp = torch.randn(1, 10, 8)
out = encoder(inp)
print(out.shape)
inp_mlp = torch.randn(64, 16)
mlp = nn.Linear(16, 2)
out_mlp = mlp(inp_mlp)
print(out_mlp.shape)