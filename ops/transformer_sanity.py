import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

encoder_layers = TransformerEncoderLayer(2, 2)
encoder = TransformerEncoder(encoder_layers, 2)
print(encoder)

inp = torch.randn(1, 18, 2)
out = encoder(inp)
print(out.shape)
inp_mlp = torch.randn(64, 16)
mlp = nn.Linear(16, 2)
out_mlp = mlp(inp_mlp)
print(out_mlp.shape)

weights = torch.tensor([9.8, 68.0, 5.3, 3.5, 10.8, 1.1, 1.4], dtype=torch.float32)
weights = weights / weights.sum()
print(weights)
weights = 1.0 / weights
weights = weights / weights.sum()
print(weights)