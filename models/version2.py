import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from .img_models import ConvBaseGFTE
from ops.sample_image_features import sample_box_features

class TbNetV2(nn.Module):
    """
    Row/Col/Multi:
    Version 2.a)
    Text features go through their respective graph convolutions before 
    edge feature generation.

    Version 2.b)
    Both text features and image features go through their respective 
    graph convolutions before edge feature generation.

    Version 2.c)
    Additional GCN on top
    """
    def __init__(self, base_params, image_model_params, trainer_params):
        super(TbNetV2, self).__init__()
        self.base_params = base_params
        self.image_model_params = image_model_params
        self.trainer_params = trainer_params

        self.img_model = ConvBaseGFTE(self.img_model_params)

        # position transformation layer
        self.conv1 = GCNConv(self.base_params.num_node_features, self.base_params.num_hidden_features)
        self.conv2 = GCNConv(self.base_params.num_hidden_features, self.base_params.num_hidden_features)
        
        # text transformation layer
        self.embeds = nn.Embedding(self.base_params.vocab_size, self.base_params.num_text_features)
        self.gru = nn.GRU(self.base_params.num_text_features, self.base_params.num_hidden_features, \
                          bidirectional=self.base_params.bidirectional, batch_first=True)
        self.conv_text_1 = GCNConv(self.base_params.num_hidden_features, self.base_params.num_hidden_features)
        self.conv_text_2 = GCNConv(self.base_params.num_hidden_features, self.base_params.num_hidden_features)

        # edge 