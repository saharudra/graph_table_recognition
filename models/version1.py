import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.nn as nn

from .img_models import ConvBaseGFTE
from ops.sample_image_features import sample_box_features


class TbNetV1(nn.Module):
    """
    Position features are transformed via multiple layers of  
    graph convolutional operator from the “Semi-supervised 
    Classification with Graph Convolutional Networks” paper
    after which edge node pairs are concatenated and passed 
    through a linear + relu block to get position pair features.

    Let's call this node feature pair concatenation followed by
    passiging through a linear + relu block as edge feature generation.

    Text features of the edge are passed through a GRU network
    and then passed through edge feature generation.


    Image is passed through ConvBaseGFT image features are sampled
    from their respective positions and then passed through edge
    feature generation.

    All of the edge features i.e position, text and image are concatenated
    and passed through another linear + relu block follwed by a softmax
    to classify each edge as either a row/not_a_row.

    Add a separate head for calssifying edge featueres concatenated and 
    passed through another linear + relu block followed by a softmax
    to classify each edge as either a col/not_a_col.
    """
    def __init__(self, base_params, img_model_params):
        super(TbNetV1, self).__init__()
        self.base_params = base_params
        self.img_model_params = img_model_params
        self.img_model = ConvBaseGFTE(self.img_model_params)

        # position transformation layer
        self.position_layer = self.position_transformation_layer
        
        # text transformation layer
        self.embeds = nn.Embedding(self.base_params.vocab_size, self.base_params.num_text_features)
        self.rnn = nn.GRU(self.base_params.num_text_features, self.base_params.num_hidden_features, \
                          bidirectional=self.base_params.bidirectional, batch_first=True)

        # edge feature generation layers
        self.lin1 = nn.Linear(self.base_params.num_hidden_features * 2, num_hidden_features)
        self.lin_img = nn.Linear(self.base_params.num_hidden_features * 2, num_hidden_features)
        self.lin_text = nn.Linear(self.base_params.num_hidden_features * 2, num_hidden_features)
        self.lin_final = nn.Linear(self.base_params.num_hidden * 3, self.base_params.num_classes)

    def position_transformation_layer(self):
        out = nn.Sequential(
            GCNConv(self.base_params.num_node_features, self.base_params.num_hidden_features),
            nn.ReLU(inplace=True),
            GCNConv(self.base_params.num_hidden_features, self.base_params.num_hidden_features),
            nn.ReLU(inplace=True)
        )
        return out

    def forward(self, data):
        x, edge_index, xtext, img, nodenum, pos, cell_wh = data.x, data.edge_index, data.xtext, data.img, data.nodenum, data.pos, data.cell_wh

        




        

