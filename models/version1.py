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
        self.position_transformation_layer()
        
        # text transformation layer
        self.embeds = nn.Embedding(self.base_params.vocab_size, self.base_params.num_text_features)
        self.gru = nn.GRU(self.base_params.num_text_features, self.base_params.num_hidden_features, \
                          bidirectional=self.base_params.bidirectional, batch_first=True)

        # edge feature generation layers
        self.lin_pos = nn.Linear(self.base_params.num_hidden_features * 2, num_hidden_features)
        self.lin_img = nn.Linear(self.base_params.num_hidden_features * 2, num_hidden_features)
        self.lin_text = nn.Linear(self.base_params.num_hidden_features * 2, num_hidden_features)

        # row/col classification heads
        self.lin_row = nn.Sequential(
            nn.Linear(self.base_params.num_hidden * 3, self.base_params.num_hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(self.base_params.num_hidden_features, self.base_params.num_classes)
        )

        self.lin_col = nn.Sequential(
            nn.Linear(self.base_params.num_hidden * 3, self.base_params.num_classes),
            nn.ReLU(inplace=True),
            nn.Linear(self.base_params.num_hidden_features, self.base_params.num_classes)
        )

    def position_transformation_layer(self):
        self.position_features = nn.Sequential(
            GCNConv(self.base_params.num_node_features, self.base_params.num_hidden_features),
            nn.ReLU(inplace=True),
            GCNConv(self.base_params.num_hidden_features, self.base_params.num_hidden_features),
            nn.ReLU(inplace=True)
        )
        return self.position_features()

    def forward(self, data):
        x, edge_index, xtext, img, nodenum, pos, cell_wh = data.x, data.edge_index, data.xtext, data.img, data.nodenum, data.pos, data.cell_wh

        # Transform position features
        position_features = self.position_features(x)

        # Transform text features 
        xtext = self.embeds(xtext)
        text_features, _ = self.gru(xtet)
        text_features = text_features[:, -1, :]

        # Transform image features
        image_global_features = self.img_model(img)
        image_features = sample_box_features(image_global_features, nodenum, 
                                                  pos, cell_wh, img, 
                                                  self.base_params.num_samples, self.base_params.div)

        # Edge feature generation
        n1_position_features = position_features[edge_index[0]]
        n2_position_features = position_features[edge_index[1]]
        edge_pos_features = torch.cat((n1_position_features, n2_position_features), dim=1)
        edge_pos_features = F.relu(self.lin_pos(edge_pos_features))


        n1_text_features = text_features[edge_index[0]]
        n2_text_features = text_features[edge_index[1]]
        edge_text_features = torch.cat((n1_text_features, n2_text_features), dim=1)
        edge_text_features = F.relu(self.lin_text(edge_text_features))

        n1_image_features = image_features[edge_index[0]]
        n2_image_features = image_features[edge_index[1]]
        edge_image_features = torch.cat((n1_image_features, n2_image_features), dim=1)
        edge_image_features = F.relu(self.lin_img(edge_image_features))

        # Separate heads for row and col classification
        edge_row_features = torch.cat((edge_pos_features, edge_text_features, edge_image_features), dim=1)
        edge_row_features = self.lin_row(edge_row_features)
        row_pred = F.log_softmax(edge_row_features, dim=1)

        edge_col_features = torch.cat((edge_pos_features, edge_text_features, edge_image_features), dim=1)
        edge_col_features = self.lin_col(edge_col_features)
        col_pred = F.log_softmax(edge_col_features, dim=1)

        return row_pred, col_pred











        

