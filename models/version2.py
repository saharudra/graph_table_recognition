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
    Additional GCN on top with and w/o both text and image gcn.
    """
    def __init__(self, base_params, image_model_params, trainer_params):
        super(TbNetV2, self).__init__()
        self.base_params = base_params
        self.image_model_params = image_model_params
        self.trainer_params = trainer_params

        # image transformation layer
        self.img_model = ConvBaseGFTE(self.image_model_params)
        if self.trainer_params.maj_ver == '2' and (self.trainer_params.min_ver == 'b' or self.trainer_params.min_ver == 'c'):
            self.conv_img_1 = GCNConv(self.base_params.num_hidden_features, self.base_params.num_hidden_features)
            self.conv_img_2 = GCNConv(self.base_params.num_hidden_features, self.base_params.num_hidden_features)

        # position transformation layer
        self.conv1 = GCNConv(self.base_params.num_node_features, self.base_params.num_hidden_features)
        self.conv2 = GCNConv(self.base_params.num_hidden_features, self.base_params.num_hidden_features)
        
        # text transformation layer
        self.embeds = nn.Embedding(self.base_params.vocab_size, self.base_params.num_text_features)
        self.gru = nn.GRU(self.base_params.num_text_features, self.base_params.num_hidden_features, \
                          bidirectional=self.base_params.bidirectional, batch_first=True)
        if self.trainer_params.maj_ver == '2' and (self.trainer_params.min_ver == 'a' or self.trainer_params.min_ver == 'c'):
            self.conv_text_1 = GCNConv(self.base_params.num_hidden_features, self.base_params.num_hidden_features)
            self.conv_text_2 = GCNConv(self.base_params.num_hidden_features, self.base_params.num_hidden_features)

        # edge feature generation layers
        self.lin_pos = nn.Linear(self.base_params.num_hidden_features * 2, self.base_params.num_hidden_features)
        self.lin_img = nn.Linear(self.base_params.num_hidden_features * 2 * self.base_params.num_samples, self.base_params.num_hidden_features)
        self.lin_text = nn.Linear(self.base_params.num_hidden_features * 2, self.base_params.num_hidden_features)

        # row/col classification heads
        self.lin_row = nn.Sequential(
            nn.Linear(self.base_params.num_hidden_features * 3, self.base_params.num_hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(self.base_params.num_hidden_features, self.base_params.num_classes)
        )

        self.lin_col = nn.Sequential(
            nn.Linear(self.base_params.num_hidden_features * 3, self.base_params.num_hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(self.base_params.num_hidden_features, self.base_params.num_classes)
        )

    def forward(self, data):
        x, edge_index, xtext, img, nodenum, pos, cell_wh = data.x, data.edge_index, data.xtext, data.img, data.nodenum, data.pos, data.cell_wh

        # Transform position features
        position_features = self.conv1(x, edge_index)
        position_features = F.relu(position_features)
        position_features = self.conv2(position_features, edge_index)
        position_features = F.relu(position_features)

        # Transform text features 
        xtext = self.embeds(xtext)
        text_features, _ = self.gru(xtext)
        # text features sliced
        # text_features = text_features[:, -1, :]
        # text features summed
        text_features = torch.sum(text_features, dim=1)

        # Transform image features
        image_global_features = self.img_model(img)
        image_features = sample_box_features(image_global_features, nodenum, 
                                                  pos, cell_wh, img, 
                                                  self.base_params.num_samples, self.base_params.div)
        
        # Edge feature generation
        # Position
        n1_position_features = position_features[edge_index[0]]
        n2_position_features = position_features[edge_index[1]]
        edge_pos_features = torch.cat((n1_position_features, n2_position_features), dim=1)
        edge_pos_features = F.relu(self.lin_pos(edge_pos_features))

        # Text
        if self.trainer_params.maj_ver == '2' and (self.trainer_params.min_ver == 'a' or self.trainer_params.min_ver == 'c'):
            text_features = F.relu(self.conv_text_1(text_features, edge_index))
            text_features = F.relu(self.conv_text_2(text_features, edge_index))
        n1_text_features = text_features[edge_index[0]]
        n2_text_features = text_features[edge_index[1]]
        edge_text_features = torch.cat((n1_text_features, n2_text_features), dim=1)
        edge_text_features = F.relu(self.lin_text(edge_text_features))

        # Image
        if self.trainer_params.maj_ver == '2' and (self.trainer_params.min_ver == 'b' or self.trainer_params.min_ver == 'c'):
            image_features = F.relu(self.conv_img_1(image_features, edge_index))
            image_features = F.relu(self.conv_img_2(image_features, edge_index))
        n1_image_features = image_features[edge_index[0]]
        n2_image_features = image_features[edge_index[1]]
        edge_image_features = torch.cat((n1_image_features, n2_image_features), dim=1)
        edge_image_features = F.relu(self.lin_img(edge_image_features))

        # Separate heads for row and col classification
        if self.trainer_params.row_only or self.trainer_params.multi_task:
            edge_row_features = torch.cat((edge_pos_features, edge_text_features, edge_image_features), dim=1)
            edge_row_features = self.lin_row(edge_row_features)
            row_pred = F.log_softmax(edge_row_features, dim=1)

        if self.trainer_params.col_only or self.trainer_params.multi_task:
            edge_col_features = torch.cat((edge_pos_features, edge_text_features, edge_image_features), dim=1)
            edge_col_features = self.lin_col(edge_col_features)
            col_pred = F.log_softmax(edge_col_features, dim=1)
        
        # Row-specific, col-specific or both
        if self.trainer_params.multi_task:
            return row_pred, col_pred
        elif self.trainer_params.row_only:
            return row_pred
        elif self.trainer_params.col_only:
            return col_pred
