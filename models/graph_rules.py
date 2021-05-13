import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from .img_models import ConvBaseGFTE
from ops.sample_image_features import sample_box_features

class GraphRulesSingleRelationship(nn.Module):
    """
    GraphRulesSingleRelationship processes graphs generated by rules that
    only have a binary classification task on the edges i.e. row-only/
    col-only classification task.
    Separate models will be trained for same-row and same-col classification.
    """
    def __init__(self, base_params, img_model_params):
        super(GraphRulesSingleRelationship, self).__init__()
        self.base_params = base_params
        self.img_model_params = img_model_params

        # position transformation layer
        self.conv1 = GCNConv(self.base_params.num_node_features, self.base_params.num_hidden_features * 2)
        self.conv2 = GCNConv(self.base_params.num_hidden_features * 2, self.base_params.num_hidden_features * 2)
        self.conv3 = GCNConv(self.base_params.num_hidden_features * 2, self.base_params.num_hidden_features)

        # image transformation layer
        self.img_model = ConvBaseGFTE(self.img_model_params)

        # edge feature generation layers
        self.lin_pos = nn.Linear(self.base_params.num_hidden_features * 2, self.base_params.num_hidden_features)
        self.lin_img = nn.Linear(self.base_params.num_hidden_features * 2 * self.base_params.num_samples, self.base_params.num_hidden_features)

        # classification head
        # Using BCEWithLogitsLoss
        self.lin = nn.Sequential(
            nn.Linear(self.base_params.num_hidden_features * 2, self.base_params.num_hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(self.base_params.num_hidden_features, self.base_params.num_hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(self.base_params.num_hidden_features, self.base_params.num_classes)
        )

    def forward(self, data):
        x, edge_index, img, nodenum, pos, cell_wh = data.x, data.edge_index, data.img, data.nodenum, data.pos, data.cell_wh

        # Transform position features
        position_features = F.relu(self.conv1(x, edge_index))
        position_features = F.relu(self.conv2(position_features, edge_index))
        position_features = F.relu(self.conv3(position_features, edge_index))

        # Transform image features
        image_global_features = self.img_model(img)
        image_features = sample_box_features(image_global_features, nodenum, 
                                                  pos)

        # Edge feature generation
        n1_position_features = position_features[edge_index[0]]
        n2_position_features = position_features[edge_index[1]]
        edge_pos_features = torch.cat((n1_position_features, n2_position_features), dim=1)

        n1_image_features = image_features[edge_index[0]]
        n2_image_features = image_features[edge_index[1]]
        edge_image_features = torch.cat((n1_image_features, n2_image_features), dim=1)

        # Transform edge features
        edge_pos_features = F.relu(self.lin_pos(edge_pos_features))
        edge_image_features = F.relu(self.lin_img(edge_image_features))
        edge_features = torch.cat((edge_pos_features, edge_image_features), dim=1)
        edge_pred = F.log_softmax(self.lin(edge_features), dim=1)

        return edge_pred


class GraphRulesMultiClass(nn.Module):
    """
    GraphRulesMultiLabel processes graphs generated by rules that have
    both same-row and same-col label associated with each of the edges
    in a multi-label fashion with a single head performing both the row
    and column classifications.
    Row and Column adjacency matrices are combined at each node level 
    to create the meta graph after finding individual adjacency matrix
    using corresponding rules.
    """
    def __init__(self, base_params):
        super(GraphRulesMultiClass, self).__init__()
        self.base_params = base_params
        self.base_params.num_classes = 3

        # position transformation layer
        self.conv1 = GCNConv(self.base_params.num_node_features, self.base_params.num_hidden_features * 2)
        self.conv2 = GCNConv(self.base_params.num_hidden_features * 2, self.base_params.num_hidden_features * 2)
        self.conv3 = GCNConv(self.base_params.num_hidden_features * 2, self.base_params.num_hidden_features)

        # edge feature generation layers
        self.lin_pos = nn.Sequential(
            nn.Linear(self.base_params.num_hidden_features * 2, self.base_params.num_hidden_features * 2)
        )

        # classification head
        # Using BCEWithLogitsLoss
        self.lin = nn.Sequential(
            nn.Linear(self.base_params.num_hidden_features * 2, self.base_params.num_hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(self.base_params.num_hidden_features, self.base_params.num_hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(self.base_params.num_hidden_features, self.base_params.num_classes)
        )
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Transform position features
        position_features = F.relu(self.conv1(x, edge_index))
        position_features = F.relu(self.conv2(position_features, edge_index))
        position_features = F.relu(self.conv3(position_features, edge_index))

        # Edge feature generation
        n1_position_features = position_features[edge_index[0]]
        n2_position_features = position_features[edge_index[1]]
        edge_pos_features = torch.cat((n1_position_features, n2_position_features), dim=1)

        # Transform edge features
        edge_pos_features = F.relu(self.lin_pos(edge_pos_features))

        # Get edge predictions
        edge_pred = self.lin(edge_pos_features)  # Using nn.CrossEntropyLoss(), no softmax

        return edge_pred

class GraphRulesMultiTask(nn.Module):
    """
    GraphRulesMultiTask processes graphs generated by rules that have
    both same-row and same-col label associated with each of the edges
    in a multi-task fashion where separate heads are used for performing
    row and column adjacency classification tasks.
    Row and Column adjacency matrices are combined at each node level 
    to create the meta graph after finding individual adjacency matrix
    using corresponding rules.
    """
    def __init__(self):
        super(GraphRulesMultiTask, self).__init__()
        pass

if __name__ == '__main__':
    from torch_geometric.data import Dataset, DataLoader
    from dataloaders.scitsr_graph_rules import ScitsrGraphRules
    from misc.args import base_params, scitsr_params

    model = GraphRulesSingleRelationship(base_params())

    train_dataset = ScitsrGraphRules(scitsr_params())
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    for idx, data in enumerate(train_loader):
        row_data, col_data = data
        pred = model(row_data)



