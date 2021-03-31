import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GraphRules(nn.Module):
    """
    Graph generated based on rules.
    
    Rules:
        :Naive Gaussian:
            Apply 1-D Gaussian in horizontal and vertical directions 
            for each of the cell text bounding boxes to find other 
            cell text bounding boxes that could be horizontal or vertically
            linked in same row or column.
            Generates 2 graphs for a given image for each of the row 
            and column rules.

    Number of models invoked depends on the number of graphs generated
    by the rule. Thus this module is agnostic of the rule being used.
    """
    def __init__(self, base_params):
        super(GraphRules, self).__init__()
        self.base_params = base_params

        # position transformation layer
        self.conv1 = GCNConv(self.base_params.num_node_features, self.base_params.num_hidden_features)
        self.conv2 = GCNConv(self.base_params.num_hidden_features, self.base_params.num_hidden_features)

        # classification head
        self.lin = nn.Sequential(
            nn.Linear(self.base_params.num_hidden_features * 2, self.base_params.num_hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(self.base_params.num_hidden_features, self.base_params.num_classes)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Transform position features
        position_features = self.conv1(x, edge_index)
        position_features = F.relu(position_features)
        position_features = self.conv2(position_features, edge_index)
        position_features = F.relu(position_features)

        # Edge feature generation
        n1_position_features = position_features[edge_index[0]]
        n2_position_features = position_features[edge_index[1]]
        edge_pos_features = torch.cat((n1_position_features, n2_position_features), dim=1)
        edge_pos_features = F.relu(self.lin(edge_pos_features))
        import pdb; pdb.set_trace()


if __name__ == '__main__':
    from torch_geometric.data import Dataset, DataLoader
    from dataloaders.scitsr_graph_rules import ScitsrGraphRules
    from misc.args import base_params, scitsr_params

    model = GraphRules(base_params())

    train_dataset = ScitsrGraphRules(scitsr_params())
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    for idx, data in enumerate(train_loader):
        row_data, col_data = data
        pred = model(row_data)



