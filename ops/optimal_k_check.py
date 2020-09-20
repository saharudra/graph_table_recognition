from torch_geometric.data import DataLoader

from dataloaders.scitsr import ScitsrDataset
from misc.args import *

scitsr_params = scitsr_params()
print(vars(scitsr_params))
train_dataset = ScitsrDataset(scitsr_params)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)

for idx, data in enumerate(train_dataloader):
    edge_index, rels, y_col, y_row = data.edge_index.numpy(), data.rels.numpy(), data.y_col.numpy(), data.y_row.numpy()

    for rel in rels:
    
    