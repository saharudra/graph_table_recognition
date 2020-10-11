from torch_geometric.data import DataLoader

from dataloaders.scitsr import ScitsrDataset
from misc.args import *

scitsr_params = scitsr_params()
print(vars(scitsr_params))
inc = 0
not_optimal = True

# Search for optimal k from following k_val
k_val = [6, 10, 15, 20, 25, 30, 35, 40, 45, 50]

while not_optimal:
    print("Current k value: {}".format(scitsr_params.graph_k))
    train_dataset = ScitsrDataset(scitsr_params)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    all_idx_rel = []
    max_num_nodes = 0
    for idx, data in enumerate(train_dataloader):
        edge_index, rels, y_col, y_row = data.edge_index.numpy(), data.rels.numpy(), data.y_col.numpy(), data.y_row.numpy()
        all_rel = []
        
        # Below logic won't work if number of nodes in the graph are less than graph_k
        if edge_index[:, -1][1] > max_num_nodes:
            max_num_nodes = edge_index[:, -1][1]

        if edge_index[:, -1][1] < scitsr_params.graph_k:
            all_idx_rel.append(1)
        else:
            for rel in rels:
                # For each rel, check whether tuple or reverse of tuple exists in edge_index
                # If it does, check the corresponding label in y_row or y_col
                n0 = rel[0]
                n1 = rel[1]
                edge_type = rel[2]

                # Get the source node id for edge index
                n0_s_idx = n0 * scitsr_params.graph_k
                n1_s_idx = n1 * scitsr_params.graph_k
                
                rel_exists = False
                rel_f_exists = False
                rel_b_exists = False
                for edge in range(scitsr_params.graph_k - 1):
                    try:
                        edge_1 = edge_index[:, n0_s_idx][::-1]
                        edge_2 = edge_index[:, n1_s_idx]
                    except IndexError:
                        print('Index error')
                        print(edge_index)
                        import pdb; pdb.set_trace()
                        
                    # Match node value
                    if edge_1[0] == n0 and edge_1[1] == n1:
                        # rel_f_exists = True
                        # Match edge type
                        if edge_type == 1 and y_row[n0_s_idx] == 1:
                            rel_f_exists = True
                        elif edge_type == 2 and y_col[n0_s_idx] == 1:
                            rel_f_exists = True
                    
                    if edge_2[0] == n0 and edge_2[1] == n1:
                        # rel_b_exists = True
                        if edge_type == 1 and y_row[n1_s_idx] == 1:
                            rel_b_exists = True
                        elif edge_type == 2 and y_col[n1_s_idx] == 1:
                            rel_b_exists = True
                    
                    n0_s_idx += 1
                    n1_s_idx += 1

                if rel_f_exists and rel_b_exists:
                    all_rel.append(rel)
                    rel_exists = True      

            if len(all_rel) == len(rels):
                all_idx_rel.append(1)
    print(len(all_idx_rel), len(train_dataloader))

    if len(all_idx_rel) == len(train_dataloader):
        print("Optimal k reached, value for optimal k: ".format(scitsr_params.graph_k))
        not_optimal = False
    else:
        print("Not the Optimal value for k. Increasing k")
        inc += 1
        scitsr_params.graph_k = k_val[inc]

