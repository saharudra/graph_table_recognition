import numpy as np
import torch

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as pat

import networkx as nx

class GraphRules():
    """
    Rules:
        :Naive Gaussian:
            Apply 1-D Gaussian in horizontal and vertical directions 
            for each of the cell text bounding boxes to find other 
            cell text bounding boxes that could be horizontal or vertically
            linked in same row or column.
            Generates 2 graphs for a given image for each of the row 
            and column rules.

    """
    def __init__(self, params):
        super(GraphRules, self).__init__()
        self.params = params


    def naive_gaussian(self, pos, img):
        """
        There can be multiple ways of implementing rules based on a 1-D Gaussian in horizontal or 
        vertical directions.
        >>> Centroid only:              rules based on 1-D Gaussian with centroid of the cell as mean location
        >>> Top left and bottom right:  rules based on 1-D Gaussian with top left and bottom right
                                        as mean locations
        """
        row_adj_mat = []
        col_adj_mat = []
        if self.params.ng_mean_pos == 'centroid':
            for cell_idx, cell_pos in enumerate(pos):
                x_cp, y_cp = cell_pos[0].item(), cell_pos[1].item()

                align = {}

                for i, align_pos in enumerate(pos):
                    align[i] = {'x': np.exp(-25 * ((x_cp * 1024 - align_pos[0].item() * 1024) / 1024 ) ** 2),
                                'y': np.exp(-80 * ((y_cp * 1024 - align_pos[1].item() * 1024) / 1024 ) ** 2)}
                # Threshold
                # x == 0.1, y == 0.5 for the hyperparameters above
                # For every x above threshold, same col. Similarly for y and row
                # number of nodes in the graph == len(align)
                curr_row_vec = []
                curr_col_vec = []

                for idx, node in enumerate(align):
                    if idx == cell_idx:
                        # Not adding self-loops
                        curr_row_vec.append(0)
                        curr_col_vec.append(0)
                    else:
                        # Add to row or col vectors in adjacency matrix
                        if align[idx]['x'] >= 0.1:
                            curr_col_vec.append(1)
                        else:
                            curr_col_vec.append(0)
                        if align[idx]['y'] >= 0.5:
                            curr_row_vec.append(1)
                        else:
                            curr_row_vec.append(0)

                row_adj_mat.append(curr_row_vec)
                col_adj_mat.append(curr_col_vec)
        
        # Convert list of lists to numpy array
        row_adj_mat = np.array(row_adj_mat)
        col_adj_mat = np.array(col_adj_mat)

        # Visualize adjacency matrices
        self._visualize_adjacency_matrices(row_adj_mat, col_adj_mat, img, pos)
        import pdb; pdb.set_trace()

        # Convert adjacency matrix to edge index for pytorch geometric dataloader
        row_edge_index, col_edge_index = self._adj_mat_2_edge_index(row_adj_mat), self._adj_mat_2_edge_index(col_adj_mat)

        return row_edge_index, col_edge_index


    def _adj_mat_2_edge_index(self, adj_mat):
        """
        Converts adjacency matrix to edge index that can be ingested by
        pytorch geometric dataloader.
        Creating edge indexes for an undirected graph for the TSR problem at 
        hand.
        """
        edge_index = []
        rows, cols = adj_mat.shape
        
        for rid in range(rows):
            for cid in range(cols):
                if adj_mat[rid, cid] == 1:
                    edge_index.append([rid, cid])
        edge_index = torch.from_numpy(np.asarray(edge_index))
        edge_index = edge_index.t().contiguous()

        return edge_index


    def _visualize_adjacency_matrices(self, row_adj_mat, col_adj_mat, img, pos):
        # # Visualize adjacency matrix as an array
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 4))
        ax[0].set_title('Row Adjacency Matrix')
        ax[0].imshow(row_adj_mat, cmap='gray')
        ax[1].set_title('Col Adjacency Matrix')
        ax[1].imshow(col_adj_mat, cmap='gray')
        ax[2].set_title('Table Image')
        ax[2].imshow(img.squeeze(0).permute(1, 2, 0).numpy())

        # # Visualize adjacency matrix directly as a graph
        # rgr = nx.Graph()
        # cgr = nx.Graph()

        # row_rows, row_cols = np.where(row_adj_mat == 1)
        # row_edges = zip(row_rows.tolist(), row_cols.tolist())

        # col_rows, col_cols = np.where(col_adj_mat == 1)
        # col_edges = zip(col_rows.tolist(), col_cols.tolist())
        
        # rgr.add_edges_from(row_edges)
        # cgr.add_edges_from(col_edges)

        # nx.draw(rgr, ax=ax[3])
        # nx.draw(cgr, ax=ax[4])
        plt.show()

        # Visualize adjacency matrix on image
        fig, vax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
        vax[0].set_title('Row Specific Graph')
        vax[1].set_title('Col Specific Graph')
        vax[0].imshow(img.squeeze(0).permute(1, 2, 0).numpy())
        vax[1].imshow(img.squeeze(0).permute(1, 2, 0).numpy())
        if len(pos) == len(row_adj_mat) == len(col_adj_mat) == len(row_adj_mat[0]) == len(col_adj_mat[0]):
            for idx, cell_pos in enumerate(pos):
                # Add circular patch for curr pos
                x_pos, y_pos = (pos[idx][0].item() * 1024, pos[idx][1].item() * 1024)
                vax[0].add_patch(pat.Circle((x_pos, y_pos), 10, facecolor='r', edgecolor='k', fill=True))
                vax[1].add_patch(pat.Circle((x_pos, y_pos), 10, facecolor='r', edgecolor='k', fill=True))
                
                row_connections = row_adj_mat[idx]
                col_connections = col_adj_mat[idx]
                
                for edge_idx, edge in enumerate(row_connections):
                    if edge == 1:
                        # Add circular patch if edge, this might get repeated
                        # Add line between pos_idx and edge_idx
                        x, y = (pos[edge_idx][0].item() * 1024, pos[edge_idx][1].item() * 1024)
                        vax[0].add_patch(pat.Circle((x, y), 10, facecolor='r', edgecolor='k', fill=True))
                        vax[0].plot([x_pos, x], [y_pos, y], color='green')

                for edge_idx, edge in enumerate(col_connections):
                    if edge == 1:
                        # Add circular patch if edge, this might get repeated
                        # Add line between pos_idx and edge_idx
                        x, y = (pos[edge_idx][0].item() * 1024, pos[edge_idx][1].item() * 1024)
                        vax[1].add_patch(pat.Circle((x, y), 10, facecolor='r', edgecolor='k', fill=True))
                        vax[1].plot([x_pos, x], [y_pos, y], color='green')
        else:   
            print('somethings wrong I can feel it')

        plt.show()

