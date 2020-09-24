## Table Structure Recognition using Graph Neural Networks

#### Input:
* Position Features of cell text bounding box
* Text Features of cell text
* Image Features corresponding to cell text bounding box

#### Output:
* Binary Classification of `data.edge_index` for row and column
    * `data.edge_index` contains the edges formed after kNN graph_transform on position features
    * Optimal value of k for kNN transform can be identified to have full coverage to give out Precision and Recall
        * Training might be slow (Optimal k of 30 for SciTSR dataset)

#### Ablations:
* Base architecture without row and col specific attention modules
    * GConv on position features
    * MLP classification head on concatenated position, text and image features
    * Separate row and col classification
* Base architecture without row and col specific attention modules
    * Position features w/wo GConv
    * Additon of position, text and image features passed via GConv prior to MLP classification head
    * Separate row and col classification
* Base architecture with row and col specific attention modules
    * Self-attention module v/s multi-headed self-attention module
    * Position features w/wo GConv
    * Addition of position, text and image features passed via GConv prior to MLP classification head
    * Separate row and col classification

#### Datasets
* Training: SciTSR
* Validation: SciTSR, ICDAR 2013
* Testing: ICDAR 2013, ICDAR 2019