## Table Structure Recognition using Graph Neural Networks

##### To-Dos:
- [ ] Dataloaders
    - [ ] S-A PubTabNet, SciTSR, ICDAR 2013
    - [ ] S-B PubTabNet, SciTSR, ICDAR 2013
- [ ] Baselines
    - [ ] Vanilla GFTE multi-task version
        - [ ] Train on SciTSR, test on ICDAR 2013
        - [ ] Train on PubTabNet, test on ICDAR 2013
    - [ ] GFTE with GAT multi-task version
        - [ ] Train on SciTSR, test on ICDAR 2013
        - [ ] Train on PubTabNet, test on ICDAR 2013
    - [ ] Transformer multi-task version
        - [ ] Train on SciTSR, test on ICDAR 2013
        - [ ] Train on PubTabNet, test on ICDAR 2013
 
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

#### Architecture Experiments:
- [ ] Global self-attention
- [ ] Multi-headed self-attention : https://atcold.github.io/pytorch-Deep-Learning/en/week12/12-3/
- [ ] Attention augmented convolution

#### Datasets
* Training: SciTSR, Synthetic dataset
* Validation: SciTSR
    * Transfer learning on ICDAR 2013
* Testing: ICDAR 2013, ICDAR 2019

#### Additions:
* Sampling of cells 
    * How to sample?
    * Get edges only for the sampled cells
* Cartesian Products for reducing double for loop in edge generation
    * Sort Join:
        * Right join in RDBMS literature does this
    * Hash Join