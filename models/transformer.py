"""
Transformer based Table Structure Recognition (TbTSR)
Setting up architecture as depicted in /architecture/transformer_table_arch.png.
Position features are directly being obtained from the dataloader.

The image processing pipeline before sampling N features and passing through
transformer is optional i.e. using CNN of any kind is optional. Initially setting
up with using ResNet18. 
For the layer of ResNet from which the features to take, try from layer 2 onwards.
layer 2 features shape: 512 x 128 x 128
layer 3 features shape: 1024 x 64 x 64
layer 4 features shape: 2048 x 32 x 32

Try with resnet variants as well as resnet with fpn.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from models.modules import ISAB

from models.resnet import resnet18, resnet50, resnext50_32x4d, wide_resnet50_2
from ops.sample_image_features import sample_box_features
from ops.misc import pairwise_combinations


class TbTSR(nn.Module):
    def __init__(self, base_params, img_model_params, trainer_params):
        super(TbTSR, self).__init__()
        self.base_params = base_params
        self.img_model_params = img_model_params
        self.trainer_params = trainer_params
        
        # Define image processing module
        self.img_model = self.get_img_model(self.img_model_params)

        # Define transformer encoder module
        # self.num_expected_features = ((self.base_params.num_hidden_features // 2) \
        #                         * (2 ** self.img_model_params.resnet_out_layer)) \
        #                         * self.base_params.num_samples \
        #                         + self.base_params.num_pos_features
        self.num_expected_features = self.base_params.num_pos_features

        if self.base_params.transformer:
            self.encoder_layers = TransformerEncoderLayer(self.num_expected_features, self.base_params.num_attn_heads)
            self.encoder = TransformerEncoder(self.encoder_layers, self.base_params.num_encoder_layers, self.base_params.transformer_norm) 
        elif self.base_params.set_transformer:
            self.encoder = nn.Sequential(
                ISAB(self.num_expected_features, self.base_params.num_set_hidden_features,
                     self.base_params.num_set_attn_heads, self.base_params.num_set_inds)
                # ISAB(self.base_params.num_set_hidden_features, self.base_params.num_set_hidden_features,
                #      self.base_params.num_set_attn_heads, self.base_params.num_set_inds)
            )   

        # row/col classification heads
        self.lin_row = nn.Sequential(
            nn.Linear(self.num_expected_features * 2, self.base_params.num_classes),
            # nn.ReLU(inplace=True),
            # nn.Linear(self.base_params.num_hidden_features, self.base_params.num_classes)
        )

        self.lin_col = nn.Sequential(
            nn.Linear(self.num_expected_features * 2, self.base_params.num_classes),
            # nn.ReLU(inplace=True),
            # nn.Linear(self.base_params.num_hidden_features, self.base_params.num_classes)
        )

    def get_img_model(self, img_model_params):
        """
        TODO: Add FPN with forward and backward feature concatenation with ResNet
        """
        if self.img_model_params.resnet:
            if self.img_model_params.resnet_model == 'resnet18':
                img_model = resnet18(pretrained=self.img_model_params.resnet_pretrained)
            elif self.img_model_params.resnet_model == 'resnet50':
                img_model = resnet50(pretrained=self.img_model_params.resnet_pretrained)
            elif self.img_model_params.resnet_model == 'resnext50_32x4d':
                img_model = resnext50_32x4d(pretrained=self.img_model_params.resnet_pretrained)
            elif self.img_model_params.resnet_model == 'wide_resnet50_2':
                img_model = wide_resnet50_2(pretrained=self.img_model_params.resnet_pretrained)

        return img_model

    def get_img_features(self, img):
        if self.img_model_params.resnet:
            img_features = self.img_model(img)
            return img_features

    def forward(self, data):
        # x, img, pos, nodenum, cell_wh = data.x, data.img, data.pos, data.nodenum, data.cell_wh
        x, pos = data['x'], data['pos']
        # import pdb; pdb.set_trace()

        # Get image features
        # img_features_global = self.get_img_features(img)
        # img_features_sampled = sample_box_features(img_features_global, nodenum,
        #                                     pos, cell_wh, img, 
        #                                     self.base_params.num_samples,
        #                                     self.base_params.div,
        #                                     self.trainer_params.device)

        # Both x and pos can be passed
        # pos_img_features = torch.cat((pos, img_features_sampled), dim=1)
        # pos_img_features = pos_img_features.unsqueeze(0)
        # Batched processing, uncomment below for singular
        # pos_features = pos.unsqueeze(0)

        # Transform image and position features
        # transformed_features = self.encoder(pos_img_features)
        # print(pos_features.shape)
        transformed_features = self.encoder(pos_features)

        # Concatenate transformed features 
        # Batched processing, uncomment below for singular
        # transformed_features = transformed_features.squeeze(0)
        paired_transformed_features = pairwise_combinations(transformed_features, batched=self.trainer_params.batched)
        
        # Row and Col classification  
        row_pred = self.lin_row(paired_transformed_features)
        col_pred = self.lin_col(paired_transformed_features)
        import pdb; pdb.set_trace()

        return row_pred, col_pred


if __name__ == '__main__':
    from torch_geometric.data import DataLoader

    from misc.args import *
    from dataloaders.scitsr_transformer import ScitsrDatasetSB

    base_params = base_params()
    img_model_params = img_model_params()
    trainer_params = trainer_params()

    dataloader_params = scitsr_params()
    train_dataset = ScitsrDatasetSB(dataloader_params)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    model = TbTSR(base_params, img_model_params, trainer_params)

    for idx, data in enumerate(train_loader):
        row_logits, col_logits = model(data)
        import pdb; pdb.set_trace()



            



    


