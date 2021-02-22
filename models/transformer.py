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
"""

class TbTSR(nn.Module):
    def __init__(self, base_params, img_model_params, trainer_params):
        super(TbTSR, self).__init__()
        self.base_params = base_params
        self.img_model_params = img_model_params
        self.trainer_params = trainer_params

        


