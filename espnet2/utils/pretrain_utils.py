# refer to https://github.com/andi611/Mockingjay-Speech-Representation/blob/3d3c407c9f9bd048feb0ef3753b75dc715d29462/utility/mam.py#L28

#Mockingjay-Speech-Representation/utility/mam.py


import torch
import numpy as np



MASK_PROPORTION = 0.15
MASK_CONSECUTIVE = 1 # 7







def process_train_MAM_data(spec,config=None):
    """
    input: feats i.e. fbank (Batch,Length,Dim)
    output: feats_masked: feats with the selected frames processed \
            as masked frames during training (Batch,Length,Dim)
            mask_label: (Batch,Length) with indices selected in [1, 0], 1 means masked feature


    """

    mask_label = np.zeros_like(spec)
    spec_masked = spec

    mask_label = torch.ByteTensor(mask_label).to(dtype=torch.uint8)



    return spec_masked, mask_label


class MockingjayForMaskedAcousticModel():
    
    # Mockingjay-Speech-Representation/mockingjay/model.py 
    return None

