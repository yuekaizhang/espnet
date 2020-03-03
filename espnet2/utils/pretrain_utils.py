# refer to https://github.com/andi611/Mockingjay-Speech-Representation/blob/3d3c407c9f9bd048feb0ef3753b75dc715d29462/utility/mam.py#L28

#Mockingjay-Speech-Representation/utility/mam.py
# a little bit confused about we should masked feats or encoder downsampling encoder input?

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
    mask_proportion = MASK_PROPORTION
    mask_consecutive = MASK_CONSECUTIVE
    
    
    mask_label = np.zeros_like(spec.detach().cpu()) # by Yuekai
    spec_masked = spec

    mask_label = torch.ByteTensor(mask_label).to(dtype=torch.uint8)

    # so input spec is before trunck? which with different size ? 
    spec_len = np.sum(np.sum(spec.detach().cpu().numpy(), axis=-1) != 0, axis=-1)
    spec_len = [int(sl) for sl in spec_len]


    for idx in range(len(spec)):
        dice = torch.rand(1).data.cpu()
        valid_index_range = int(spec_len[idx] - mask_consecutive - 1) # compute valid len for consecutive masking
        proportion = int(spec_len[idx] * mask_proportion // mask_consecutive)
        chosen_index = torch.randperm(valid_index_range).data.cpu().numpy()[:proportion] # draw `proportion` samples from the range (0, valid_index_range) and without replacement

        # mask to zero 
        if bool(dice < 0.8):
            for i in range(mask_consecutive):
                spec_masked[idx][chosen_index+i] = 0
        elif bool(dice >= 0.8) and bool(dice < 0.9):
            random_index = torch.randperm(valid_index_range).data.cpu().numpy()[:proportion]
            for i in range(mask_consecutive):
                spec_masked[idx][chosen_index+i] = spec_masked[idx][random_index+i]
        else:
            pass
        mask_label[idx][chosen_index] = 1
        spec_masked = spec_masked.to(dtype=torch.float32)
        mask_label = torch.ByteTensor(mask_label).to(dtype=torch.uint8)
        spec = spec.to(dtype=torch.float32)

    return spec_masked, mask_label.cuda()


