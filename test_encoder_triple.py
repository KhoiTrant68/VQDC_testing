import torch
import random
import unittest
from encoder_triple_new import TripleGrainEncoder as NewTripleGrainEncoder


router_config_new = {
    "target": "router_triple.TripleGrainFeatureRouter",
    "params": {
        "num_channels": 256,
        "normalization_type": "group-32",
        "gate_type": "2layer-fc-SiLu",
    },
}


encoder_config_new = {
    "ch": 128,
    "ch_mult": [1, 1, 2, 2, 4, 4],
    "num_res_blocks": 2,
    "attn_resolutions": [8, 16, 32],
    "dropout": 0.0,
    "resamp_with_conv": True,
    "in_channels": 3,
    "resolution": 256,
    "z_channels": 256,
    "router_config": router_config_new,
}

input_shape = (4, 3, 256, 256)
x = torch.randn(input_shape)
x_entropy = torch.randn(input_shape)

seed_value = 2024
random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)  # If using GPU

encoder1 = NewTripleGrainEncoder(**encoder_config_new).eval()


with torch.no_grad():
    try:
        output1 = encoder1(x, x_entropy)
    except:
        pass

