import torch
import random
import unittest
from VQDC_testing.encoder_triple_new import TripleGrainEncoder as NewTripleGrainEncoder
from DynamicVectorQuantization.modules.dynamic_modules.EncoderTriple import (
    TripleGrainEncoder as OrigTripleGrainEncoder,
)

router_config_orig = {
    "target": "DynamicVectorQuantization.modules.dynamic_modules.RouterTriple.TripleGrainFeatureRouter",
    "params": {
        "num_channels": 256,
        "normalization_type": "group-32",
        "gate_type": "2layer-fc-SiLu",
    },
}


router_config_new = {
    "target": "VQDC_testing.router_triple.TripleGrainFeatureRouter",
    "params": {
        "num_channels": 256,
        "normalization_type": "group-32",
        "gate_type": "2layer-fc-SiLu",
    },
}


encoder_config_orig = {
    "ch": 128,
    "ch_mult": [1, 1, 2, 2, 4, 4],
    "num_res_blocks": 2,
    "attn_resolutions": [8, 16, 32],
    "dropout": 0.0,
    "resamp_with_conv": True,
    "in_channels": 3,
    "resolution": 256,
    "z_channels": 256,
    "router_config": router_config_orig,
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
encoder2 = OrigTripleGrainEncoder(**encoder_config_orig).eval()


with torch.no_grad():
    output1 = encoder1(x, x_entropy)
    output2 = encoder2(x, x_entropy)

    for (idx,key) in enumerate(output1.keys()):
        print(idx)
        try:
            print(output1[key].shape, output2[key].shape)
            torch.testing.assert_close(
                output1[key].max(),
                output2[key].max(),
                rtol=2e-1,
                atol=2e-1,
                msg=f"{output1[key].max()}, {output2[key].max()}!",
            )

            torch.testing.assert_close(
                output1[key].min(),
                output2[key].min(),
                rtol=2e-1,
                atol=2e-1,
                msg=f"{output1[key].min()}, {output2[key].min()}!",

            )
        except AssertionError as e:
            print(f"Test failed on seed {seed_value}!")
            print("===========")
            
            raise e
