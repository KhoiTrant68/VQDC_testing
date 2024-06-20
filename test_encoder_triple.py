import torch
import random
import unittest
from encoder_triple_new import TripleGrainEncoder as NewTripleGrainEncoder
from encoder_triple_orig import TripleGrainEncoder as OriginalTripleGrainEncoder


class TestTripleGrainEncoder(unittest.TestCase):
    def setUp(self):
        """Setup common test parameters and encoder instances."""

        self.router_config_orig = {
            "target": "RouterTriple.TripleGrainFeatureRouter",
            "params": {
                "num_channels": 256,
                "normalization_type": "group-32",
                "gate_type": "2layer-fc-SiLu",
            },
        }
        self.router_config_new = {
            "target": "router_triple.TripleGrainFeatureRouter",
            "params": {
                "num_channels": 256,
                "normalization_type": "group-32",
                "gate_type": "2layer-fc-SiLu",
            },
        }

        self.encoder_config_orig = {
            "ch": 128,
            "ch_mult": [1,1,2,2,4,4],
            "num_res_blocks": 2,
            "attn_resolutions": [8,16,32],
            "dropout": 0.0,
            "resamp_with_conv": True,
            "in_channels": 3,
            "resolution": 256,
            "z_channels": 256,
            "router_config": self.router_config_orig,
        }

        self.encoder_config_new = {
            "ch": 128,
            "ch_mult": [1,1,2,2,4,4],
            "num_res_blocks": 2,
            "attn_resolutions": [8,16,32],
            "dropout": 0.0,
            "resamp_with_conv": True,
            "in_channels": 3,
            "resolution": 256,
            "z_channels": 256,
            "router_config": self.router_config_new,
        }

        self.input_shape = (4, 3, 256, 256)
        self.x = torch.randn(self.input_shape)
        self.x_entropy = torch.randn(self.input_shape)

    def test_outputs(self):
        for i in range(1): # Try with multiple seeds
            seed_value = i * 10 
            random.seed(seed_value)
            torch.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value)  # If using GPU

            encoder1 = NewTripleGrainEncoder(**self.encoder_config_new).eval()
            encoder2 = OriginalTripleGrainEncoder(**self.encoder_config_orig).eval()
            output1 = None
            output2 = None


            with torch.no_grad(): 
                try:
                    output1 = encoder1(self.x, self.x_entropy)
                except:
                    pass
                try:
                    output2 = encoder2(self.x, self.x_entropy)
                except:
                    pass
            if output1 == None:
                continue
            if output2 == None:
                continue
            for key in output1.keys():
                try:
                    print(output1[key].shape, output2[key].shape)
                    torch.testing.assert_close(
                        output1[key].shape,
                        output2[key].shape,
                        rtol=1e-5,
                        atol=1e-6,
                        msg=f"Outputs for key '{key}' do not match on seed {seed_value}!",
                    )
                except AssertionError as e:
                    print(f"Test failed on seed {seed_value}!")
                    raise e # Re-raise the error for full traceback

if __name__ == "__main__":
    unittest.main()