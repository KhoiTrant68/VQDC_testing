import torch
import numpy as np
from DynamicVectorQuantization.modules.dynamic_modules.DecoderPositional import Decoder as OldDecoder  # Import the new Decoder class
from VQDC_testing.decoder import Decoder as NewDecoder  # Assuming the old Decoder is in 'old_decoder.py'

def test_decoder_equivalence():
    """
    Tests whether the new Decoder produces the same output as the old Decoder
    for the same input and parameters.
    """

    # Define decoder parameters
    ch = 64
    in_ch = 128
    out_ch = 3
    ch_mult = (1, 2, 4, 8)
    num_res_blocks = 2
    resolution = 256
    attn_resolutions = [32, 16]
    dropout = 0.1
    resamp_with_conv = True
    give_pre_end = False
    latent_size = 32
    window_size = 4
    position_type = "learned-relative"  # Choose a position type to test

    # Create instances of old and new Decoders
    old_decoder = OldDecoder(ch, in_ch, out_ch, ch_mult, num_res_blocks, resolution, 
                           attn_resolutions, dropout, resamp_with_conv, give_pre_end,
                           latent_size, window_size, position_type)
    new_decoder = NewDecoder(ch, in_ch, out_ch, ch_mult, num_res_blocks, resolution, 
                           attn_resolutions, dropout, resamp_with_conv, give_pre_end,
                           latent_size, window_size, position_type)

    # Initialize weights to be the same (for a fair comparison)
    for old_param, new_param in zip(old_decoder.parameters(), new_decoder.parameters()):
        old_param.data = torch.clone(new_param.data)

    # Create a random input tensor and grain indices
    batch_size = 4
    h = torch.randn(batch_size, in_ch, latent_size, latent_size)
    grain_indices = torch.randint(0, 3, (batch_size, latent_size, latent_size)) # Example indices

    # Get outputs from both decoders
    with torch.no_grad():  # No need to calculate gradients
        old_output = old_decoder(h, grain_indices)
        new_output = new_decoder(h, grain_indices)

    print(old_output.shape, new_output.shape)
    print(old_output.max(), new_output.max())
    print(old_output.min(), new_output.min())

    # Compare the outputs
    assert torch.allclose(old_output, new_output, atol=1e-1), "Outputs are not the same!"
    print("Test passed: Outputs from old and new decoders are the same.")

# Run the test
test_decoder_equivalence()