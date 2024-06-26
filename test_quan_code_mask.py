import torch 
from quantize_code_mask_orig import MaskVectorQuantize as OldMaskVectorQuantize
from quantize_code_mask import MaskVectorQuantize
# Define parameters for the MaskVectorQuantize module

kmeans_init = False
use_cosine_sim = False



def test_mask_vector_quantize(kmeans_init=False, use_cosine_sim=False):
    """Tests the MaskVectorQuantize module for consistent results."""
    torch.manual_seed(42) # Fix the random seed for reproducibility
    config = {
        "codebook_size": 512,
        "codebook_dim": 64,
        "kmeans_init": kmeans_init,
        "use_cosine_sim": use_cosine_sim,
        "channel_last": False,
        "accept_image_fmap": True,
        "commitment_beta": 0.25,
        "orthogonal_reg_weight": 0.1,
        "activate_mask_quantize": True,
    }

    vq1 = OldMaskVectorQuantize(**config)
    vq2 = MaskVectorQuantize(**config) # Create a separate instance 

    vq2.load_state_dict(vq1.state_dict()) # Ensure both have the same initial weights 

    batch_size = 2
    channels = config["codebook_dim"]
    height = 16
    width = 16
    dummy_input = torch.randn(batch_size, channels, height, width)
    codebook_mask = torch.ones(batch_size, 1, height, width) 

    # Forward pass for the first instance
    output1, loss1, (_, _, embed_ind1) = vq1(dummy_input.clone(), codebook_mask=codebook_mask.clone())

    # Forward pass for the second instance
    output2, loss2, (_, _, embed_ind2) = vq2(dummy_input.clone(), codebook_mask=codebook_mask.clone())

    # Assertions to check for consistency
    assert torch.allclose(output1, output2, atol=1e-6, rtol=1e-6), "Outputs are different!"
    assert torch.allclose(loss1, loss2, atol=1e-6, rtol=1e-6), "Losses are different!"
    assert torch.equal(embed_ind1, embed_ind2), "Embedding indices are different!"
    print("Test passed successfully! Results are consistent.")

test_mask_vector_quantize(kmeans_init=False, use_cosine_sim=False)