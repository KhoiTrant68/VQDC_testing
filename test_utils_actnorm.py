import torch
import torch.nn as nn
from DynamicVectorQuantization.utils.utils import ActNorm as ActNormOld  # Import the new ActNorm 
from VQDC_testing.utils.utils_modules import ActNorm as ActNormNew # Assuming you have the old code in 'old_actnorm.py'

def test_actnorm_equivalence():
    """Tests if the new ActNorm produces the same results as the old implementation."""

    # Test settings
    torch.manual_seed(0) # For reproducibility
    num_features = 16 
    batch_size = 32
    input_shape = (batch_size, num_features, 8, 8)

    # Create instances of both ActNorm versions
    actnorm_old = ActNormOld(num_features)
    actnorm_new = ActNormNew(num_features)

    # Initialize weights of the old model to match the new model (optional but helpful)
    actnorm_new.loc.data.copy_(torch.randn_like(actnorm_new.loc))
    actnorm_new.scale.data.copy_(torch.randn_like(actnorm_new.scale))

    actnorm_old.loc.data.copy_(actnorm_new.loc)
    actnorm_old.scale.data.copy_(actnorm_new.scale)

    # Create sample input 
    input_tensor = torch.randn(input_shape)

    # Forward pass
    output_old = actnorm_old(input_tensor)
    output_new = actnorm_new(input_tensor)

    # Check if outputs are close 
    assert torch.allclose(output_old, output_new, atol=1e-5), "Forward pass outputs do not match!"

    # Reverse pass (optional - if your use case involves the reverse method)
    output_reverse_old = actnorm_old(output_old, reverse=True)
    output_reverse_new = actnorm_new(output_new, reverse=True)

    assert torch.allclose(output_reverse_old, output_reverse_new, atol=1e-5), "Reverse pass outputs do not match!"

    print("ActNorm implementations match for both forward and reverse passes!")

# Run the test
test_actnorm_equivalence()