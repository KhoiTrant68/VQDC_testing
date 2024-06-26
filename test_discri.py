import torch
import torch.nn as nn
from VQDC_testing.discriminator import NLayerDiscriminator as NLayerDiscriminatorNew  # Import the new Discriminator
from DynamicVectorQuantization.modules.discriminator.model import NLayerDiscriminator as NLayerDiscriminatorOld # Assuming old code is in 'old_discriminator.py'

def test_nlayerdiscriminator_equivalence():
    """Tests if the new NLayerDiscriminator produces the same results as the old implementation."""

    # Test settings
    torch.manual_seed(42)  # For reproducibility
    input_nc = 3  # Example: RGB image
    ndf = 64
    n_layers = 3
    use_actnorm = True  # Test with ActNorm 
    batch_size = 16
    image_size = 64 
    input_shape = (batch_size, input_nc, image_size, image_size)

    # Create instances of both Discriminator versions
    discriminator_old = NLayerDiscriminatorOld(input_nc, ndf, n_layers, use_actnorm)
    discriminator_new = NLayerDiscriminatorNew(input_nc, ndf, n_layers, use_actnorm)

    # Initialize weights of the old model to match the new model (optional but helpful)
    for (name_new, param_new), (name_old, param_old) in zip(discriminator_new.named_parameters(), discriminator_old.named_parameters()):
        if name_new == name_old:
            param_old.data.copy_(param_new.data)

    # Create sample input
    input_tensor = torch.randn(input_shape)

    # Forward pass
    output_old = discriminator_old(input_tensor)
    output_new = discriminator_new(input_tensor)

    # Check if outputs are close
    assert torch.allclose(output_old, output_new, atol=1e-5), "Outputs do not match!"

    print("NLayerDiscriminator implementations produce the same results!")

# Run the test
test_nlayerdiscriminator_equivalence() 