import torch
from quant_v2_orig import VectorQuantize2 as VectorQuantizeOriginal 
from quant_v2 import VectorQuantize2 as VectorQuantizeOptimized 

def compare_models(original_model, optimized_model, input_data, atol=1e-6):
    # Run the input data through both models
    original_output, original_loss, original_code = original_model(input_data)
    optimized_output, optimized_loss, optimized_code = optimized_model(input_data)
    
    print(original_output.shape, optimized_output.shape)
    print(original_loss, optimized_loss)


    # Compare the outputs
    outputs_close = torch.allclose(original_output, optimized_output, atol=atol)
    losses_close = torch.allclose(original_loss, optimized_loss, atol=atol)
    codes_close = torch.allclose(original_code[2], optimized_code[2], atol=atol)

    return outputs_close, losses_close, codes_close

def main():
    # Parameters
    codebook_size = 512
    codebook_dim = 64
    batch_size = 8
    height = 16
    width = 16
    channels = codebook_dim
    accept_image_fmap = True

    # Generate random input data
    input_data = torch.randn(batch_size, channels, height, width)

    # Initialize both models with the same parameters
    original_model = VectorQuantizeOriginal(
        codebook_size=codebook_size,
        codebook_dim=codebook_dim,
        accept_image_fmap=accept_image_fmap
    )

    optimized_model = VectorQuantizeOptimized(
        codebook_size=codebook_size,
        codebook_dim=codebook_dim,
        accept_image_fmap=accept_image_fmap
    )



    # Copy the weights from the original model to the optimized model
    optimized_model.codebook.load_state_dict(original_model.codebook.state_dict())

    # Compare the models
    outputs_close, losses_close, codes_close = compare_models(original_model, optimized_model, input_data)

    # Print the comparison results
    print(f"Outputs are close: {outputs_close}")
    print(f"Losses are close: {losses_close}")
    print(f"Codes are close: {codes_close}")

if __name__ == "__main__":
    main()
