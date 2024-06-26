from DynamicVectorQuantization.modules.dynamic_modules.DecoderPositional import FourierPositionEmbedding as OrigFourierPositionEmbedding
from VQDC_testing.fourier_embedding import FourierPositionEmbedding as NewFourierPositionEmbedding
import torch 


x = torch.randn(10, 64, 32, 32)
module1 = OrigFourierPositionEmbedding(coord_size=32, hidden_size=64)
module2 = NewFourierPositionEmbedding(coord_size=32, hidden_size=64)

result1 = module1(x)
result2 = module2(x)

with open("a.txt", 'w') as f:

    f.write(f"\n{result1}")
    f.write("=========================================================")
    f.write(f"\n{result2}")

print(result1.shape, result2.shape)
print(result1.max(), result2.max())
print(result1.min(), result2.min())
