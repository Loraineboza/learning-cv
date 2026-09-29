from batch_norm import BN 
import numpy as np 

bn = BN(num_features=3)

np.random.seed(42)
mock_images = np.random.randn(2, 3, 2, 2) * 100 

output_images = bn.forward(mock_images, training=True)

print(
    f"Исходная форма батча: {mock_images.shape}\n"
    f"Форма на выходе: {output_images.shape}\n"
    f"Значения 1ого канала 1ой картинки после batchnorm: {output_images[0, 0]}\n"
)
