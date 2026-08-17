import torch

x = torch.tensor([[1, 2, 3], [4, 5, 6]])

print(f"Количество измерений: {x.dim()}")
print(f"Альтернативный способ: {x.ndim}")  

vector = torch.tensor([1, 2, 3]) 
row_matrix = vector.unsqueeze(0) 
print(f"vector = {vector}; после добавления unsqueeze(0): {row_matrix}")
col_matrix = vector.unsqueeze(1) 
print(f"vector = {vector}; после добавления unsqueeze(1): {row_matrix}\n")

messy_tensor = torch.randn(1, 1, 5) 
print(f"messy_tensor = {messy_tensor}")
clean_tensor = messy_tensor.squeeze() # 
print(f"messy_tensor.squeeze(): {clean_tensor}")
print(f"it was 'dim': {messy_tensor.dim()}, after dim: {clean_tensor.dim()}")