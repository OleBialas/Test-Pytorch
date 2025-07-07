"""
Building a simple multi-layer perceptron (MLP) with PyTorch
"""

# %%
import torch

# 1. Tensors

# send tenso to GPU in hardware agnostic way
# %%
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data, dtype=int)
print(x_data.device)
if torch.accelerator.is_available():
    x_data = x_data.to(torch.accelerator.current_accelerator())
print(x_data.device)

# PyTorch does not do automatic type conversions!
# %%
agg = x_data.mean(dtype=float)
agg_item = agg.item()  # covert scalar from tensor to builtin type
print(agg, agg_item, type(agg_item))
