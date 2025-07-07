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

# %%
# In-place operations Operations that store the result into the operand are called in-place. They are denoted by a _ suffix. For example: x.copy_(y), x.t_(), will change x.
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")
# %%
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")
# %%
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
# %%
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
# %%
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
# %%
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
# %%
# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
# %%
print(train_features.shape)
print(train_labels.shape)
# %%
