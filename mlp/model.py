# %%
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from load_data import test_dataloader, train_dataloader, labels_map, plot_random_images
%load_ext autoreload
%autoreload 2

# Test importing worked
#%%
plot_random_images()

#%%
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# %%
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
# %%
model = NeuralNetwork().to(device)
print(model)
# %%
X, _train_labels = next(iter(train_dataloader))
X = X.to(device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")
# %%
print(pred_probab)
# %%
loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_probab.to('cpu'), _train_labels)
loss
