from torch.utils.data import DataLoader
import torch
import torch.nn as nn

from src.dataset.patch_dataset import MethanePatchDataset
from src.models.simple_cnn_patch_classifier import SimplePatchCNN

dataset = MethanePatchDataset(
    x_path = "outputs/dataset/X_data.npy",
    y_path = "outputs/dataset/y_data.npy",
    meta_path = "outputs/dataset/meta_data.csv"
)

loader = DataLoader(dataset, batch_size=16, shuffle=True)

x_batch, y_batch = next(iter(loader))

model = SimplePatchCNN()
out = model(x_batch)

criterion = nn.CrossEntropyLoss(),
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("x_batch shape:", x_batch.shape)
print("out shape:", out.shape)
print("out:", out)

preds = torch.argmax(out, dim=1)
print("preds:", preds)
print("y_batch:", y_batch)


