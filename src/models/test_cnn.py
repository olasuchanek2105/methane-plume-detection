from torch.utils.data import DataLoader

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


print("x_batch shape:", x_batch.shape)
print("out shape:", out.shape)
print("out:", out)