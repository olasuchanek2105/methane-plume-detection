"Only for test if patch_dataset works corectlly"


from src.dataset.patch_dataset import MethanePatchDataset
from torch.utils.data import DataLoader


dataset = MethanePatchDataset(
    x_path="outputs/dataset/X_data.npy",
    y_path="outputs/dataset/y_data.npy",
    meta_path="outputs/dataset/meta_data.csv"
)

loader = DataLoader(dataset, batch_size=16, shuffle=True)

print("Liczba próbek:", len(dataset))

x, y = dataset[0]
print("x shape:", x.shape)
print("x dtype:", x.dtype)
print("y:", y)
print("y dtype:", y.dtype)

print("-------------------------------")

x_batch, y_batch = next(iter(loader))

print("x_batch shape:", x_batch.shape)
print("x_batch dtype:", x_batch.dtype)
print("y_batch shape:", y_batch.shape)
print("y_batch dtype:", y_batch.dtype)
print("y_batch:", y_batch)


# Liczba próbek: 300
# x shape: (5, 256, 256)
# x dtype: float32
# y: tensor(1)
# y dtype: torch.int64