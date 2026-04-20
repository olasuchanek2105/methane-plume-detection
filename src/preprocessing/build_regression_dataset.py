from pathlib import Path
import pandas as pd
import numpy as np

from src.io.loader import load_scene
from src.preprocessing.scene_features import build_multichannel_image
from src.dataset.target_utils import log_minmax_normalize


def main():
    data_dir = Path("data")
    out_dir = Path("outputs/scene_dataset")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_dir / "final_dataset_2024_stratified.csv")
    df_subset = df.sample(n=5, random_state=42)   # na start do testów

    X_list = []
    y_list = []
    meta_list = []

    channels = ["B6", "B7", "B8", "B11", "SANCHEZ", "NDMI"]

    for _, row in df_subset.iterrows():
        folder_id = row["folder_id"]
        methane_rate = row["methane_rate"]

        folder = data_dir / folder_id
        if not folder.exists():
            print("Brak folderu:", folder)
            continue

        scene = load_scene(folder)
        X = build_multichannel_image(scene, channels)

        X_list.append(X)
        y_list.append(methane_rate)

        meta_list.append({
            "folder_id": folder_id,
            "methane_rate_raw": methane_rate,
            "height": X.shape[0],
            "width": X.shape[1],
            "channels": X.shape[2],
        })

        print(f"Dodano scenę: {folder_id}, shape={X.shape}, methane_rate={methane_rate}")

    if len(X_list) == 0:
        print("Nie zebrano żadnych scen.")
        return

    first_shape = X_list[0].shape
    same_shapes = all(x.shape == first_shape for x in X_list)

    if not same_shapes:
        print("Sceny mają różne rozmiary. Nie można użyć np.stack bez resize lub crop.")
        for i, x in enumerate(X_list):
            print(f"Scena {i}: shape={x.shape}")
        return

    X_data = np.stack(X_list)
    y_raw = np.array(y_list, dtype=np.float32)
    y_data = log_minmax_normalize(y_raw)

    meta_df = pd.DataFrame(meta_list)
    meta_df["methane_rate_log"] = np.log1p(y_raw)
    meta_df["methane_rate_norm"] = y_data

    np.save(out_dir / "X_scene_data.npy", X_data)
    np.save(out_dir / "y_scene_data.npy", y_data)
    meta_df.to_csv(out_dir / "scene_metadata.csv", index=False)

    print("Zapisano dataset scenowy.")
    print("X_data shape:", X_data.shape)
    print("y_data shape:", y_data.shape)
    print("Zakres y:", y_data.min(), y_data.max())



if __name__ == "__main__":
    main()