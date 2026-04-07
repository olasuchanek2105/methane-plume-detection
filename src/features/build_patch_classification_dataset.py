import pandas as pd
import numpy as np
import rasterio
import random
from pathlib import Path
import matplotlib.pyplot as plt

from src.models.sanchez import compute_sanchez_robust
from src.io.loader import load_scene




def build_multichannel_image(scene):

    img = scene["image"][:12].astype(np.float32) / 10000.0
    band_ids = [6, 7, 8, 10]

    X = img[band_ids]
    X = np.transpose(X, (1,2,0))

    S, B12_hat, mask = compute_sanchez_robust(scene["image"])
    S = np.expand_dims(S, axis=-1)
    X = np.concatenate([X, S], axis=-1)

    print(X.shape)

    return X


def latlon_to_pixel(tiff_path, lat, lon):
    with rasterio.open(tiff_path) as ds:
        row_px, col_px = ds.index(lon, lat)

    return row_px, col_px

def sample_patch_centers(
    H, W,
    source_row, source_col,
    patch_size=256,
    n_positive=20,
    n_negative=40,
    near_radius=64,
    far_radius=192,
    random_state=42
):
    rng = np.random.default_rng(random_state)
    half = patch_size // 2

    positive_centers = set()
    negative_centers = set()

    max_tries = 5000

    # -------- positive --------
    tries_pos = 0
    while len(positive_centers) < n_positive and tries_pos < max_tries:
        tries_pos += 1

        row = rng.integers(source_row - near_radius, source_row + near_radius + 1)
        col = rng.integers(source_col - near_radius, source_col + near_radius + 1)

        if not (half <= row < H - half and half <= col < W - half):
            continue

        distance = np.hypot(row - source_row, col - source_col)
        if distance <= near_radius:
            positive_centers.add((row, col))

    # -------- negative --------
    tries_neg = 0
    while len(negative_centers) < n_negative and tries_neg < max_tries:
        tries_neg += 1

        row = rng.integers(half, H - half)
        col = rng.integers(half, W - half)

        distance = np.hypot(row - source_row, col - source_col)
        if distance >= far_radius:
            negative_centers.add((row, col))

    centers = list(positive_centers) + list(negative_centers)
    return centers

def extract_patch(X, center_row, center_col, patch_size=256):
    
    H, W, C = X.shape
    half = patch_size // 2
    r0 = center_row - half
    r1 = center_row + half
    c0 = center_col - half
    c1 = center_col + half

    if r0 >= 0 and c0 >= 0 and r1 <= H and c1 <= W:
        patch = X[r0:r1, c0:c1]
    else: return None

    return patch

def label_patch(center_row, center_col, source_row, source_col, positive_radius=64, negative_radius=192):
    distance = (((center_row - source_row)**2) + (center_col - source_col)**2)**0.5


    if distance <= positive_radius: return 1, distance
    if distance >= negative_radius: return 0, distance
    else: return None, distance


from pathlib import Path
from src.io.loader import load_scene

def main():
    patch_size = 256
    positive_radius = 64
    negative_radius = 192


    data_dir = Path("data")
    df = pd.read_csv(data_dir / "final_dataset_2024_stratified.csv")
    df_subset = df.nlargest(5, "methane_rate")

    X_list = []
    y_list = []
    meta_list = []

    for _, row in df_subset.iterrows():
        folder_id = row["folder_id"]
        lat = row["json_lat"]
        lon = row["json_lon"]
        methane_rate = row["methane_rate"]

        folder = data_dir / folder_id
        if not folder.exists():
            print("Brak folderu:", folder)
            continue

        scene = load_scene(folder)
        X = build_multichannel_image(scene)

        tiff_path = scene["tiff_path"]
        source_row, source_col = latlon_to_pixel(tiff_path, lat, lon)

        centers = sample_patch_centers(
            H=X.shape[0],
            W=X.shape[1],
            source_row=source_row,
            source_col=source_col,
            patch_size=patch_size,
            n_positive=20,
            n_negative=40,
            near_radius=positive_radius,
            far_radius=negative_radius,
            random_state=42
        )

        for center_row, center_col in centers:
            patch = extract_patch(X, center_row, center_col, patch_size=patch_size)
            label, distance = label_patch(
                center_row, center_col,
                source_row, source_col,
                positive_radius=positive_radius,
                negative_radius=negative_radius
            )

            if patch is None or label is None:
                continue

            X_list.append(patch)
            y_list.append(label)

            meta_list.append({
                "folder_id": folder_id,
                "methane_rate": methane_rate,
                "source_row": source_row,
                "source_col": source_col,
                "center_row": center_row,
                "center_col": center_col,
                "distance": distance,
                "label": label,
            })

    print("Liczba patchy:", len(X_list))
    print("Liczba etykiet:", len(y_list))

    if len(X_list) == 0:
        print("Nie zebrano żadnych patchy.")
        return

    X_data = np.stack(X_list)
    y_data = np.array(y_list)
    meta_df = pd.DataFrame(meta_list)

    print("X_data shape:", X_data.shape)
    print("y_data shape:", y_data.shape)
    print(meta_df["label"].value_counts())

    out_dir = Path("outputs/dataset")

    np.save(out_dir/"X_data.npy", X_data)
    np.save(out_dir/"y_data.npy", y_data)
    meta_df.to_csv(out_dir/"meta_data.csv", index=False)

    debug_dir = Path("outputs/debug_patches")
    debug_dir.mkdir(parents=True, exist_ok=True)

    saved_pos = 0
    saved_neg = 0
    max_per_class = 5

    for i in range(len(X_list)):
        patch = X_list[i]
        label = y_list[i]
        meta = meta_list[i]

        if label == 1 and saved_pos >= max_per_class:
            continue
        if label == 0 and saved_neg >= max_per_class:
            continue

        rgb = patch[:, :, [4, 2, 0]]  # B11, B08, B06
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)

        folder_id = meta["folder_id"]
        plt.imsave(
            debug_dir / f"patch_{i}_folder_{folder_id}_label_{label}.png",
            rgb
        )

        if label == 1:
            saved_pos += 1
        elif label == 0:
            saved_neg += 1

        if saved_pos >= max_per_class and saved_neg >= max_per_class:
            break

    print("Zapisano dataset.")


if __name__ == "__main__":
    main()