"""
Builds a patch dataset for methane plume detection.

The script loads selected scenes, creates two multichannel images, one from
spectral bands and the Sánchez map and second pseudo RGB from Sanchez and Varon, then samples patches around the
known source location.

Patches close to the source are labeled as positive, and patches far
from the source are labeled as negative. The final patches, labels,
and metadata are saved to disk.

A few example patches are also saved for debugging and visual inspection.
"""

import pandas as pd
import numpy as np
import rasterio
import random
from pathlib import Path
import matplotlib.pyplot as plt

from pathlib import Path
from src.io.loader import load_scene


from src.models.sanchez import compute_sanchez_robust
from src.io.loader import load_scene
from src.visualisation.pseudorgb import make_pseudorgb
from src.visualisation.varon import compute_varon
from src.dataset.patch_utils import latlon_to_pixel, sample_patch_centers, extract_patch, label_patch
# from dataset.target_utils import normalize_methane_rate, log_minmax_normalize
from src.preprocessing.scene_features import build_multichannel_image, build_pseudorgb_image


def main():
    patch_size = 256
    positive_radius = 64
    negative_radius = 192


    data_dir = Path("data")
    df = pd.read_csv(data_dir / "final_dataset_2024_stratified.csv")
    df_subset = df.nlargest(10, "methane_rate") #bierzemy sceny z najwieksza emisja metanu, 10 najwiekszych
    df_subset = df.sample(n=20, random_state=42)
    # df_subset = df

    X_list = []
    y_list = []
    meta_list = []
    X_pseudo_list = []


    for _, row in df_subset.iterrows():
        folder_id = row["folder_id"]
        lat = row["json_lat"]
        lon = row["json_lon"]
        methane_rate = row["methane_rate"]

        folder = data_dir / folder_id
        if not folder.exists():
            print("Brak folderu:", folder)
            continue

        channels = ["B6", "B7", "B8", "B11", "SANCHEZ"]
        scene = load_scene(folder)
        X = build_multichannel_image(scene, channels)
        X_pseudo = build_pseudorgb_image(scene)

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
            patch_pseudo = extract_patch(X_pseudo, center_row, center_col, patch_size=patch_size)
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
            
            X_pseudo_list.append(patch_pseudo)

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

    # print("Liczba patchy:", len(X_list))
    # print("Liczba etykiet:", len(y_list))

    if len(X_list) == 0:
        print("Nie zebrano żadnych patchy.")
        return

    X_data = np.stack(X_list)
    y_data = np.array(y_list)
    meta_df = pd.DataFrame(meta_list)

    X_pseudo_data = np.stack(X_pseudo_list)

    # print("X_data shape:", X_data.shape)
    # print("y_data shape:", y_data.shape)
    # print(meta_df["label"].value_counts())

    out_dir = Path("outputs/dataset")

    np.save(out_dir/"X_data.npy", X_data)
    np.save(out_dir/"y_data.npy", y_data)
    meta_df.to_csv(out_dir/"meta_data.csv", index=False)

    np.save(out_dir / "X_pseudo_data.npy", X_pseudo_data)

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