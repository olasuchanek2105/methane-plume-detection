from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.io.loader import load_scene
from src.models.sanchez import compute_sanchez_robust
from src.visualisation.varon import compute_varon
from src.visualisation.pseudorgb import make_pseudorgb


def crop_roi(arr: np.ndarray, center_row: int, center_col: int, half_size: int = 250):
    h, w = arr.shape[:2]

    r0 = max(center_row - half_size, 0)
    r1 = min(center_row + half_size, h)
    c0 = max(center_col - half_size, 0)
    c1 = min(center_col + half_size, w)

    roi = arr[r0:r1, c0:c1]
    return roi, r0, r1, c0, c1


def save_roi_heatmap(arr: np.ndarray, out_path: str | Path, title: str, point_rc=None):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vmin = np.nanpercentile(arr, 2)
    vmax = np.nanpercentile(arr, 98)

    plt.figure(figsize=(6, 6))
    plt.imshow(arr, cmap="coolwarm", vmin=vmin, vmax=vmax)

    if point_rc is not None:
        pr, pc = point_rc
        plt.scatter(pc, pr, s=40, c="yellow", marker="x")

    plt.title(title)
    plt.colorbar()
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_roi_rgb(arr: np.ndarray, out_path: str | Path, title: str, point_rc=None):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.imshow(arr)

    if point_rc is not None:
        pr, pc = point_rc
        plt.scatter(pc, pr, s=40, c="yellow", marker="x")

    plt.title(title)
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_roi_mask(mask: np.ndarray, out_path: str | Path, title: str, point_rc=None):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.imshow(mask, cmap="gray")

    if point_rc is not None:
        pr, pc = point_rc
        plt.scatter(pc, pr, s=40, c="red", marker="x")

    plt.title(title)
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def make_local_mask(S_roi: np.ndarray, percentile: float = 99.5):
    thr = np.nanpercentile(S_roi, percentile)
    mask_roi = S_roi > thr
    return mask_roi, thr


def main():
    data_dir = Path("data")
    csv_path = data_dir / "final_dataset_2024_stratified.csv"
    out_root = Path("outputs/roi_batch")
    out_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    # na start weź 5 rekordów, potem zwiększysz do 10
    # df_subset = df.iloc[235:240]
    df_subset = df.nlargest(5, "methane_rate")
    
    print("Liczba rekordów do analizy:", len(df_subset))

    for idx, row in df_subset.iterrows():
        folder_id = row["folder_id"]
        lat = row["json_lat"]
        lon = row["json_lon"]
        methane_rate = row["methane_rate"]

        print("\n" + "=" * 60)
        print(f"[{idx}] folder_id:", folder_id)
        print("lat/lon:", lat, lon)
        print("methane_rate:", methane_rate)

        folder = data_dir / folder_id
        if not folder.exists():
            print("Brak folderu:", folder)
            continue

        scene = load_scene(folder)

        print("Scene ID:", scene["scene_id"])
        print("Shape:", scene["image"].shape)

        # liczenie metod
        v, c, z = compute_varon(scene["image"])
        S, B12_hat, mask_global = compute_sanchez_robust(scene["image"])
        pseudo = make_pseudorgb(v, S)

        # otwarcie rastra i konwersja lat/lon -> pixel
        tiff_path = scene["tiff_path"]
        with rasterio.open(tiff_path) as ds:
            row_px, col_px = ds.index(lon, lat)

        print("Pixel row/col:", row_px, col_px)

        img = scene["image"][:12].astype(np.float32) / 10000.0
        rgb = np.stack([img[3], img[2], img[1]], axis=-1)
        rgb = np.clip(rgb, 0, 1)

        # ROI
        half_size = 150

        S_roi, r0, r1, c0, c1 = crop_roi(S, row_px, col_px, half_size=half_size)
        pseudo_roi, _, _, _, _ = crop_roi(pseudo, row_px, col_px, half_size=half_size)
        rgb_roi, _, _, _, _ = crop_roi(rgb, row_px, col_px, half_size=half_size)

        # współrzędne punktu wewnątrz ROI
        roi_row = row_px - r0
        roi_col = col_px - c0
        point_rc = (roi_row, roi_col)

        # lokalna maska liczona z ROI, nie z całej sceny
        mask_roi, thr_roi = make_local_mask(S_roi, percentile=99.5)
        print("Local ROI threshold:", thr_roi)
        print("Local ROI mask pixels:", int(mask_roi.sum()))

        # folder wyjściowy per scena
        scene_out = out_root / folder_id
        scene_out.mkdir(parents=True, exist_ok=True)

        save_roi_rgb(
            rgb_roi,
            scene_out / f"{folder_id}_roi_rgb.png",
            title=f"ROI RGB | rate={methane_rate}",
            point_rc=point_rc,
        )

        save_roi_rgb(
            pseudo_roi,
            scene_out / f"{folder_id}_roi_pseudorgb.png",
            title=f"ROI pseudoRGB | rate={methane_rate}",
            point_rc=point_rc,
        )

        save_roi_heatmap(
            S_roi,
            scene_out / f"{folder_id}_roi_sanchez.png",
            title=f"ROI Sanchez | rate={methane_rate}",
            point_rc=point_rc,
        )

        save_roi_mask(
            mask_roi,
            scene_out / f"{folder_id}_roi_mask_local.png",
            title=f"ROI local mask | rate={methane_rate}",
            point_rc=point_rc,
        )

        print("Zapisano do:", scene_out)

    print("\nDONE")
    

if __name__ == "__main__":
    main()