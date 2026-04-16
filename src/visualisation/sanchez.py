from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def save_sanchez_preview(s: np.ndarray, out_path: str | Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vmin = np.nanpercentile(s, 2)
    vmax = np.nanpercentile(s, 98)

    plt.figure(figsize=(8, 8))
    plt.imshow(s, cmap="gray", vmin=vmin, vmax=vmax)
    plt.title("Sanchez")
    plt.colorbar()
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_sanchez_map_preview(s: np.ndarray, out_path: str | Path):

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vmin = np.nanpercentile(s, 2)
    vmax = np.nanpercentile(s, 98)

    print("S min:", np.nanmin(s))
    print("S max:", np.nanmax(s))
    print("S mean:", np.nanmean(s))
    print("S std:", np.nanstd(s))


    plt.figure(figsize=(8, 8))
    plt.imshow(s, cmap="coolwarm", vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title("Sanchez robust")
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()



def save_mask_preview(mask: np.ndarray, out_path: str | Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 8))
    plt.imshow(mask, cmap="gray")
    plt.title("Mask (S > threshold)")
    plt.axis("off")

    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()