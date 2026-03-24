from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def save_sanchez_preview(s: np.ndarray, out_path: str | Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vmin = np.percentile(s, 2)
    vmax = np.percentile(s, 98)

    plt.figure(figsize=(8, 8))
    plt.imshow(s, cmap="gray", vmin=vmin, vmax=vmax)
    plt.title("Sanchez")
    plt.colorbar()
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()