from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def save_rgb_preview(img: np.ndarray, out_path: str | Path):
    img = img[:12]

    red = img[3]
    green = img[2]
    blue = img[1]

    rgb = np.stack([red, green, blue], axis=-1).astype(np.float32)

    rgb_min = np.percentile(rgb, 2)
    rgb_max = np.percentile(rgb, 98)
    rgb_scaled = np.clip((rgb - rgb_min) / (rgb_max - rgb_min + 1e-6), 0, 1)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 8))
    plt.imshow(rgb_scaled)
    plt.title("RGB")
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()