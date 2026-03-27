from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def normalize_for_display(arr: np.ndarray, p_low: float = 2, p_high: float = 98) -> np.ndarray:
    arr = arr.astype(np.float32)

    low = np.nanpercentile(arr, p_low)
    high = np.nanpercentile(arr, p_high)

    arr = np.clip(arr, low, high)
    arr = (arr - low) / (high - low + 1e-6)

    return arr


def make_pseudorgb(v: np.ndarray, s: np.ndarray) -> np.ndarray:
    v_norm = normalize_for_display(v)
    s_norm = normalize_for_display(s)

    pseudo = np.stack([v_norm, s_norm, v_norm], axis=-1)
    return pseudo


def save_pseudorgb_preview(pseudo: np.ndarray, out_path: str | Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 8))
    plt.imshow(pseudo)
    plt.title("PseudoRGB [V, S, V]")
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()