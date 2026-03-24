from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter


def compute_varon(img: np.ndarray):
    # bierzemy tylko pierwsze 12 kanałów, bo 13. kanał był pusty
    img = img[:12].astype(np.float32)

    # wybieramy pasma B11 i B12
    b11 = img[10] / 10000.0  # skalowanie wartości
    b12 = img[11] / 10000.0

    # mała stała, żeby nie dzielić przez zero
    eps = 1e-6

    # skalowanie B12 tak, żeby było w podobnej skali do B11, tylko dla srodka obrazu
    h, w = b11.shape

    b11_center = b11[h//4:3*h//4, w//4:3*w//4]
    b12_center = b12[h//4:3*h//4, w//4:3*w//4]

    c = np.sum(b12_center * b11_center) / (np.sum(b12_center * b12_center) + 1e-6)

    # liczenie Varon ratio
    V = (c * b12 - b11) / (b11 + eps)

    mean = uniform_filter(V, size=50)
    mean_sq = uniform_filter(V**2, size=50)
    std = np.sqrt(mean_sq - mean**2)

    z = (V - mean) / (std + 1e-6)

    return V, c, z


def save_anomaly_mask(z: np.ndarray, out_path: str | Path):
    # liczy wartość powyżej której leży top 1% największych wartości V
    # czyli szukamy najbardziej ekstremalnych pikseli
    # robimy maskę ekstremalnych pikseli


    mask = np.abs(z) > 3

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.imshow(mask, cmap="gray")
    plt.title("Anomaly mask")
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_varon_histogram(v: np.ndarray, out_path: str | Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.hist(v.flatten(), bins=100)
    plt.title("Histogram V")
    plt.xlabel("V values")
    plt.ylabel("Count")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_varon_center(v: np.ndarray, out_path: str | Path):
    # pobieramy wysokość i szerokość obrazu V
    h, w = v.shape

    # wycinamy środkową część obrazu
    center = v[
        h // 4: 3 * h // 4,
        w // 4: 3 * w // 4
    ]

    # lepsze skalowanie tylko dla środka
    vmin = np.percentile(center, 2)
    vmax = np.percentile(center, 98)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.imshow(center, cmap="gray", vmin=vmin, vmax=vmax)
    plt.title("Varon center")
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_varon_preview(v: np.ndarray, out_path: str | Path):
    # zapis pełnego obrazu Varon ratio
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vmin = np.percentile(v, 2)
    vmax = np.percentile(v, 98)

    plt.figure(figsize=(8, 8))
    plt.imshow(v, cmap="gray", vmin=vmin, vmax=vmax)
    plt.title("Varon ratio")
    plt.colorbar()
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def print_varon_stats(v: np.ndarray):
    print("V min:", v.min())
    print("V max:", v.max())
    print("V mean:", v.mean())