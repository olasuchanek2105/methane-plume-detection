import rasterio
import numpy as np

def latlon_to_pixel(tiff_path, lat, lon):
    with rasterio.open(tiff_path) as ds:
        row_px, col_px = ds.index(lon, lat)

    return row_px, col_px

def sample_patch_centers(
    H, W,
    source_row, source_col,
    patch_size=256,
    n_positive=20,
    n_negative=20,
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
