import numpy as np
from sklearn.linear_model import LinearRegression, HuberRegressor
import matplotlib.pyplot as plt
from scipy.ndimage import label
from scipy.ndimage import gaussian_filter


def compute_sanchez_basic(img: np.ndarray):
    # tylko 12 kanałów
    img = img[:12].astype(np.float32)

    # skalowanie
    img = img / 10000.0

    # target (to co chcemy przewidzieć)
    B12 = img[11]

    # features (wszystko oprócz B12)
    X = img[:11]  # (11, H, W)

    # reshape → (pixels, features)
    H, W = B12.shape
    X = X.reshape(11, -1).T   # (N, 11)
    y = B12.flatten()         # (N,)

    # model regresji
    model = LinearRegression()
    model.fit(X, y)

    # predykcja
    y_hat = model.predict(X)

    # reshape z powrotem do obrazu
    B12_hat = y_hat.reshape(H, W)

    # Sánchez (residual)
    eps = 1e-6
    S = (B12 - B12_hat) / (B12_hat + eps)

    return S, B12_hat


def compute_sanchez_robust(img: np.ndarray):
    img = img[:12].astype(np.float32)
    img = img /10000.0

    B12 = img[11]

    band_ids = [5,6,7,8,10]
    X_img = img[band_ids]

    H,W = B12.shape
    X = X_img.reshape(len(band_ids), -1).T
    y = B12.flatten()

    valid = np.all(X > 0, axis=1) & (y > 0)

    valid = valid & (y > 0.02) #wywalamy ciemne piksele

    B8 = img[7].flatten()
    valid = valid & (B8 > 0.05) #sprobowac 0.03 i 0.08

    if valid.sum() < 100:
        raise ValueError("Za mało poprawnych pikseli po maskowaniu")

    y_valid = y[valid]
    lower = np.nanpercentile(y_valid, 5)
    upper = np.nanpercentile(y_valid, 95)

    valid = valid & (y >= lower) & (y <= upper)

    X_train = X[valid]
    y_train = y[valid]

    max_train_samples = 50000

    if len(X_train) > max_train_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(X_train.shape[0], size = max_train_samples, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]

    if len(X_train) < 100:
        raise ValueError("Za mało poprawnych pikseli")
    
    model = HuberRegressor(max_iter=300)

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    

    model.fit(X_train, y_train)

    y_hat = model.predict(X)
    B12_hat =  y_hat.reshape(H,W)

    eps = 1e-6
    B12_hat_safe = np.maximum(B12_hat, 1e-3)
    S = (B12 - B12_hat) / B12_hat_safe
    S = S - np.nanmedian(S)

    S_smooth = gaussian_filter(S, sigma=2)

    thr = np.nanpercentile(S, 99)
    mask = S_smooth > thr

    labeled, num = label(mask)
    sizes = np.bincount(labeled.ravel())
    keep = sizes > 200
    mask_clean = keep[labeled]

    return S, B12_hat, mask_clean