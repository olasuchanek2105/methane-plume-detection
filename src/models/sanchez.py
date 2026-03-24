import numpy as np
from sklearn.linear_model import LinearRegression


def compute_sanchez(img: np.ndarray):
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