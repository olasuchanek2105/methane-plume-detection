import numpy as np

from src.models.sanchez import compute_sanchez_robust
from src.visualisation.varon import compute_varon
from src.visualisation.pseudorgb import make_pseudorgb
from src.visualisation.ndmi import compute_ndmi

def build_multichannel_image(scene, channels):
    channel_dict = {
        "B6": 5,
        "B7": 6,
        "B8": 7,
        "B11": 10,
        "B12": 11,
    }

    img = scene["image"][:12].astype(np.float32) / 10000.0
    features = []

    for channel in channels:
        if channel in channel_dict:
            band = img[channel_dict[channel]]
            band = np.expand_dims(band, axis=-1)
            features.append(band)

        elif channel == "SANCHEZ":
            S, _, _ = compute_sanchez_robust(scene["image"])
            S = np.expand_dims(S, axis=-1)
            features.append(S)

        elif channel == "NDMI":
            ndmi = compute_ndmi(img)
            ndmi = np.expand_dims(ndmi, axis=-1)
            features.append(ndmi)

        elif channel == "VARON":
            varon, _, _ = compute_varon(scene["image"])
            varon = np.expand_dims(varon, axis=-1)
            features.append(varon)

        else:
            raise ValueError(f"Nieznany kanał: {channel}")

    X = np.concatenate(features, axis=-1)
    return X


def build_pseudorgb_image(scene):
    v, _, _ = compute_varon(scene["image"])
    S, _, _ = compute_sanchez_robust(scene["image"])

    pseudo = make_pseudorgb(v, S)
    pseudo = pseudo.astype(np.float32)

    return pseudo  # (H, W, 3)

