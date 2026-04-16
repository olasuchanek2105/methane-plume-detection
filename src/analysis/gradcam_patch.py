"""
Grad-CAM visualization for patch-based methane plume classification.

This script loads a trained CNN model and generates Grad-CAM heatmaps
for selected image patches. It visualizes:
- RGB patch preview (pseudo-RGB from selected bands)
- Sánchez residual map (S)
- Grad-CAM attention map
- Overlay of Sánchez + Grad-CAM
"""


from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import random

from torch.utils.data import DataLoader

from src.dataset.patch_dataset import MethanePatchDataset
from src.models.simple_cnn_patch_classifier import SimplePatchCNN


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_hook = target_layer.register_forward_hook(self.save_activations)
        self.backward_hook = target_layer.register_full_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output.detach()

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, x, class_idx=None):
        self.model.zero_grad()

        logits = self.model(x)

        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1).item()

        score = logits[:, class_idx]
        score.backward()

        grads = self.gradients[0]       # [C, H, W]
        activations = self.activations[0]  # [C, H, W]

        weights = torch.mean(grads, dim=(1, 2))  # [C]

        cam = torch.zeros(
            activations.shape[1:],
            dtype=torch.float32,
            device=activations.device
        )

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)

        if torch.max(cam) > 0:
            cam = cam / torch.max(cam)

        cam = cam.cpu().numpy()
        return cam, logits.detach().cpu()

    def close(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


def make_rgb_preview(x_chw):
    """
    x_chw: (C, H, W)
    Dla wejścia [B06, B07, B08, B11, S]
    tworzymy podgląd:
    R = B11 (idx 3)
    G = B08 (idx 2)
    B = B06 (idx 0)
    """
    x_hwc = np.transpose(x_chw, (1, 2, 0))
    rgb = x_hwc[:, :, [3, 2, 0]]

    rgb = rgb.astype(np.float32)
    rgb_min = np.nanmin(rgb)
    rgb_max = np.nanmax(rgb)
    rgb = (rgb - rgb_min) / (rgb_max - rgb_min + 1e-8)
    rgb = np.clip(rgb, 0, 1)

    return rgb


def make_sanchez_map(x_chw):
    """
    Kanał S (Sánchez) z wejścia [B06, B07, B08, B11, S]
    """
    S = x_chw[4].astype(np.float32)
    vmin = np.nanpercentile(S, 2)
    vmax = np.nanpercentile(S, 98)
    return S, vmin, vmax


def save_gradcam_figure(rgb, cam, S, vmin, vmax, out_path, true_label, pred_label, prob_class1):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(16, 4))

    # Patch preview
    plt.subplot(1, 4, 1)
    plt.imshow(rgb)
    plt.title("Patch preview")
    plt.axis("off")

    # Sánchez
    plt.subplot(1, 4, 2)
    plt.imshow(S, cmap="inferno", vmin=vmin, vmax=vmax)
    plt.title("Sánchez")
    plt.axis("off")

    # Grad-CAM
    plt.subplot(1, 4, 3)
    plt.imshow(cam, cmap="jet")
    plt.title("Grad-CAM")
    plt.axis("off")

    # Overlay Sánchez + Grad-CAM
    plt.subplot(1, 4, 4)
    plt.imshow(S, cmap="inferno", vmin=vmin, vmax=vmax)
    plt.imshow(cam, cmap="jet", alpha=0.4)
    plt.title(
        f"Overlay S+CAM\ntrue={true_label} pred={pred_label} p(class1)={prob_class1:.3f}"
    )
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    x_path = "outputs/dataset/X_data.npy"
    y_path = "outputs/dataset/y_data.npy"
    meta_path = "outputs/dataset/meta_data.csv"
    model_path = "outputs/models/best_patch_cnn.pt"

    out_dir = Path("outputs/gradcam/random")
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = MethanePatchDataset(
        x_path=x_path,
        y_path=y_path,
        meta_path=meta_path,
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = SimplePatchCNN(in_channels=5).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    target_layer = model.features[6]
    gradcam = GradCAM(model, target_layer)

    saved = 0
    max_to_save = 10

    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)

        cam, logits = gradcam.generate(x, class_idx=1)
        probs = torch.softmax(logits, dim=1).numpy()[0]

        pred = int(np.argmax(probs))
        true = int(y.item())
        prob_class1 = float(probs[1])

        # Tylko false negatives
        # if not (true == 1 and pred == 1):
        #     continue

        

        # tylko predykcje = 1
        if pred != 1:
            continue

        # z tych wybierz losowe 10%
        if random.random() > 0.1:
            continue

        x_np = x[0].detach().cpu().numpy()

        rgb = make_rgb_preview(x_np)
        S, vmin, vmax = make_sanchez_map(x_np)

        out_path = out_dir / f"sample_{idx}_true_{true}_pred_{pred}.png"

        save_gradcam_figure(
            rgb=rgb,
            cam=cam,
            S=S,
            vmin=vmin,
            vmax=vmax,
            out_path=out_path,
            true_label=true,
            pred_label=pred,
            prob_class1=prob_class1,
        )

        print(f"Saved: {out_path}")
        saved += 1

        if saved >= max_to_save:
            break

    gradcam.close()
    print("Done.")


if __name__ == "__main__":
    main()