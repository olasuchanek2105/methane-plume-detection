from pathlib import Path

from src.models.sanchez import compute_sanchez
from src.visualisation.sanchez import save_sanchez_preview

from src.io.loader import find_scene_folders, load_scene
from src.visualisation.rgb import save_rgb_preview
from src.visualisation.varon import (
    compute_varon,
    save_varon_preview,
    save_varon_histogram,
    save_varon_center,
    save_anomaly_mask,
    print_varon_stats,
)


def main():
    data_dir = Path("data")
    scene_folders = find_scene_folders(data_dir)

    if not scene_folders:
        print("Nie znaleziono scen.")
        return

    for folder in scene_folders[21:30]:
        scene = load_scene(folder)

        print("Scene ID:", scene["scene_id"])
        print("Shape:", scene["image"].shape)

        out_rgb = Path("outputs/previews") / f"{scene['scene_id']}_rgb.png"
        save_rgb_preview(scene["image"], out_rgb)

        v, c, z = compute_varon(scene["image"])
        print("c =", c)
        print_varon_stats(v)

        save_varon_preview(v, Path("outputs/varon") / f"{scene['scene_id']}_varon.png")
        save_varon_histogram(v, Path("outputs/hist") / f"{scene['scene_id']}_hist.png")
        save_varon_center(v, Path("outputs/center") / f"{scene['scene_id']}_center.png")
        save_anomaly_mask(v, Path("outputs/mask") / f"{scene['scene_id']}_mask.png")

        print("-" * 40)


        S, B12_hat = compute_sanchez(scene["image"])

        save_sanchez_preview(
            S,
            Path("outputs/sanchez") / f"{scene['scene_id']}_sanchez.png"
        )


if __name__ == "__main__":
    main()