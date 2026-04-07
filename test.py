from pathlib import Path

from src.models.sanchez import compute_sanchez_basic, compute_sanchez_robust
from src.visualisation.sanchez import save_sanchez_preview, save_sanchez_map_preview, save_mask_preview

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
from src.visualisation.pseudorgb import make_pseudorgb, save_pseudorgb_preview

# def main():
#     data_dir = Path("data")
#     scene_folders = find_scene_folders(data_dir)

#     if not scene_folders:
#         print("Nie znaleziono scen.")
#         return

#     for folder in scene_folders[308:312]:
#         scene = load_scene(folder)

#         print("Scene ID:", scene["scene_id"])
#         print("Shape:", scene["image"].shape)

#         out_rgb = Path("outputs/previews") / f"{scene['scene_id']}_rgb.png"
#         save_rgb_preview(scene["image"], out_rgb)

#         v, c, z = compute_varon(scene["image"])
#         print("c =", c)
#         print_varon_stats(v)

#         save_varon_preview(v, Path("outputs/varon") / f"{scene['scene_id']}_varon.png")
#         save_varon_histogram(v, Path("outputs/hist") / f"{scene['scene_id']}_hist.png")
#         save_varon_center(v, Path("outputs/center") / f"{scene['scene_id']}_center.png")
#         save_anomaly_mask(v, Path("outputs/mask") / f"{scene['scene_id']}_mask.png")
        

#         print("-" * 40)


#         S, B12_hat, mask = compute_sanchez_robust(scene["image"])

#         save_sanchez_preview(
#             S,
#             Path("outputs/sanchez") / f"{scene['scene_id']}_sanchez.png"
#         )
#         save_sanchez_map_preview(
#             S, Path("outputs/sanchez_map") / f"{scene['scene_id']}_sanchez.png"
#         )
#         save_mask_preview(mask,  Path("outputs/mask.png") / f"{scene['scene_id']}_sanchez.png")

#         pseudo = make_pseudorgb(v, S)

#         save_pseudorgb_preview(
#             pseudo,
#             Path("outputs/pseudorgb") / f"{scene['scene_id']}_pseudorgb.png"
#         )

# if __name__ == "__main__":
#     main()


from pathlib import Path

def main():
    data_dir = Path("data")

    top_scenes = [
        "00ba2312cab12f013293d0c63f194223",
        "bcd498d38cea7ccb05e3ec7727333221",
        "9ddd110ce4adbb52fdf323bf22b80993",
        "3600207c4ecd24a4f9d35c7e91cbb2dd",
        "a9327ae740df6ff3dfdd3983fd80eb79",
    ]

    for folder_id in top_scenes:
        folder = data_dir / folder_id

        if not folder.exists():
            print("Brak folderu:", folder)
            continue

        print("\n" + "=" * 50)
        print("Processing:", folder_id)

        scene = load_scene(folder)

        print("Scene ID:", scene["scene_id"])
        print("Shape:", scene["image"].shape)

        # ---------------- RGB ----------------
        out_rgb = Path("outputs/previews") / f"{scene['scene_id']}_rgb.png"
        save_rgb_preview(scene["image"], out_rgb)

        # ---------------- VARON ----------------
        v, c, z = compute_varon(scene["image"])
        print("c =", c)
        print_varon_stats(v)

        save_varon_preview(v, Path("outputs/varon") / f"{scene['scene_id']}_varon.png")
        save_varon_histogram(v, Path("outputs/hist") / f"{scene['scene_id']}_hist.png")
        save_varon_center(v, Path("outputs/center") / f"{scene['scene_id']}_center.png")
        save_anomaly_mask(v, Path("outputs/mask_varon") / f"{scene['scene_id']}_mask.png")

        # ---------------- SANCHEZ ----------------
        S, B12_hat, mask = compute_sanchez_robust(scene["image"])

        save_sanchez_preview(
            S,
            Path("outputs/sanchez") / f"{scene['scene_id']}_sanchez.png"
        )

        save_sanchez_map_preview(
            S,
            Path("outputs/sanchez_map") / f"{scene['scene_id']}_sanchez_map.png"
        )

        save_mask_preview(
            mask,
            Path("outputs/mask_sanchez") / f"{scene['scene_id']}_mask.png"
        )

        # ---------------- PSEUDO RGB ----------------
        pseudo = make_pseudorgb(v, S)

        save_pseudorgb_preview(
            pseudo,
            Path("outputs/pseudorgb") / f"{scene['scene_id']}_pseudorgb.png"
        )

        print("Zrobione:", folder_id)

    print("\nDONE")


if __name__ == "__main__":
    main()