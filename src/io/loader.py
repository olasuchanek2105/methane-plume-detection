from pathlib import Path
import json
import rasterio
import numpy as np


def find_scene_folders(data_dir: str | Path):
    data_dir = Path(data_dir)
    return [p for p in data_dir.iterdir() if p.is_dir()]


def load_tiff(tiff_path: str | Path) -> np.ndarray:
    tiff_path = Path(tiff_path)
    with rasterio.open(tiff_path) as src:
        img = src.read().astype(np.float32)
    return img


def load_json(json_path: str | Path) -> dict:
    json_path = Path(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta


def load_scene(scene_dir: str | Path):
    scene_dir = Path(scene_dir)
    tiff_path = scene_dir / "response.tiff"
    json_path = scene_dir / "request.json"

    img = load_tiff(tiff_path)
    meta = load_json(json_path)

    return {
        "scene_id": scene_dir.name,
        "image": img,
        "meta": meta,
        "tiff_path": tiff_path,
        "json_path": json_path,
    }