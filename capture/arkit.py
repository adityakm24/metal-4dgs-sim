"""Loader for iPhone ARKit captures written by apps/iphone (docs/formats.md §2)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np

from engine.metal_splat.camera import Camera

from .dataset import Dataset, Frame

# ARKit camera (+x right, +y up, +z back) -> OpenCV camera (+x right, +y down, +z forward)
ARKIT_CAM_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])
# ARKit world (+y up) -> sim world (+z up): x_sim = -z_ark, y_sim = -x_ark, z_sim = y_ark
ARKIT_WORLD_TO_SIM = np.array([[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)


class ConfidenceFrame(Frame):
    """Frame whose LiDAR depth is masked by ARKit confidence (keeps medium + high by default)."""

    confidence_path: Path | None = None
    min_confidence: int = 1

    def depth(self, size=None):
        d = super().depth(None)
        if d is None:
            return None
        if self.confidence_path is not None and Path(self.confidence_path).exists():
            import cv2

            conf = cv2.imread(str(self.confidence_path), cv2.IMREAD_UNCHANGED)
            if conf is not None and conf.shape == d.shape:
                d = np.where(conf >= self.min_confidence, d, 0.0).astype(np.float32)
        if size is not None and (d.shape[1], d.shape[0]) != tuple(size):
            import cv2

            d = cv2.resize(d, size, interpolation=cv2.INTER_NEAREST)
        return d


def _resolve_root(path: Path) -> Path:
    if path.suffix == ".zip":
        out = path.with_suffix("")
        if not out.exists():
            with zipfile.ZipFile(path) as z:
                z.extractall(out)
        path = out
    if (path / "capture.json").exists():
        return path
    found = sorted(path.glob("*/capture.json"))
    if not found:
        raise FileNotFoundError(f"no capture.json under {path}")
    return found[0].parent


def read_ascii_ply(path: Path) -> np.ndarray:
    lines = path.read_text().splitlines()
    end = next(i for i, l in enumerate(lines) if l.strip() == "end_header")
    data = np.loadtxt(lines[end + 1:], ndmin=2) if len(lines) > end + 1 else np.zeros((0, 3))
    return data[:, :3]


def arkit_pose_to_sim(c2w_arkit: np.ndarray) -> np.ndarray:
    return ARKIT_WORLD_TO_SIM @ np.asarray(c2w_arkit, dtype=np.float64) @ ARKIT_CAM_TO_CV


def load_arkit_capture(path: str | Path, min_confidence: int = 1, colorize_frames: int = 30) -> Dataset:
    root = _resolve_root(Path(path))
    doc = json.loads((root / "capture.json").read_text())
    if doc.get("format") != "m4dgs-arkit-capture":
        raise ValueError(f"{root}/capture.json is not an m4dgs ARKit capture")
    W, H = int(doc["image_width"]), int(doc["image_height"])
    frames = []
    for f in doc["frames"]:
        if f.get("tracking_state", "normal") != "normal":
            continue
        K = np.asarray(f["intrinsics"], dtype=np.float64)
        cam = Camera(W, H, K[0, 0], K[1, 1], K[0, 2], K[1, 2], arkit_pose_to_sim(f["camera_to_world"]))
        fr = ConfidenceFrame(float(f["timestamp"]), cam, root / f["image"], root / f["depth"] if f.get("depth") else None)
        fr.confidence_path = root / f["confidence"] if f.get("confidence") else None
        fr.min_confidence = min_confidence
        frames.append(fr)
    if not frames:
        raise ValueError("capture has no frames with normal tracking")

    points = None
    ply = root / "points.ply"
    if ply.exists():
        xyz_ark = read_ascii_ply(ply)
        if len(xyz_ark):
            xyz = xyz_ark @ ARKIT_WORLD_TO_SIM[:3, :3].T
            rgb = np.full((len(xyz), 3), np.nan)
            for fr in frames[:: max(1, len(frames) // colorize_frames)]:
                img = fr.image()
                uv, z = fr.camera.project(xyz)
                ok = np.isnan(rgb[:, 0]) & (z > 0.05) & (uv[:, 0] >= 0) & (uv[:, 0] < W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < H - 1)
                rgb[ok] = img[uv[ok, 1].astype(int), uv[ok, 0].astype(int)]
            points = np.concatenate([xyz, rgb, np.full((len(xyz), 1), np.nan)], 1)
    print(f"[arkit] {len(frames)} frames, {W}x{H}, lidar={doc.get('has_lidar')}, "
          f"{0 if points is None else len(points)} VIO feature points ({doc.get('device_model')})")
    return Dataset(frames, source="arkit", points=points, root=root)
