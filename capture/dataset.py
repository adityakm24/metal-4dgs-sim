"""Common in-memory dataset + the generic on-disk log format (docs/formats.md §5).

Every source (iPhone ARKit capture, nuScenes, Waymo, synthetic) is converted to a `Dataset`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from engine.metal_splat.camera import Camera
from engine.scene_graph.tracks import Track


@dataclass
class Frame:
    t: float
    camera: Camera
    image_path: Path
    depth_path: Path | None = None        # metric depth (.npy float32 metres, or 16-bit PNG millimetres)
    _image: np.ndarray | None = field(default=None, repr=False)

    def image(self, size: tuple[int, int] | None = None) -> np.ndarray:
        """RGB float32 in [0,1], optionally resized to (width, height)."""
        if self._image is None:
            bgr = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(self.image_path)
            self._image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = self._image
        if size is not None and (img.shape[1], img.shape[0]) != tuple(size):
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        return img

    def depth(self, size: tuple[int, int] | None = None) -> np.ndarray | None:
        if self.depth_path is None:
            return None
        p = Path(self.depth_path)
        if p.suffix == ".npy":
            d = np.load(p).astype(np.float32)
        else:
            d = cv2.imread(str(p), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        if size is not None and (d.shape[1], d.shape[0]) != tuple(size):
            d = cv2.resize(d, size, interpolation=cv2.INTER_NEAREST)
        return d


@dataclass
class ActorAnnotation:
    name: str
    track: Track
    size: list[float]
    category: str = "vehicle.car"
    visible_range: tuple[float, float] | None = None


@dataclass
class Dataset:
    frames: list[Frame]
    source: str = "unknown"
    ego_track: Track | None = None
    camera_to_ego: np.ndarray | None = None
    actors: list[ActorAnnotation] = field(default_factory=list)
    points: np.ndarray | None = None          # (M,7) xyz rgb t   (t = observation time, NaN if unknown)
    background_color: tuple[float, float, float] | None = None
    root: Path | None = None

    @property
    def time_range(self) -> tuple[float, float]:
        ts = [f.t for f in self.frames]
        return float(min(ts)), float(max(ts))

    def split(self, every: int = 8) -> tuple[list[Frame], list[Frame]]:
        """Hold out every `every`-th frame for evaluation (0 = no hold-out)."""
        if every <= 0:
            return self.frames, []
        train = [f for i, f in enumerate(self.frames) if i % every != every - 1]
        test = [f for i, f in enumerate(self.frames) if i % every == every - 1]
        return train, test


# ------------------------------------------------------------------------------ generic log io
def save_log(ds: Dataset, root: str | Path) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    frames = []
    for f in ds.frames:
        c = f.camera
        frames.append({
            "t": f.t,
            "image": str(Path(f.image_path).relative_to(root)),
            "depth": str(Path(f.depth_path).relative_to(root)) if f.depth_path else None,
            "camera_to_world": c.c2w.tolist(),
            "intrinsics": c.K.tolist(),
            "width": c.width, "height": c.height,
        })
    doc = {
        "format": "m4dgs-log", "version": 1, "source": ds.source, "frames": frames,
        "ego_track": ds.ego_track.to_json() if ds.ego_track is not None else None,
        "camera_to_ego": ds.camera_to_ego.tolist() if ds.camera_to_ego is not None else None,
        "actors": [{"name": a.name, "category": a.category, "size": list(map(float, a.size)),
                    "track": a.track.to_json(),
                    "visible_range": list(a.visible_range) if a.visible_range else None} for a in ds.actors],
        "points": None,
        "background_color": list(ds.background_color) if ds.background_color else None,
    }
    if ds.points is not None:
        np.save(root / "points.npy", ds.points.astype(np.float32))
        doc["points"] = "points.npy"
    (root / "log.json").write_text(json.dumps(doc, indent=1))
    return root / "log.json"


def load_log(root: str | Path) -> Dataset:
    root = Path(root)
    doc = json.loads((root / "log.json").read_text())
    frames = []
    for d in doc["frames"]:
        K = np.array(d["intrinsics"])
        cam = Camera(int(d["width"]), int(d["height"]), K[0, 0], K[1, 1], K[0, 2], K[1, 2],
                     np.array(d["camera_to_world"], dtype=np.float64))
        frames.append(Frame(float(d["t"]), cam, root / d["image"], root / d["depth"] if d.get("depth") else None))
    actors = [ActorAnnotation(a["name"], Track.from_json(a["track"]), a["size"], a.get("category", ""),
                              tuple(a["visible_range"]) if a.get("visible_range") else None)
              for a in doc.get("actors", [])]
    pts = np.load(root / doc["points"]) if doc.get("points") else None
    return Dataset(
        frames=frames, source=doc.get("source", "log"),
        ego_track=Track.from_json(doc["ego_track"]) if doc.get("ego_track") else None,
        camera_to_ego=np.array(doc["camera_to_ego"]) if doc.get("camera_to_ego") else None,
        actors=actors, points=pts,
        background_color=tuple(doc["background_color"]) if doc.get("background_color") else None,
        root=root,
    )


def load_any(path: str | Path, **kw) -> Dataset:
    """Auto-detect: m4dgs log, iPhone ARKit capture (dir or .zip), KITTI raw drive, or nuScenes root."""
    path = Path(path)
    if path.suffix == ".zip":
        from .arkit import load_arkit_capture
        return load_arkit_capture(path, **kw)
    if (path / "log.json").exists():
        return load_log(path)
    if (path / "capture.json").exists() or any(path.glob("*/capture.json")):
        from .arkit import load_arkit_capture
        return load_arkit_capture(path, **kw)
    if (path / "oxts").exists() or any(path.glob("*/*_sync/oxts")) or any(path.glob("*_sync/oxts")):
        from .kitti import load_kitti
        return load_kitti(path, **kw)
    if (path / "v1.0-mini").exists() or (path / "v1.0-trainval").exists():
        from .nuscenes import load_nuscenes
        return load_nuscenes(path, **kw)
    raise ValueError(f"could not detect dataset type at {path}")
