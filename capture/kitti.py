"""KITTI raw loader (http://www.cvlibs.net/datasets/kitti/raw_data.php, CC BY-NC-SA 3.0).

Expects the standard layout produced by unzipping `<date>_calib.zip`, `<drive>_sync.zip` and
`<drive>_tracklets.zip` in the same folder:

    <root>/<date>/calib_{cam_to_cam,imu_to_velo,velo_to_cam}.txt
    <root>/<date>/<date>_drive_XXXX_sync/{image_02,oxts,velodyne_points}/...
    <root>/<date>/<date>_drive_XXXX_sync/tracklet_labels.xml       (optional)

Ego = OXTS IMU frame (x forward, y left, z up). Poses from GPS/IMU via the standard Mercator conversion,
re-centred on the first frame. Camera = rectified left colour camera (image_02). Tracklets of moving
objects become dynamic actors; parked objects stay in the background.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from engine.metal_splat.camera import Camera
from engine.scene_graph.tracks import R_to_quat, Track, yaw_quat

from .dataset import ActorAnnotation, Dataset, Frame


def _read_calib(path: Path) -> dict[str, np.ndarray]:
    out = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        try:
            out[k.strip()] = np.array([float(x) for x in v.split()])
        except ValueError:
            pass
    return out


def _rt(d: dict) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = d["R"].reshape(3, 3)
    T[:3, 3] = d["T"]
    return T


def _timestamps(path: Path) -> np.ndarray:
    ts = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        day, frac = line.split(".")
        dt = datetime.strptime(day, "%Y-%m-%d %H:%M:%S")
        ts.append(dt.timestamp() + float("0." + frac))
    return np.array(ts)


def oxts_poses(oxts_dir: Path) -> np.ndarray:
    """(N,4,4) IMU-to-world poses (first frame = identity) from OXTS packets."""
    packets = [np.array([float(x) for x in f.read_text().split()]) for f in sorted(oxts_dir.glob("*.txt"))]
    er = 6378137.0
    scale = np.cos(packets[0][0] * np.pi / 180.0)
    poses = []
    for p in packets:
        lat, lon, alt, roll, pitch, yaw = p[:6]
        tx = scale * lon * np.pi * er / 180.0
        ty = scale * er * np.log(np.tan((90.0 + lat) * np.pi / 360.0))
        Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        T = np.eye(4)
        T[:3, :3] = Rz @ Ry @ Rx
        T[:3, 3] = [tx, ty, alt]
        poses.append(T)
    T0inv = np.linalg.inv(poses[0])
    return np.stack([T0inv @ T for T in poses])


def parse_tracklets(xml_path: Path):
    """Yields (object_type, h, w, l, first_frame, [(tx,ty,tz,rz), ...]) in velodyne coordinates."""
    root = ET.parse(xml_path).getroot()
    for item in root.find("tracklets").findall("item"):
        typ = item.findtext("objectType")
        h, w, l = (float(item.findtext(k)) for k in ("h", "w", "l"))
        first = int(item.findtext("first_frame"))
        poses = [(float(p.findtext("tx")), float(p.findtext("ty")), float(p.findtext("tz")), float(p.findtext("rz")))
                 for p in item.find("poses").findall("item")]
        yield typ, h, w, l, first, poses


def _find_drive(root: Path) -> Path:
    if (root / "oxts").exists():
        return root
    found = sorted(p for p in root.rglob("*_sync") if (p / "oxts").exists())
    if not found:
        raise FileNotFoundError(f"no KITTI *_sync drive under {root}")
    return found[0]


def load_kitti(root: str | Path, max_frames: int | None = None, lidar_points_per_frame: int = 8000,
               min_motion: float = 1.0, seed: int = 0) -> Dataset:
    drive = _find_drive(Path(root))
    date_dir = drive.parent
    cam = _read_calib(date_dir / "calib_cam_to_cam.txt")
    T_velo_imu = _rt(_read_calib(date_dir / "calib_imu_to_velo.txt"))
    T_cam0_velo = _rt(_read_calib(date_dir / "calib_velo_to_cam.txt"))
    R_rect = np.eye(4)
    R_rect[:3, :3] = cam["R_rect_00"].reshape(3, 3)
    P2 = cam["P_rect_02"].reshape(3, 4)
    K = P2[:, :3]
    T_cam2_cam0rect = np.eye(4)
    T_cam2_cam0rect[:3, 3] = np.linalg.solve(K, P2[:, 3])
    T_cam2_velo = T_cam2_cam0rect @ R_rect @ T_cam0_velo
    T_cam2_imu = T_cam2_velo @ T_velo_imu
    T_imu_cam2 = np.linalg.inv(T_cam2_imu)
    T_imu_velo = np.linalg.inv(T_velo_imu)

    imu_poses = oxts_poses(drive / "oxts" / "data")
    img_files = sorted((drive / "image_02" / "data").glob("*.png"))
    ts = _timestamps(drive / "image_02" / "timestamps.txt")
    ts = ts - ts[0]
    n = min(len(img_files), len(imu_poses), len(ts))
    if max_frames:
        n = min(n, max_frames)
    h, w = cv2.imread(str(img_files[0])).shape[:2]
    rng = np.random.default_rng(seed)
    frames, pts = [], []
    velo_files = sorted((drive / "velodyne_points" / "data").glob("*.bin"))
    for i in range(n):
        c2w = imu_poses[i] @ T_imu_cam2
        camera = Camera(w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2], c2w)
        frames.append(Frame(float(ts[i]), camera, img_files[i]))
        if i < len(velo_files):
            xyz = np.fromfile(velo_files[i], dtype=np.float32).reshape(-1, 4)[:, :3].astype(np.float64)
            pc = xyz @ T_cam2_velo[:3, :3].T + T_cam2_velo[:3, 3]
            z = pc[:, 2]
            u = K[0, 0] * pc[:, 0] / np.maximum(z, 1e-6) + K[0, 2]
            v = K[1, 1] * pc[:, 1] / np.maximum(z, 1e-6) + K[1, 2]
            ok = np.nonzero((z > 1.0) & (z < 80) & (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1))[0]
            if len(ok):
                ok = rng.choice(ok, min(len(ok), lidar_points_per_frame), replace=False)
                img = cv2.cvtColor(cv2.imread(str(img_files[i])), cv2.COLOR_BGR2RGB) / 255.0
                col = img[v[ok].astype(int), u[ok].astype(int)]
                T_w_velo = imu_poses[i] @ T_imu_velo
                xw = xyz[ok] @ T_w_velo[:3, :3].T + T_w_velo[:3, 3]
                pts.append(np.concatenate([xw, col, np.full((len(ok), 1), ts[i])], 1))

    actors = []
    xml = drive / "tracklet_labels.xml"
    if xml.exists():
        for typ, hh, ww, ll, first, poses in parse_tracklets(xml):
            if typ in ("DontCare", "Misc"):
                continue
            t_list, p_list, q_list = [], [], []
            for k, (tx, ty, tz, rz) in enumerate(poses):
                f = first + k
                if f >= n:
                    break
                B = np.eye(4)
                B[:3, :3] = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
                B[:3, 3] = [tx, ty, tz + hh / 2]          # tracklet z is the box bottom
                Wb = imu_poses[f] @ T_imu_velo @ B
                t_list.append(float(ts[f]))
                p_list.append(Wb[:3, 3])
                q_list.append(R_to_quat(Wb[:3, :3]))
            if len(t_list) < 2:
                continue
            pos = np.stack(p_list)
            if np.linalg.norm(pos[-1] - pos[0]) < min_motion:
                continue
            actors.append(ActorAnnotation(f"actor_{len(actors) + 1}", Track(t_list, pos, q_list), [ll, ww, hh],
                                          f"kitti.{typ}", (t_list[0] - 0.05, t_list[-1] + 0.05)))
    ego_track = Track(ts[:n], imu_poses[:n, :3, 3], [R_to_quat(T[:3, :3]) for T in imu_poses[:n]])
    print(f"[kitti] {drive.name}: {n} frames {w}x{h}, {len(actors)} moving actors, "
          f"{sum(len(p) for p in pts)} lidar points")
    return Dataset(frames, source="kitti", ego_track=ego_track, camera_to_ego=T_imu_cam2, actors=actors,
                   points=np.concatenate(pts) if pts else None, root=drive)
