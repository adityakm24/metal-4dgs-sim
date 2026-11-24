"""nuScenes loader (reads the raw JSON tables directly; no devkit needed).

Uses CAM_FRONT key frames as training images, LIDAR_TOP key-frame sweeps (coloured by CAM_FRONT) as
initial points, and sample_annotation tracks of *moving* objects as dynamic actors. The global frame is
re-centred on the first ego pose (x forward, z up).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from engine.metal_splat.camera import Camera
from engine.scene_graph.tracks import Track, pose_matrix, quat_to_R, R_to_quat, yaw_quat, quat_yaw

from .dataset import ActorAnnotation, Dataset, Frame

DYNAMIC_PREFIXES = ("vehicle.", "human.", "cycle.")


def _table(root: Path, version: str, name: str) -> list[dict]:
    return json.loads((root / version / f"{name}.json").read_text())


def load_nuscenes(root: str | Path, scene_name: str | None = None, version: str | None = None, camera: str = "CAM_FRONT",
                  min_motion: float = 1.0, lidar_points_per_sweep: int = 6000, seed: int = 0) -> Dataset:
    root = Path(root)
    version = version or ("v1.0-mini" if (root / "v1.0-mini").exists() else "v1.0-trainval")
    T = {n: _table(root, version, n) for n in ("scene", "sample", "sample_data", "ego_pose", "calibrated_sensor",
                                                  "sensor", "sample_annotation", "instance", "category")}
    by_token = {n: {r["token"]: r for r in rows} for n, rows in T.items()}
    scene = next((s for s in T["scene"] if s["name"] == scene_name), None) if scene_name else T["scene"][0]
    if scene is None:
        raise KeyError(f"scene {scene_name!r} not found; available: {[s['name'] for s in T['scene']][:10]}...")
    samples = []
    tok = scene["first_sample_token"]
    while tok:
        s = by_token["sample"][tok]
        samples.append(s)
        tok = s["next"]
    sd_by_sample = defaultdict(dict)
    for sd in T["sample_data"]:
        if not sd["is_key_frame"]:
            continue
        cs = by_token["calibrated_sensor"][sd["calibrated_sensor_token"]]
        ch = by_token["sensor"][cs["sensor_token"]]["channel"]
        sd_by_sample[sd["sample_token"]][ch] = sd

    def ego_pose(sd):
        ep = by_token["ego_pose"][sd["ego_pose_token"]]
        return pose_matrix(ep["translation"], ep["rotation"])

    def sensor_pose(sd):
        cs = by_token["calibrated_sensor"][sd["calibrated_sensor_token"]]
        return pose_matrix(cs["translation"], cs["rotation"]), cs

    first = sd_by_sample[samples[0]["token"]][camera]
    E0 = ego_pose(first)
    origin = pose_matrix(E0[:3, 3], yaw_quat(quat_yaw(R_to_quat(E0[:3, :3]))))
    to_local = np.linalg.inv(origin)
    t0 = first["timestamp"] * 1e-6

    rng = np.random.default_rng(seed)
    frames, ego_t, ego_pos, ego_q, pts = [], [], [], [], []
    cam_to_ego = None
    import cv2

    for s in samples:
        sd = sd_by_sample[s["token"]].get(camera)
        if sd is None:
            continue
        t = sd["timestamp"] * 1e-6 - t0
        E = to_local @ ego_pose(sd)
        S, cs = sensor_pose(sd)
        cam_to_ego = S
        K = np.asarray(cs["camera_intrinsic"])
        cam = Camera(int(sd["width"]), int(sd["height"]), K[0, 0], K[1, 1], K[0, 2], K[1, 2], E @ S)
        frames.append(Frame(t, cam, root / sd["filename"]))
        ego_t.append(t)
        ego_pos.append(E[:3, 3])
        ego_q.append(R_to_quat(E[:3, :3]))
        lsd = sd_by_sample[s["token"]].get("LIDAR_TOP")
        if lsd is not None and (root / lsd["filename"]).exists():
            raw = np.fromfile(root / lsd["filename"], dtype=np.float32).reshape(-1, 5)[:, :3]
            L, _ = sensor_pose(lsd)
            W = to_local @ ego_pose(lsd) @ L
            xyz = raw @ W[:3, :3].T + W[:3, 3]
            uv, z = cam.project(xyz)
            ok = (z > 1) & (uv[:, 0] >= 0) & (uv[:, 0] < cam.width) & (uv[:, 1] >= 0) & (uv[:, 1] < cam.height)
            idx = np.nonzero(ok)[0]
            if len(idx):
                idx = rng.choice(idx, min(len(idx), lidar_points_per_sweep), replace=False)
                bgr = cv2.imread(str(root / sd["filename"]))
                col = bgr[uv[idx, 1].astype(int), uv[idx, 0].astype(int), ::-1] / 255.0
                pts.append(np.concatenate([xyz[idx], col, np.full((len(idx), 1), t)], 1))

    # dynamic actors from annotations
    sample_t = {s["token"]: by_token["sample"][s["token"]]["timestamp"] * 1e-6 - t0 for s in samples}
    tracks = defaultdict(list)
    for ann in T["sample_annotation"]:
        if ann["sample_token"] in sample_t:
            tracks[ann["instance_token"]].append(ann)
    actors = []
    for k, (inst, anns) in enumerate(sorted(tracks.items(), key=lambda kv: -len(kv[1]))):
        cat = by_token["category"][by_token["instance"][inst]["category_token"]]["name"]
        if not cat.startswith(DYNAMIC_PREFIXES):
            continue
        anns.sort(key=lambda a: sample_t[a["sample_token"]])
        ts = [sample_t[a["sample_token"]] for a in anns]
        P = [to_local @ pose_matrix(a["translation"], a["rotation"]) for a in anns]
        pos = np.stack([p[:3, 3] for p in P])
        if np.linalg.norm(pos[-1] - pos[0]) < min_motion:
            continue    # parked / static: stays in the background
        w, l, h = anns[0]["size"]
        actors.append(ActorAnnotation(f"actor_{len(actors) + 1}", Track(ts, pos, [R_to_quat(p[:3, :3]) for p in P]),
                                      [l, w, h], cat, (ts[0] - 0.25, ts[-1] + 0.25)))
    ds = Dataset(frames, source="nuscenes", ego_track=Track(ego_t, ego_pos, ego_q), camera_to_ego=cam_to_ego,
                 actors=actors, points=np.concatenate(pts) if pts else None, root=root)
    print(f"[nuscenes] {scene['name']}: {len(frames)} {camera} frames, {len(actors)} moving actors, "
          f"{0 if ds.points is None else len(ds.points)} lidar points")
    return ds
