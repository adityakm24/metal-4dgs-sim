"""Waymo Open Dataset (Perception v1 tfrecords) -> m4dgs log converter.

EXPERIMENTAL / UNTESTED in this repo: needs `pip install waymo-open-dataset-tf-2-12-0 tensorflow` (Linux
x86 wheels only at the time of writing, so convert on a Linux box and copy the log to the Mac).
Usage: python -m capture.waymo segment.tfrecord out_dir
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from engine.metal_splat.camera import Camera
from engine.scene_graph.tracks import R_to_quat, Track, yaw_quat

from .dataset import ActorAnnotation, Dataset, Frame, save_log

# Waymo camera frame: +x forward, +y left, +z up  ->  OpenCV
WAYMO_CAM_TO_CV = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float64).T


def convert(tfrecord: str, out_dir: str, camera_name: int = 1, max_frames: int | None = None) -> Path:
    import tensorflow as tf
    from waymo_open_dataset import dataset_pb2

    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    frames, ego_t, ego_p, ego_q, tracks = [], [], [], [], {}
    origin_inv, t0, cam_to_ego = None, None, None
    for i, rec in enumerate(tf.data.TFRecordDataset(tfrecord, compression_type="")):
        if max_frames and i >= max_frames:
            break
        fr = dataset_pb2.Frame()
        fr.ParseFromString(bytearray(rec.numpy()))
        E = np.array(fr.pose.transform).reshape(4, 4)
        if origin_inv is None:
            origin_inv, t0 = np.linalg.inv(E), fr.timestamp_micros * 1e-6
        E = origin_inv @ E
        t = fr.timestamp_micros * 1e-6 - t0
        calib = next(c for c in fr.context.camera_calibrations if c.name == camera_name)
        img = next(c for c in fr.images if c.name == camera_name)
        S = np.array(calib.extrinsic.transform).reshape(4, 4) @ WAYMO_CAM_TO_CV
        cam_to_ego = S
        f_u, f_v, c_u, c_v = calib.intrinsic[:4]
        p = out / "images" / f"{i:06d}.jpg"
        p.write_bytes(img.image)
        frames.append(Frame(t, Camera(calib.width, calib.height, f_u, f_v, c_u, c_v, E @ S), p))
        ego_t.append(t), ego_p.append(E[:3, 3]), ego_q.append(R_to_quat(E[:3, :3]))
        for lab in fr.laser_labels:
            b = lab.box
            pos = (E @ np.array([b.center_x, b.center_y, b.center_z, 1.0]))[:3]
            yaw = np.arctan2(E[1, 0], E[0, 0]) + b.heading
            tr = tracks.setdefault(lab.id, {"t": [], "pos": [], "q": [], "size": [b.length, b.width, b.height], "type": lab.type})
            tr["t"].append(t), tr["pos"].append(pos), tr["q"].append(yaw_quat(yaw))
    actors = []
    for d in tracks.values():
        pos = np.stack(d["pos"])
        if len(pos) > 1 and np.linalg.norm(pos[-1] - pos[0]) > 1.0:
            actors.append(ActorAnnotation(f"actor_{len(actors) + 1}", Track(d["t"], pos, d["q"]), d["size"],
                                          f"waymo.type{d['type']}", (d["t"][0] - 0.1, d["t"][-1] + 0.1)))
    ds = Dataset(frames, "waymo", Track(ego_t, ego_p, ego_q), cam_to_ego, actors, root=out)
    return save_log(ds, out)


if __name__ == "__main__":
    print(convert(sys.argv[1], sys.argv[2]))
