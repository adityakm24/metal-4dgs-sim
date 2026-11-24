import json

import cv2
import numpy as np

from capture.arkit import ARKIT_CAM_TO_CV, ARKIT_WORLD_TO_SIM, load_arkit_capture
from engine.metal_splat.camera import look_at


def test_arkit_capture_roundtrip(tmp_path):
    root = tmp_path / "capture_20261124_101500"
    (root / "images").mkdir(parents=True)
    (root / "depth").mkdir()
    c2w_sim = look_at([0, 0, 1.5], [10, 1, 1.0])
    # sim -> ARKit (inverse of the loader's conversion)
    c2w_ark = np.linalg.inv(ARKIT_WORLD_TO_SIM) @ c2w_sim @ np.linalg.inv(ARKIT_CAM_TO_CV)
    cv2.imwrite(str(root / "images/000000.jpg"), np.full((48, 64, 3), 128, np.uint8))
    cv2.imwrite(str(root / "depth/000000.png"), np.full((12, 16), 2500, np.uint16))
    doc = {"format": "m4dgs-arkit-capture", "version": 1, "image_width": 64, "image_height": 48,
           "depth_width": 16, "depth_height": 12, "has_lidar": True,
           "frames": [{"index": 0, "timestamp": 0.0, "image": "images/000000.jpg", "depth": "depth/000000.png",
                       "confidence": None, "camera_to_world": c2w_ark.tolist(),
                       "intrinsics": [[50, 0, 32], [0, 50, 24], [0, 0, 1]], "tracking_state": "normal"}]}
    (root / "capture.json").write_text(json.dumps(doc))
    (root / "points.ply").write_text("ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\n"
                                     "property float z\nend_header\n0 1 -5\n")
    ds = load_arkit_capture(tmp_path)
    f = ds.frames[0]
    assert np.allclose(f.camera.c2w, c2w_sim, atol=1e-9)
    assert np.allclose(f.depth(), 2.5)
    # ARKit point (0,1,-5): 5 m in front (-z), 1 m up  ->  sim (5, 0, 1)
    assert np.allclose(ds.points[0, :3], [5, 0, 1])
