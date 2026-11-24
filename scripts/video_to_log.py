#!/usr/bin/env python
"""Turn an ordinary video (phone, dashcam, drone) into a posed m4dgs log with COLMAP structure-from-motion.

    python scripts/video_to_log.py my_video.mp4 --out data/my_video [--driving --camera_height 1.5]

Steps: sharp-frame extraction -> COLMAP (SIMPLE_RADIAL, sequential matching) -> undistortion to PINHOLE ->
gravity alignment (cameras assumed upright: world +z = mean camera "up") -> optional metric scale ->
log.json (+ points.npy from the sparse SfM points) that train_4dgs.py reads.

Metric scale: SfM is scale-free. Use --camera_height (driving: fits the ground plane under the camera path),
--path_length (known distance travelled), or --scale. Otherwise units are arbitrary (fine for viewing, but
the closed-loop physics assume metres).
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

import _bootstrap  # noqa: F401
import cv2
import numpy as np

from capture.dataset import Dataset, Frame, save_log
from engine.metal_splat.camera import Camera
from engine.scene_graph.scene import default_camera_to_ego
from engine.scene_graph.tracks import R_to_quat, Track, yaw_quat


def extract_frames(video: Path, out: Path, max_frames: int, max_width: int, step: int | None) -> tuple[list[Path], float]:
    out.mkdir(parents=True, exist_ok=True)
    if video.is_dir():
        srcs = sorted([p for p in video.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
        fps = 10.0
        reader = (cv2.imread(str(p)) for p in srcs)
        total = len(srcs)
    else:
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        def gen():
            while True:
                ok, f = cap.read()
                if not ok:
                    return
                yield f

        reader = gen()
    step = step or max(1, total // max_frames)
    window, paths, times = [], [], []
    for i, img in enumerate(reader):
        if img is None:
            continue
        window.append((cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var(), i, img))
        if len(window) == step:   # keep the sharpest frame of every `step` frames
            _, j, best = max(window, key=lambda w: w[0])
            window = []
            if best.shape[1] > max_width:
                s = max_width / best.shape[1]
                best = cv2.resize(best, (max_width, int(round(best.shape[0] * s))), interpolation=cv2.INTER_AREA)
            p = out / f"{len(paths):06d}.jpg"
            cv2.imwrite(str(p), best, [cv2.IMWRITE_JPEG_QUALITY, 95])
            paths.append(p)
            times.append(j / fps)
            if len(paths) >= max_frames:
                break
    np.save(out.parent / "frame_times.npy", np.array(times))
    return paths, fps


def colmap(*args, cwd=None):
    r = subprocess.run(["colmap", *args], cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"colmap {args[0]} failed:\n{r.stderr[-3000:]}")
    return r.stdout


def run_sfm(work: Path, images: Path, gpu: bool = False) -> Path:
    db = work / "database.db"
    db.unlink(missing_ok=True)
    colmap("feature_extractor", "--database_path", str(db), "--image_path", str(images),
           "--ImageReader.single_camera", "1", "--ImageReader.camera_model", "SIMPLE_RADIAL",
           "--FeatureExtraction.use_gpu", "1" if gpu else "0")
    colmap("sequential_matcher", "--database_path", str(db), "--SequentialMatching.overlap", "15",
           "--FeatureMatching.use_gpu", "1" if gpu else "0")
    sparse = work / "sparse"
    shutil.rmtree(sparse, ignore_errors=True)
    sparse.mkdir()
    colmap("mapper", "--database_path", str(db), "--image_path", str(images), "--output_path", str(sparse))
    models = sorted(sparse.iterdir(), key=lambda p: -sum(1 for _ in p.glob("*")))
    if not models:
        raise RuntimeError("COLMAP produced no reconstruction (too little texture / motion?)")
    # pick the model with the most registered images
    best = max(models, key=lambda m: len(_read_images_txt(_to_txt(m))))
    und = work / "undistorted"
    shutil.rmtree(und, ignore_errors=True)
    colmap("image_undistorter", "--image_path", str(images), "--input_path", str(best), "--output_path", str(und),
           "--output_type", "COLMAP")
    return und


def _to_txt(model: Path) -> Path:
    txt = model.parent / (model.name + "_txt")
    txt.mkdir(exist_ok=True)
    colmap("model_converter", "--input_path", str(model), "--output_path", str(txt), "--output_type", "TXT")
    return txt


def _read_images_txt(txt: Path) -> dict:
    lines = [l for l in (txt / "images.txt").read_text().splitlines() if l and not l.startswith("#")]
    out = {}
    for l in lines[0::2]:
        e = l.split()
        out[e[9]] = (np.array(list(map(float, e[1:5]))), np.array(list(map(float, e[5:8]))), int(e[8]))
    return out


def read_model(und: Path):
    txt = _to_txt(und / "sparse")
    cams = {}
    for l in (txt / "cameras.txt").read_text().splitlines():
        if l and not l.startswith("#"):
            e = l.split()
            assert e[1] == "PINHOLE", e[1]
            cams[int(e[0])] = (int(e[2]), int(e[3]), *map(float, e[4:8]))
    imgs = _read_images_txt(txt)
    pts = []
    for l in (txt / "points3D.txt").read_text().splitlines():
        if l and not l.startswith("#"):
            e = l.split()
            pts.append([float(e[1]), float(e[2]), float(e[3]), int(e[4]) / 255, int(e[5]) / 255, int(e[6]) / 255])
    return cams, imgs, np.array(pts)


def quat_to_R(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def gravity_align(c2ws: np.ndarray, pts: np.ndarray):
    """Rotate so the mean camera 'up' (-y in OpenCV) is +z, and the first camera looks along +x."""
    up = -np.mean([T[:3, 1] for T in c2ws], axis=0)
    up /= np.linalg.norm(up)
    fwd = c2ws[0][:3, 2] - np.dot(c2ws[0][:3, 2], up) * up
    fwd /= np.linalg.norm(fwd)
    left = np.cross(up, fwd)
    R = np.stack([fwd, left, up])            # rows: new x, y, z in old coords
    origin = c2ws[0][:3, 3]
    A = np.eye(4)
    A[:3, :3] = R
    A[:3, 3] = -R @ origin
    return np.stack([A @ T for T in c2ws]), (pts[:, :3] - origin) @ R.T


def metric_scale(c2ws, pts, camera_height=None, path_length=None):
    if path_length:
        L = np.sum(np.linalg.norm(np.diff(c2ws[:, :3, 3], axis=0), axis=1))
        return path_length / max(L, 1e-9)
    if camera_height:
        # ground = robust low percentile of point heights in a corridor around the camera path
        cz = np.median(c2ws[:, 2, 3])
        below = pts[pts[:, 2] < cz]
        if len(below) < 50:
            raise RuntimeError("not enough points below the cameras to find the ground plane")
        ground = np.percentile(below[:, 2], 10)
        return camera_height / max(cz - ground, 1e-9)
    return 1.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="video file or folder of images")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_frames", type=int, default=150)
    ap.add_argument("--step", type=int, default=None, help="keep 1 sharp frame per `step` input frames")
    ap.add_argument("--max_width", type=int, default=960)
    ap.add_argument("--driving", action="store_true", help="build an ego (vehicle) track under the camera")
    ap.add_argument("--camera_height", type=float, default=None)
    ap.add_argument("--path_length", type=float, default=None)
    ap.add_argument("--scale", type=float, default=None)
    ap.add_argument("--fps", type=float, default=None, help="override time base (frames/s of the kept frames)")
    a = ap.parse_args()
    out = Path(a.out)
    work = out / "colmap"
    raw = work / "frames"
    paths, fps = extract_frames(Path(a.video), raw, a.max_frames, a.max_width, a.step)
    print(f"[video] kept {len(paths)} sharp frames; running COLMAP (this can take a few minutes)", flush=True)
    und = run_sfm(work, raw)
    cams, imgs, pts = read_model(und)
    names = sorted(imgs)
    times_all = np.load(raw.parent / "frame_times.npy")
    c2ws, Ks, sizes, ts, files = [], [], [], [], []
    for n in names:
        q, t, cid = imgs[n]
        R = quat_to_R(q)
        w2c = np.eye(4)
        w2c[:3, :3], w2c[:3, 3] = R, t
        c2ws.append(np.linalg.inv(w2c))
        W, H, fx, fy, cx, cy = cams[cid]
        Ks.append((fx, fy, cx, cy))
        sizes.append((W, H))
        idx = int(Path(n).stem)
        ts.append(idx / a.fps if a.fps else float(times_all[idx]))
        files.append(und / "images" / n)
    c2ws = np.stack(c2ws)
    c2ws, xyz = gravity_align(c2ws, pts)
    s = a.scale or metric_scale(c2ws, xyz, a.camera_height, a.path_length)
    c2ws[:, :3, 3] *= s
    xyz *= s
    t0 = min(ts)
    (out / "images").mkdir(parents=True, exist_ok=True)
    frames = []
    for i, (T, K, (W, H), t, f) in enumerate(zip(c2ws, Ks, sizes, ts, files)):
        dst = out / "images" / f"{i:06d}.jpg"
        shutil.copy(f, dst)
        frames.append(Frame(t - t0, Camera(W, H, *K, T), dst))
    frames.sort(key=lambda fr: fr.t)
    ego_track, c2e = None, None
    if a.driving:
        h = a.camera_height or 1.5
        c2e = default_camera_to_ego(h, 0.0)
        e2c = np.linalg.inv(c2e)
        poses = []
        for fr in frames:
            E = fr.camera.c2w @ e2c
            yaw = np.arctan2(E[1, 0], E[0, 0])    # keep heading only: vehicle stays level
            poses.append((E[:3, 3], yaw_quat(yaw)))
        ego_track = Track([fr.t for fr in frames], [p for p, _ in poses], [q for _, q in poses])
        # with a level ego, the camera inherits only yaw: refit camera_to_ego from the first frame
        c2e = np.linalg.inv(ego_track.pose(frames[0].t)) @ frames[0].camera.c2w
    points = np.concatenate([xyz, pts[:, 3:6], np.full((len(xyz), 1), np.nan)], 1)
    ds = Dataset(frames, source="video", ego_track=ego_track, camera_to_ego=c2e, points=points, root=out)
    save_log(ds, out)
    L = np.sum(np.linalg.norm(np.diff(c2ws[:, :3, 3], axis=0), axis=1))
    print(f"[video] {len(frames)}/{len(paths)} frames registered, {len(xyz)} SfM points, scale {s:.4f}, "
          f"path length {L:.2f} -> {out}/log.json")


if __name__ == "__main__":
    main()
