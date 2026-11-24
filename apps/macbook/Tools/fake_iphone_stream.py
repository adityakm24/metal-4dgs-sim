#!/usr/bin/env python
"""Stand-in for the iPhone capture app: streams a capture to M4DGS Studio over the live link (docs/formats.md §6).

Source (one of):
  --replay DIR    an existing m4dgs ARKit capture (capture.json + images/ depth/ confidence/ points.ply), sent as is
  (default)       synthesize an ARKit capture from <repo>/data/sample_scene: ~30 frames of its images and poses
                  (sim/OpenCV poses converted back to ARKit conventions = inverse of capture/arkit.py), LiDAR-like
                  16-bit depth + confidence PNGs, and a points.ply from points.npy

Target: --host/--port, or Bonjour discovery of _m4dgs-live._tcp via `dns-sd -B` / `dns-sd -L` (--service picks an
instance name; default: the first one found).

  ../../.venv/bin/python Tools/fake_iphone_stream.py                       # discover + synthesize 30 frames
  ../../.venv/bin/python Tools/fake_iphone_stream.py --host 127.0.0.1 --port 50123 --save-local /tmp/phone_copy
  ../../.venv/bin/python Tools/fake_iphone_stream.py --replay ../../data/iphone/capture_20261124_101500
  ... --drop-after 5     # close the socket mid-capture (Mac should leave capture.partial.json)
  ... --cancel-after 5   # send `cancel` (Mac should delete the partial directory)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import pty
import random
import re
import select
import socket
import struct
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
SERVICE = "_m4dgs-live._tcp"
MAGIC = b"M4DG"
HELLO, FRAME, POINTS, END, STATUS, CANCEL, SAVED = 1, 2, 3, 4, 5, 6, 7
NAMES = {1: "hello", 2: "frame", 3: "points", 4: "end", 5: "status", 6: "cancel", 7: "saved"}

# must match capture/arkit.py
ARKIT_CAM_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])
ARKIT_WORLD_TO_SIM = np.array([[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)


def encode(mtype: int, header: bytes | dict, payload: bytes = b"") -> bytes:
    if isinstance(header, dict):
        header = json.dumps(header, separators=(",", ":")).encode()
    return MAGIC + bytes([mtype, 0, 0, 0]) + struct.pack(">II", len(header), len(payload)) + header + payload


# ---------------------------------------------------------------------------------------------- discovery

def _run_lines(cmd: list[str], pattern: re.Pattern, timeout: float) -> list[re.Match]:
    """Runs a long-lived dns-sd command on a pty (it block-buffers into pipes) and collects matching lines."""
    master, slave = pty.openpty()
    p = subprocess.Popen(cmd, stdout=slave, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, close_fds=True)
    os.close(slave)
    found, t_end, buf = [], time.time() + timeout, b""
    try:
        while time.time() < t_end:
            r, _, _ = select.select([master], [], [], max(0.0, t_end - time.time()))
            if not r:
                break
            try:
                chunk = os.read(master, 4096)
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            *lines, buf = buf.split(b"\n")
            for line in lines:
                m = pattern.search(line.decode(errors="replace").rstrip("\r"))
                if m:
                    found.append(m)
                    t_end = min(t_end, time.time() + (0.0 if pattern is _LOOKUP else 1.0))
    finally:
        p.kill()
        p.wait()
        os.close(master)
    return found


def resolve_host(host: str, timeout: float) -> str:
    """getaddrinfo on the mDNS name, falling back to `dns-sd -G v4v6`."""
    try:
        return socket.getaddrinfo(host, None, socket.AF_INET, socket.SOCK_STREAM)[0][4][0]
    except OSError:
        pass
    ms = _run_lines(["dns-sd", "-G", "v4v6", host], _ADDR, timeout)
    if not ms:
        sys.exit(f"could not resolve host {host}")
    return ms[0].group(1)


_BROWSE = re.compile(r"\sAdd\s+\d+\s+\d+\s+(\S+)\s+" + re.escape(SERVICE) + r"\.\s+(.+?)\s*$")
_LOOKUP = re.compile(r"can be reached at\s+(\S+?)\.?:(\d+)")
_ADDR = re.compile(r"\sAdd\s+\S+\s+\d+\s+\S+\s+(\d+\.\d+\.\d+\.\d+)")


def discover(service_name: str | None, timeout: float) -> tuple[str, int, str]:
    if service_name is None:
        ms = _run_lines(["dns-sd", "-B", SERVICE, "local."], _BROWSE, timeout)
        if not ms:
            sys.exit(f"no {SERVICE} service found within {timeout:.0f} s (is M4DGS Studio / --receive running?)")
        names = list(dict.fromkeys(m.group(2) for m in ms))
        print(f"[fake-iphone] found: {names}")
        service_name = names[0]
    ms = _run_lines(["dns-sd", "-L", service_name, SERVICE, "local."], _LOOKUP, timeout)
    if not ms:
        sys.exit(f"could not resolve {service_name!r}")
    return resolve_host(ms[0].group(1), timeout), int(ms[0].group(2)), service_name


# ---------------------------------------------------------------------------------------------- sources

def to_ply(xyz: np.ndarray) -> bytes:
    head = ("ply\nformat ascii 1.0\ncomment ARKit rawFeaturePoints (world)\n"
            f"element vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
    body = "".join(f"{x:.5f} {y:.5f} {z:.5f}\n" for x, y, z in xyz)
    return (head + body).encode()


def sim_pose_to_arkit(c2w_sim) -> np.ndarray:
    """Inverse of capture.arkit.arkit_pose_to_sim (both matrices are orthonormal, CAM_TO_CV is an involution)."""
    return ARKIT_WORLD_TO_SIM.T @ np.asarray(c2w_sim, dtype=np.float64) @ ARKIT_CAM_TO_CV


def synthesize(scene: Path, n_frames: int, capture_name: str):
    log = json.loads((scene / "log.json").read_text())
    src = log["frames"]
    idx = np.unique(np.linspace(0, len(src) - 1, min(n_frames, len(src))).round().astype(int))
    t0 = float(src[idx[0]]["t"])
    W, H = int(src[0]["width"]), int(src[0]["height"])
    dW, dH = 256, max(1, round(256 * H / W))          # LiDAR-like low-res depth
    frames, files = [], []
    for k, i in enumerate(idx):
        f = src[i]
        img = cv2.imread(str(scene / f["image"]), cv2.IMREAD_COLOR)
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        assert ok
        depth_png = conf_png = None
        if f.get("depth"):
            dp = scene / f["depth"]
            d = np.load(dp) if dp.suffix == ".npy" else cv2.imread(str(dp), cv2.IMREAD_UNCHANGED) / 1000.0
            d = cv2.resize(d.astype(np.float32), (dW, dH), interpolation=cv2.INTER_NEAREST)
            valid = np.isfinite(d) & (d > 0) & (d < 65.0)
            mm = np.where(valid, np.round(d * 1000), 0).astype(np.uint16)
            depth_png = cv2.imencode(".png", mm)[1].tobytes()
            conf_png = cv2.imencode(".png", np.where(valid, 2, 0).astype(np.uint8))[1].tobytes()
        name = f"{k:06d}"
        fr = {
            "index": k,
            "timestamp": round(float(f["t"]) - t0, 6),
            "image": f"images/{name}.jpg",
            "depth": f"depth/{name}.png" if depth_png else None,
            "confidence": f"confidence/{name}.png" if conf_png else None,
            "camera_to_world": sim_pose_to_arkit(f["camera_to_world"]).tolist(),
            "intrinsics": f["intrinsics"],
            "tracking_state": "normal",
            "exposure_duration": 0.008,
            "motion_blur_score": 0.0,
        }
        frames.append(fr)
        files.append((jpg.tobytes(), depth_png or b"", conf_png or b""))
    has_lidar = any(f["depth"] for f in frames)
    doc = {
        "format": "m4dgs-arkit-capture", "version": 1,
        "device_model": "fake_iphone_stream", "system_version": platform.mac_ver()[0] or "0",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "world_convention": "arkit", "camera_convention": "arkit",
        "image_width": W, "image_height": H,
        "depth_width": dW if has_lidar else None, "depth_height": dH if has_lidar else None,
        "has_lidar": has_lidar, "frames": frames,
    }
    ply = None
    if (scene / log.get("points", "points.npy")).exists():
        pts = np.load(scene / log.get("points", "points.npy"))[:, :3].astype(np.float64)
        pts = pts[np.isfinite(pts).all(1)]
        if len(pts) > 20000:
            pts = pts[np.random.default_rng(0).choice(len(pts), 20000, replace=False)]
        ply = to_ply(pts @ ARKIT_WORLD_TO_SIM[:3, :3])     # x_ark = R^T x_sim (row vectors)
    return doc, files, ply, json.dumps(doc, indent=1).encode()


def load_replay(root: Path):
    raw = (root / "capture.json").read_bytes()
    doc = json.loads(raw)
    files = []
    for f in doc["frames"]:
        files.append(tuple((root / f[k]).read_bytes() if f.get(k) and (root / f[k]).exists() else b""
                           for k in ("image", "depth", "confidence")))
    ply = (root / "points.ply").read_bytes() if (root / "points.ply").exists() else None
    return doc, files, ply, raw


def save_local(out: Path, doc, files, ply, raw):
    out.mkdir(parents=True, exist_ok=True)
    for f, parts in zip(doc["frames"], files):
        for key, b in zip(("image", "depth", "confidence"), parts):
            if f.get(key) and b:
                (out / f[key]).parent.mkdir(parents=True, exist_ok=True)
                (out / f[key]).write_bytes(b)
    if ply:
        (out / "points.ply").write_bytes(ply)
    (out / "capture.json").write_bytes(raw)
    print(f"[fake-iphone] local copy → {out}")


# ---------------------------------------------------------------------------------------------- streaming

class Link:
    def __init__(self, host: str, port: int, chunk_max: int):
        self.sock = socket.create_connection((host, port), timeout=10)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.settimeout(None)
        self.chunk_max, self.pending, self.sent = chunk_max, b"", 0
        self.statuses, self.saved, self.closed = [], None, threading.Event()
        self.rng = random.Random(1)
        threading.Thread(target=self._reader, daemon=True).start()

    def send(self, msg: bytes, flush: bool = False):
        """Writes in random-sized chunks that straddle message boundaries (exercises reassembly on the Mac)."""
        self.pending += msg
        self.sent += len(msg)
        while self.pending and (flush or len(self.pending) > self.chunk_max):
            n = min(len(self.pending), self.rng.randint(1, self.chunk_max)) if self.chunk_max > 0 else len(self.pending)
            self.sock.sendall(self.pending[:n])
            self.pending = self.pending[n:]

    def _reader(self):
        buf = b""
        try:
            while True:
                d = self.sock.recv(65536)
                if not d:
                    break
                buf += d
                while len(buf) >= 16:
                    if buf[:4] != MAGIC:
                        print("[fake-iphone] bad magic from Mac", file=sys.stderr)
                        return
                    t = buf[4]
                    hl, pl = struct.unpack(">II", buf[8:16])
                    if len(buf) < 16 + hl + pl:
                        break
                    h = json.loads(buf[16:16 + hl] or b"{}")
                    buf = buf[16 + hl + pl:]
                    if t == STATUS:
                        self.statuses.append(h)
                    elif t == SAVED:
                        self.saved = h
                        print(f"[fake-iphone] saved on Mac: {h}")
                    else:
                        print(f"[fake-iphone] unexpected {NAMES.get(t, t)} from Mac: {h}")
        except OSError:
            pass
        finally:
            self.closed.set()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host")
    ap.add_argument("--port", type=int)
    ap.add_argument("--service", help="Bonjour instance name to connect to (default: first found)")
    ap.add_argument("--discover-timeout", type=float, default=5.0)
    ap.add_argument("--replay", type=Path, help="existing ARKit capture directory to stream")
    ap.add_argument("--scene", type=Path, default=REPO / "data" / "sample_scene", help="m4dgs log to synthesize from")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--name", help="capture_name (default: capture_<now>)")
    ap.add_argument("--fps", type=float, default=15.0, help="frame rate to stream at (0 = as fast as possible)")
    ap.add_argument("--chunk-max", type=int, default=4096, help="max bytes per socket write (0 = whole messages)")
    ap.add_argument("--save-local", type=Path, help="also write the capture here (the phone's local copy)")
    ap.add_argument("--drop-after", type=int, help="close the connection abruptly after this many frames")
    ap.add_argument("--cancel-after", type=int, help="send cancel after this many frames")
    a = ap.parse_args()

    if a.replay:
        doc, files, ply, raw = load_replay(a.replay)
        name = a.name or a.replay.name
    else:
        name = a.name or datetime.now().strftime("capture_%Y%m%d_%H%M%S")
        doc, files, ply, raw = synthesize(a.scene, a.frames, name)
    print(f"[fake-iphone] {name}: {len(doc['frames'])} frames {doc['image_width']}x{doc['image_height']}, "
          f"lidar={doc.get('has_lidar')}, points.ply={'%d B' % len(ply) if ply else 'none'}")
    if a.save_local:
        save_local(a.save_local, doc, files, ply, raw)

    if a.host and a.port:
        host, port = a.host, a.port
    else:
        host, port, svc = discover(a.service, a.discover_timeout)
        print(f"[fake-iphone] resolved {svc!r} → {host}:{port}")
    link = Link(host, port, a.chunk_max)
    print(f"[fake-iphone] connected to {host}:{port}")

    link.send(encode(HELLO, {
        "protocol": 1, "capture_name": name,
        "device_model": doc.get("device_model"), "system_version": doc.get("system_version"),
        "image_width": doc["image_width"], "image_height": doc["image_height"],
        "depth_width": doc.get("depth_width"), "depth_height": doc.get("depth_height"),
        "has_lidar": doc.get("has_lidar", False),
    }))
    t_start = time.time()
    for k, (f, (img, dep, conf)) in enumerate(zip(doc["frames"], files)):
        if a.drop_after is not None and k == a.drop_after:
            link.send(b"", flush=True)
            link.sock.shutdown(socket.SHUT_RDWR)
            link.sock.close()
            print(f"[fake-iphone] dropped the connection after {k} frames")
            return 0
        if a.cancel_after is not None and k == a.cancel_after:
            link.send(encode(CANCEL, {}), flush=True)
            time.sleep(0.5)
            link.sock.close()
            print(f"[fake-iphone] cancelled after {k} frames")
            return 0
        h = dict(f, image_bytes=len(img), depth_bytes=len(dep), confidence_bytes=len(conf))
        link.send(encode(FRAME, h, img + dep + conf))
        if a.fps > 0:
            link.send(b"", flush=True)
            time.sleep(max(0.0, t_start + (k + 1) / a.fps - time.time()))
    if ply:
        m = re.search(rb"element vertex (\d+)", ply)
        link.send(encode(POINTS, {"count": int(m.group(1)) if m else 0}, ply))
    link.send(encode(END, raw), flush=True)
    link.closed.wait(30)
    st = link.statuses[-1] if link.statuses else {}
    print(f"[fake-iphone] sent {link.sent} bytes; {len(link.statuses)} status replies, last {st}")
    ok = link.saved is not None and len(link.statuses) == len(doc["frames"]) and st.get("frames_received") == len(doc["frames"])
    if not ok:
        print("[fake-iphone] FAILED: no saved reply or status count mismatch", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
