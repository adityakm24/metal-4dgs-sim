"""3-panel live telemetry: [raw 4DGS render + depth/flow insets | refined view | BEV + signals + latency]."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import cv2
import numpy as np

from eval.metrics.safety import box_corners_2d

PANEL = (512, 288)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _u8(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def depth_vis(depth: np.ndarray, max_d: float = 60.0) -> np.ndarray:
    inv = np.where(depth > 0, 1.0 / np.maximum(depth, 0.5), 0.0)
    return cv2.applyColorMap((np.clip(inv / 2.0, 0, 1) ** 0.5 * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)


def flow_vis(flow: np.ndarray, scale: float = 20.0) -> np.ndarray:
    mag, ang = cv2.cartToPolar(flow[..., 0].astype(np.float32), flow[..., 1].astype(np.float32))
    hsv = np.zeros(flow.shape[:2] + (3,), np.uint8)
    hsv[..., 0] = (ang * 90 / np.pi).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / scale * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _label(img, text, org=(8, 18), color=(255, 255, 255)):
    cv2.putText(img, text, org, FONT, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, 0.45, color, 1, cv2.LINE_AA)


class TelemetryUI:
    def __init__(self, record: str | None = None, show: bool = False, fps: float = 10.0, history: int = 80):
        self.writer = None
        self.record = record
        self.show = show
        self.fps = fps
        self.hist = {k: deque(maxlen=history) for k in ("speed", "steer", "accel", "ttc")}
        self.trail = deque(maxlen=400)

    def _bev(self, rec, scene, sim, size=(256, 288)) -> np.ndarray:
        w, h = size
        img = np.full((h, w, 3), 32, np.uint8)
        ego = rec.ego
        ppm = 4.0   # pixels per metre; ego-centred, heading up

        def to_px(xy):
            d = np.asarray(xy, dtype=np.float64) - [ego.x, ego.y]
            c, s = np.cos(-ego.yaw + np.pi / 2), np.sin(-ego.yaw + np.pi / 2)
            r = d @ np.array([[c, s], [-s, c]])
            return np.stack([w / 2 + r[..., 0] * ppm, h * 0.72 - r[..., 1] * ppm], -1).astype(np.int32)

        route = to_px(sim.route.pts)
        cv2.polylines(img, [route.reshape(-1, 1, 2)], False, (90, 90, 90), 14)
        cv2.polylines(img, [route.reshape(-1, 1, 2)], False, (140, 140, 140), 1)
        self.trail.append((ego.x, ego.y))
        if len(self.trail) > 1:
            cv2.polylines(img, [to_px(np.array(self.trail)).reshape(-1, 1, 2)], False, (80, 200, 255), 1)
        for n in scene.actors:
            if not n.visible(rec.t) or n.track is None:
                continue
            pts = to_px(box_corners_2d(n.track.pose(rec.t), n.size))
            hit = any(e == f"collision:{n.name}" for e in rec.events)
            cv2.fillPoly(img, [pts], (60, 60, 230) if hit else (200, 150, 60))
        cv2.fillPoly(img, [to_px(box_corners_2d(ego.pose(), scene.ego.size))], (80, 220, 80))
        _label(img, "BEV", (6, 16))
        return img

    def _plot(self, img, x0, y0, w, h, vals, lo, hi, color, name):
        cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (70, 70, 70), 1)
        v = [x for x in vals if x is not None and np.isfinite(x)]
        if len(v) > 1:
            pts = np.array([[x0 + i * w / (len(vals) - 1), y0 + h - (np.clip(x, lo, hi) - lo) / (hi - lo) * h]
                            for i, x in enumerate(vals) if x is not None and np.isfinite(x)], np.int32)
            cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, color, 1, cv2.LINE_AA)
        last = vals[-1] if vals else float("nan")
        _label(img, f"{name} {last:.2f}" if last is not None and np.isfinite(last) else f"{name} -", (x0 + 3, y0 + 13), color)

    def compose(self, rec, frames, scene, sim) -> np.ndarray:
        raw = cv2.resize(_u8(frames["raw"]), PANEL)
        ins_w, ins_h = PANEL[0] // 4, PANEL[1] // 4
        raw[-ins_h:, :ins_w] = cv2.resize(depth_vis(frames["depth"]), (ins_w, ins_h))
        if frames.get("flow") is not None:
            raw[-ins_h:, ins_w:2 * ins_w] = cv2.resize(flow_vis(frames["flow"]), (ins_w, ins_h))
        _label(raw, f"4DGS raster  {rec.latency_ms['raster']:.1f} ms")
        _label(raw, "depth", (4, PANEL[1] - ins_h + 12))
        _label(raw, "flow", (ins_w + 4, PANEL[1] - ins_h + 12))
        ref = cv2.resize(_u8(frames["refined"]), PANEL)
        _label(ref, f"Refined  {rec.latency_ms['refine']:.1f} ms")
        if any(e.startswith("collision") for e in rec.events):
            cv2.rectangle(ref, (0, 0), (PANEL[0] - 1, PANEL[1] - 1), (0, 0, 255), 6)
            _label(ref, "COLLISION", (PANEL[0] // 2 - 45, PANEL[1] // 2), (80, 80, 255))
        tel = np.full((PANEL[1], PANEL[0], 3), 24, np.uint8)
        tel[:, :256] = self._bev(rec, scene, sim)
        for k, v in (("speed", rec.ego.v), ("steer", rec.action.steer), ("accel", rec.action.accel),
                     ("ttc", min(rec.min_ttc, 10.0))):
            self.hist[k].append(v)
        x0, w = 262, 244
        self._plot(tel, x0, 6, w, 48, list(self.hist["speed"]), 0, 20, (120, 220, 120), "speed m/s")
        self._plot(tel, x0, 60, w, 48, list(self.hist["steer"]), -0.5, 0.5, (240, 200, 90), "steer rad")
        self._plot(tel, x0, 114, w, 48, list(self.hist["accel"]), -8, 3, (90, 160, 250), "accel m/s2")
        self._plot(tel, x0, 168, w, 40, list(self.hist["ttc"]), 0, 10, (120, 120, 250), "TTC s")
        # latency bars
        y = 222
        total = rec.latency_ms.get("total", 0.0)
        for k, col in (("raster", (90, 200, 250)), ("refine", (250, 160, 90)), ("policy", (150, 250, 150))):
            v = rec.latency_ms.get(k, 0.0)
            cv2.rectangle(tel, (x0 + 60, y - 8), (x0 + 60 + int(min(v, 60) / 60 * (w - 60)), y), col, -1)
            _label(tel, f"{k[:6]} {v:4.1f}", (x0, y))
            y += 14
        _label(tel, f"t={rec.t:4.1f}s  step {total:.1f} ms  ({1000 / max(total, 1e-3):.0f} FPS)", (x0, y + 4))
        if rec.events:
            _label(tel, " ".join(rec.events)[:40], (4, PANEL[1] - 8), (80, 80, 255))
        return np.concatenate([raw, ref, tel], axis=1)

    def __call__(self, rec, frames, scene, sim):
        frame = self.compose(rec, frames, scene, sim)
        if self.record:
            if self.writer is None:
                Path(self.record).parent.mkdir(parents=True, exist_ok=True)
                self.writer = cv2.VideoWriter(self.record, cv2.VideoWriter_fourcc(*"mp4v"), self.fps,
                                              (frame.shape[1], frame.shape[0]))
            self.writer.write(frame)
        if self.show:
            cv2.imshow("metal-4dgs-sim", frame)
            cv2.waitKey(1)
        self.last = frame

    def close(self):
        if self.writer is not None:
            self.writer.release()
        if self.show:
            cv2.destroyAllWindows()
