"""Independent decoder check for the live-link wire format (docs/formats.md section 6).

Parses ``stream.bin`` written by the Swift encoder (Tests/CoreChecks/LiveLinkChecks.swift) using only the spec
(stdlib ``struct`` + ``json``), and checks every payload is byte-identical to the files it stands for.

    python live_link_decode_check.py <dir containing stream.bin, frame0.jpg, ...>
"""

import json
import os
import struct
import sys

MAGIC = b"M4DG"
TYPES = {1: "hello", 2: "frame", 3: "points", 4: "end", 5: "status", 6: "cancel", 7: "saved"}
FRAME_KEYS = {"index", "timestamp", "image", "depth", "confidence", "camera_to_world", "intrinsics",
              "tracking_state", "exposure_duration", "motion_blur_score"}


def decode(buf):
    off, out = 0, []
    while off < len(buf):
        assert len(buf) - off >= 16, f"truncated prefix at {off}"
        magic, mtype, reserved, hlen, plen = struct.unpack_from(">4sB3sII", buf, off)
        assert magic == MAGIC, f"bad magic {magic!r} at {off}"
        assert reserved == b"\0\0\0", f"reserved bytes {reserved!r} at {off}"
        assert mtype in TYPES, f"unknown type {mtype} at {off}"
        start = off + 16
        assert start + hlen + plen <= len(buf), f"truncated message at {off}"
        header = buf[start:start + hlen]
        payload = buf[start + hlen:start + hlen + plen]
        out.append((TYPES[mtype], header, payload))
        off = start + hlen + plen
    return out


def main(d):
    def read(name):
        with open(os.path.join(d, name), "rb") as f:
            return f.read()

    msgs = decode(read("stream.bin"))
    names = [m[0] for m in msgs]
    assert names == ["hello", "frame", "frame", "points", "end", "status", "saved", "cancel"], names
    jh = [json.loads(h.decode("utf-8")) for _, h, _ in msgs]

    hello = jh[0]
    assert hello["protocol"] == 1 and hello["capture_name"] == "capture_20261124_101500", hello
    assert set(hello) == {"protocol", "capture_name", "device_model", "system_version", "image_width",
                          "image_height", "depth_width", "depth_height", "has_lidar"}, sorted(hello)
    assert hello["depth_width"] is None and hello["has_lidar"] is False and msgs[0][2] == b""

    cap_bytes = read("capture.json")
    cap = json.loads(cap_bytes)
    expected_parts = [
        (read("frame0.jpg"), read("frame0_depth.png"), read("frame0_confidence.png")),
        (read("frame1.jpg"), b"", b""),
    ]
    for i, (h, (_, _, payload)) in enumerate(zip(jh[1:3], msgs[1:3])):
        sizes = [h["image_bytes"], h["depth_bytes"], h["confidence_bytes"]]
        assert sum(sizes) == len(payload), (i, sizes, len(payload))
        parts = [payload[:sizes[0]], payload[sizes[0]:sizes[0] + sizes[1]], payload[sizes[0] + sizes[1]:]]
        assert tuple(parts) == expected_parts[i], f"frame {i} payload parts differ from the files"
        frame = {k: v for k, v in h.items() if not k.endswith("_bytes")}
        assert set(frame) <= FRAME_KEYS and {"index", "image", "camera_to_world", "intrinsics"} <= set(frame), sorted(frame)
        assert frame == cap["frames"][i], f"frame {i} header differs from capture.json frame object"
        for key, size in (("depth", sizes[1]), ("confidence", sizes[2])):
            assert (frame[key] is None) == (size == 0), (i, key, frame[key], size)

    assert jh[3] == {"count": 2} and msgs[3][2] == read("points.ply"), "points"
    assert msgs[3][2].startswith(b"ply\n")
    assert msgs[4][1] == cap_bytes and msgs[4][2] == b"", "end header must be capture.json byte for byte"
    assert jh[5] == {"frames_received": 2, "bytes_received": 5_000_000_123}, jh[5]
    assert jh[6]["frames"] == 2 and jh[6]["path"].endswith("capture_20261124_101500"), jh[6]
    assert jh[7] == {} and msgs[7][2] == b"", "cancel"
    print(f"python live-link decoder cross-check: all passed ({len(msgs)} messages)")


if __name__ == "__main__":
    main(sys.argv[1])
