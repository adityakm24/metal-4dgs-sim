#!/bin/sh
# Compiles the platform-independent capture-format and live-link code (M4DGSCapture/Core) together with
# CoreChecks/*.swift for macOS, runs it, re-parses the live-link wire bytes with an independent Python decoder
# (live_link_decode_check.py, stdlib only), then (if available) re-reads the sample files with Python + OpenCV
# the same way the capture loader does.
#
#   apps/iphone/Tests/run_core_checks.sh [python]
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
CORE="$HERE/../M4DGSCapture/Core"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/m4dgs_core_checks.XXXXXX")"
PY="${1:-python3}"

xcrun swiftc -O -swift-version 6 -o "$WORK/core_checks" "$CORE"/*.swift "$HERE"/CoreChecks/*.swift
"$WORK/core_checks" "$WORK/sample"

# Independent (pure-Python, stdlib only) decode of the live-link bytes produced by the Swift encoder.
if "$PY" -c "import sys" 2>/dev/null; then
  "$PY" "$HERE/live_link_decode_check.py" "$WORK/sample/live"
else
  echo "python not found ($PY); skipped live-link decoder cross-check"
fi

if "$PY" -c "import cv2, numpy" 2>/dev/null; then
  "$PY" - "$WORK/sample" <<'EOF'
import json, sys, os
import cv2, numpy as np
d = sys.argv[1]
cap = json.load(open(os.path.join(d, "capture.json")))
assert cap["format"] == "m4dgs-arkit-capture" and cap["version"] == 1
f0 = cap["frames"][0]
T = np.array(f0["camera_to_world"], dtype=np.float64)
assert T.shape == (4, 4) and np.allclose(T[:3, 3], [0.25, -0.5, 1.5]) and np.allclose(T[3], [0, 0, 0, 1]), T
K = np.array(f0["intrinsics"]); assert K.shape == (3, 3) and np.allclose(K, [[725, 0, 480], [0, 725, 360], [0, 0, 1]]), K
assert cap["frames"][1]["depth"] is None and cap["frames"][1]["confidence"] is None
dep = cv2.imread(os.path.join(d, f0["depth"]), cv2.IMREAD_UNCHANGED)
assert dep.dtype == np.uint16 and dep.shape == (3, 4), (dep.dtype, dep.shape)
assert dep.ravel().tolist() == [0, 1, 255, 256, 1000, 1234, 4095, 4096, 12345, 30000, 65534, 65535], dep.ravel().tolist()
conf = cv2.imread(os.path.join(d, f0["confidence"]), cv2.IMREAD_UNCHANGED)
assert conf.dtype == np.uint8 and conf.shape == (3, 4) and conf.ravel().tolist() == [0, 1, 2, 2, 1, 0, 0, 1, 2, 2, 2, 2]
lines = open(os.path.join(d, "points.ply")).read().splitlines()
hdr = lines.index("end_header"); pts = np.array([[float(v) for v in l.split()] for l in lines[hdr + 1:]])
assert pts.shape == (2, 3)
print("python/opencv cross-check: all passed")
EOF
else
  echo "python with cv2+numpy not found ($PY); skipped OpenCV cross-check"
fi
rm -rf "$WORK"
