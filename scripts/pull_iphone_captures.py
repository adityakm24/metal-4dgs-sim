#!/usr/bin/env python
"""Copy captures from the M4DGS Capture app on a USB-connected iPhone into data/iphone/ (no AirDrop).

Works for development-signed builds (Xcode / devicectl installs), which expose their app container to
`xcrun devicectl`. Only complete captures (with capture.json) are copied; existing ones are skipped.
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

BUNDLE = "com.adityakm24.m4dgscapture"


def devicectl(*args) -> str:
    r = subprocess.run(["xcrun", "devicectl", *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip() or r.stdout.strip())
    return r.stdout


def physical_iphones() -> list[str]:
    with tempfile.NamedTemporaryFile(suffix=".json") as f:
        devicectl("list", "devices", "--json-output", f.name)
        doc = json.load(open(f.name))
    out = []
    for d in doc.get("result", {}).get("devices", []):
        hw = d.get("hardwareProperties", {})
        if hw.get("reality") == "physical" and hw.get("platform") == "iOS":
            out.append(hw.get("udid") or d.get("identifier"))
    return out


def list_files(device: str, bundle: str) -> list[str]:
    with tempfile.NamedTemporaryFile(suffix=".json") as f:
        devicectl("device", "info", "files", "--device", device, "--domain-type", "appDataContainer",
                  "--domain-identifier", bundle, "--subdirectory", "Documents", "--json-output", f.name)
        doc = json.load(open(f.name))
    files = []
    for item in doc.get("result", {}).get("files", []):
        rel = item.get("relativePath") or item.get("name") or ""
        if item.get("type", "") != "directory" and not rel.endswith("/"):
            files.append(rel)
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default=None, help="UDID (default: the only connected iPhone)")
    ap.add_argument("--bundle", default=BUNDLE)
    ap.add_argument("--out", default="data/iphone")
    a = ap.parse_args()
    device = a.device
    if device is None:
        phones = physical_iphones()
        if len(phones) != 1:
            sys.exit(f"found {len(phones)} connected iPhones; pass --device <udid>")
        device = phones[0]
    files = list_files(device, a.bundle)
    captures: dict[str, list[str]] = {}
    for rel in files:
        m = re.match(r"^(capture_[^/]+)/(.+)$", rel)
        if m:
            captures.setdefault(m.group(1), []).append(m.group(2))
    if not captures:
        print("no captures on the phone")
        return
    out = Path(a.out)
    for name, rels in sorted(captures.items()):
        if "capture.json" not in rels:
            print(f"skip {name}: incomplete (no capture.json yet)")
            continue
        dst = out / name
        if (dst / "capture.json").exists():
            print(f"skip {name}: already pulled")
            continue
        for rel in rels:
            (dst / rel).parent.mkdir(parents=True, exist_ok=True)
            devicectl("device", "copy", "from", "--device", device, "--domain-type", "appDataContainer",
                      "--domain-identifier", a.bundle, "--source", f"Documents/{name}/{rel}",
                      "--destination", str(dst / rel))
        print(f"pulled {name}: {len(rels)} files -> {dst}")


if __name__ == "__main__":
    main()
