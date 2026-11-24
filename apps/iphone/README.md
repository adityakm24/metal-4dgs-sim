# M4DGS Capture (iPhone)

SwiftUI + ARKit app that records posed RGB frames (plus LiDAR depth on Pro iPhones) and exports them in the
**ARKit capture format** defined in [`docs/formats.md` §2](../../docs/formats.md). The Python loader
(`capture/arkit.py`) and `scripts/train_4dgs.py` read that format.

- iOS 18+, iPhone only. ARKit world tracking (VIO) supplies the poses, which are written exactly as `ARFrame.camera.transform`.
- **iPhone 16 / 16 Plus** (no LiDAR): RGB + poses + intrinsics + sparse feature points. `has_lidar=false`, and `depth`/`confidence` are `null`.
- **iPhone 12 Pro – 16 Pro** (LiDAR): also writes `depth/*.png` (16-bit, mm, 0 = invalid) and `confidence/*.png` (0/1/2).

## Open, sign, run

1. Open `apps/iphone/M4DGSCapture.xcodeproj` in Xcode 16 or later. The project uses a file-system-synchronized
   group, so any file you add under `M4DGSCapture/` is picked up automatically.
2. Select the **M4DGSCapture** target → *Signing & Capabilities* → pick your **Team**. A free Personal Team works.
   If `com.example.m4dgscapture` is already taken, change the bundle identifier to something unique.
3. Connect the iPhone 16 by cable (or pair it over Wi-Fi). On the phone, turn on *Settings → Privacy & Security →
   Developer Mode* (this needs a restart), then select the phone as the run destination and press **Run**.
   With a Personal Team you also have to trust the developer certificate once, under *Settings → General → VPN & Device Management*.
4. Allow camera access when asked.

The Simulator has no ARKit, so the app shows "ARKit not available" there. You can still browse and share captures in the Simulator.

Command-line build (no signing):

```sh
xcodebuild -project apps/iphone/M4DGSCapture.xcodeproj -scheme M4DGSCapture \
  -destination 'generic/platform=iOS Simulator' CODE_SIGNING_ALLOWED=NO build
```

Host-side checks for the format/serialization code (row-major poses, intrinsics scaling, JSON keys/nulls, 16-bit
PNG round trip through OpenCV, keyframe selection) and the live-link wire format (every §6 message type round-tripped
through the streaming decoder, the send-queue policy, and an independent stdlib-only Python decoder
`Tests/live_link_decode_check.py` that re-parses the bytes the Swift encoder produced):

```sh
apps/iphone/Tests/run_core_checks.sh .venv/bin/python   # any python; cv2 + numpy are only needed for the OpenCV part
```

Device build from the command line (signing overrides on the command line only, not in the project):

```sh
xcodebuild -project apps/iphone/M4DGSCapture.xcodeproj -scheme M4DGSCapture -configuration Debug \
  -destination 'id=<device udid>' -derivedDataPath apps/iphone/build/device \
  DEVELOPMENT_TEAM=<team id> PRODUCT_BUNDLE_IDENTIFIER=<your bundle id> CODE_SIGN_STYLE=Automatic \
  -allowProvisioningUpdates build
xcrun devicectl device install app --device <device udid> apps/iphone/build/device/Build/Products/Debug-iphoneos/M4DGSCapture.app
```

## Using the app

- **Record**: tap the red button to start and tap it again to stop. When you stop, the app writes `points.ply` and `capture.json`.
  If the app goes to the background while recording, the capture is stopped and finalized automatically.
- **HUD**: tracking state (green = normal), LiDAR yes/no, elapsed time, keyframes kept, dropped frames
  (the writer queue was full), angular speed ω, the reason the last frame was kept or skipped, and a warning when tracking is limited.
- **Keyframes** (*Settings*): a frame is kept when the min interval has passed (default 0.1 s) **and** the phone has moved
  at least the min translation (5 cm) **or** turned at least the min rotation (5°). Frames are skipped while tracking isn't
  `normal`, or while the angular speed is above the max (default 1.0 rad/s), because those frames would be blurred.
- **Half resolution** stores 960×720 JPEGs instead of 1920×1440, and the intrinsics are scaled to match.
  **Autofocus off** keeps the focal length steadier.
- **Captures** (folder button): name, frame count, size and date. Tap the share button to zip the capture and open the share sheet.
  Swipe left and tap Delete, then confirm, to delete. Deleting removes the capture from the iPhone even after you shared it.
  Captures also appear in the **Files** app under *On My iPhone → M4DGS Capture*, so deleting or moving the folder there removes it too.
- **Stop with no keyframes**: the empty folder is removed. A folder that already contains files is never removed
  automatically; it stays in the list as *incomplete*.

### Capture tips

- **Move slowly and smoothly.** Keep ω well below the limit and avoid jerky turns. Motion blur hurts splat quality more than having fewer frames.
- **Translate, don't just rotate.** Walk around or along the scene, or orbit an object at 0.5–2 m. Turning in place
  gives no parallax: VIO drifts and the reconstruction has no depth to work from.
- **Overlap**: consecutive keyframes should share most of their view. Orbit an object 1–2 times at different heights,
  and sweep a room in overlapping passes.
- Start in a well-lit, textured area. Before you record, move the phone sideways a little until tracking shows *normal*.
  Avoid blank walls, mirrors, glass and moving people. 200–600 keyframes (about 1–3 minutes) is a good size.
- Keep the scene static (unless dynamics are the point), and hold exposure steady by not pointing straight at lights.

## Streaming to the Mac while recording (live link)

The app can send every kept keyframe over Wi-Fi to **M4DGS Studio** (`apps/macbook`) while you record. The Mac rebuilds
exactly the same `capture_YYYYMMDD_HHMMSS/` folder, so `train_4dgs.py` can use it as is. The protocol is in
[`docs/formats.md` §6](../../docs/formats.md). The iPhone **always** saves its own copy as well, whether or not the Mac is reachable.

1. Put the iPhone and the Mac on the same Wi-Fi network, and open M4DGS Studio with live receive on (see `apps/macbook/README.md`). The Mac advertises
   `_m4dgs-live._tcp` over Bonjour under its own name.
2. On the iPhone, open *Settings → Live link* and turn on **Stream to Mac**. The first time you do this, iOS asks for
   **local network** access. Allow it. If you denied it, turn it on again under *iOS Settings → Privacy & Security → Local Network*.
3. Pick your Mac in the **Mac** picker. If only one Mac is found, it is selected automatically. The choice is remembered.
4. Record as usual. A Wi-Fi line in the HUD shows the link state:
   - before recording: `Mac: searching for …` / `Mac: <name> ready`
   - while recording: `Mac: connected to <name> · sending n/m frames` (n = frames sent, m = keyframes kept). If Wi-Fi
     can't keep up it adds `lagging, k behind (kept locally)`. Frames are never skipped: they wait in a 64 MB memory
     queue, and beyond that they are read back from the saved files on the iPhone when it's their turn.
   - after Stop: `Mac: finishing …`, then **`Saved on Mac at <path> (n frames)`** once the Mac confirms.
   - `Mac stream failed: <reason>. Mac has k/m frames; the iPhone copy is complete.` if the Mac can't be reached
     (the Mac keeps what it got as `capture.partial.json`). Use the zip/AirDrop route below for that capture.
5. If the connection drops, the app reconnects up to 4 times and **re-sends the whole capture from the start** on the
   new connection. The Mac then writes it to a new folder (`<name>_1`) and keeps the broken one as partial.
6. If a recording ends with no keyframes, the iPhone discards it and sends `cancel`, and the Mac deletes its partial folder.

What goes over the wire: `hello` at Start; for each kept frame, its §2 frame object plus the **exact bytes the iPhone
saved** (JPEG, and the depth/confidence PNGs on LiDAR phones); then `points` (`points.ply`, if any) and `end`
(the same `capture.json` the iPhone wrote) at Stop. If the app is sent to the background while recording, the capture is
stopped, finalized, and streamed in the time iOS allows (about 30 s). If that isn't enough, the stream fails and
the iPhone copy is still complete.

## Getting the capture to the Mac without the live link

1. On the Captures screen, tap the share button for the capture. The app zips the capture folder with `NSFileCoordinator`,
   and the zip contains one top-level `capture_YYYYMMDD_HHMMSS/` folder.
2. Choose **AirDrop** and pick your Mac. The zip arrives in `~/Downloads`.
   You can also use *Save to Files*, or Finder → iPhone → Files → M4DGS Capture (file sharing is enabled).
3. Unzip it and train:

```sh
cd metal-4dgs-sim
unzip ~/Downloads/capture_20261124_101500.zip -d data/
python scripts/train_4dgs.py --data_dir data/capture_20261124_101500
```

## Output layout (docs/formats.md §2)

```text
capture_YYYYMMDD_HHMMSS/
├── capture.json            # written at Stop; all kept frames
├── images/000000.jpg ...   # capturedImage, landscape sensor orientation, JPEG q=0.9
├── depth/000000.png ...    # LiDAR only: uint16 millimeters, 0 = invalid
├── confidence/000000.png   # LiDAR only: uint8 ARConfidenceLevel 0/1/2
└── points.ply              # ARKit rawFeaturePoints (deduped by identifier), world frame, ascii x y z
```

- `camera_to_world` = `ARFrame.camera.transform`, written **row-major**: simd is column-major, so the translation is in the last column of rows 0–2.
  ARKit conventions: world +y up, camera +x right / +y up / +z backward. The loader converts both to OpenCV/z-up.
- `intrinsics` = `ARFrame.camera.intrinsics` scaled from `camera.imageResolution` to the stored JPEG size (`diag(sx, sy, 1)·K`).
- `timestamp` = `ARFrame.timestamp − t0`, where t0 is the first kept frame. `motion_blur_score` = angular speed (rad/s) when the frame was captured.
- `tracking_state` is always `"normal"` for kept frames, because other states are skipped.

## Code map

| File | Role |
| --- | --- |
| `M4DGSCapture/Core/CaptureFormat.swift` | Pure format code: `capture.json` Codable types, row-major/intrinsics serialization, paths, depth→mm, PLY |
| `M4DGSCapture/Core/KeyframeSelector.swift` | Pure keyframe/motion-blur selection |
| `M4DGSCapture/Core/ImageEncoding.swift` | ImageIO PNG16/PNG8/JPEG encoders |
| `M4DGSCapture/Capture/CaptureEngine.swift` | Owns the `ARSession`; delegate runs on a private serial queue; copies keyframe buffers and drops frames when the writer is backed up |
| `M4DGSCapture/Capture/CaptureWriter.swift` | Serial background writer: YCbCr→JPEG (CoreImage), depth/confidence PNGs, `points.ply` and `capture.json` at stop; hands the exact saved bytes to a `CaptureWriterSink` (the live link) |
| `M4DGSCapture/Capture/FramePacket.swift` | ARFrame → owned copies (pixel-buffer pool copy, depth/confidence arrays) |
| `M4DGSCapture/Capture/CaptureController.swift` | Main-actor UI model; starts a `LiveStreamer` per recording when streaming is on; background task while finishing |
| `M4DGSCapture/Core/LiveLinkProtocol.swift` | Pure §6 wire format: message framing (big-endian, magic `M4DG`), header types, encoder, streaming decoder |
| `M4DGSCapture/Core/LiveOutbox.swift` | Pure send-queue policy (ordered stream, bounded memory, spill to the local files, replay on reconnect) and HUD text |
| `M4DGSCapture/LiveLink/LiveLinkBrowser.swift` | `NWBrowser` for `_m4dgs-live._tcp`, list of Macs, local-network permission state |
| `M4DGSCapture/LiveLink/LiveStreamer.swift` | One capture → one `NWConnection`: send queue, `status`/`saved` parsing, timeouts, reconnect + replay, failure reporting |
| `M4DGSCapture/Captures/*` | Captures list, zip export and share sheet, delete |
| `M4DGSCapture/Views/*` | Capture screen + HUD, AR preview (`ARSCNView`), settings |
| `Tests/` | `swiftc`-built macOS checks of `Core/` (incl. live-link round trips), a Python decoder check of the wire bytes, and an OpenCV cross-check |
