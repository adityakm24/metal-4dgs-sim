#!/usr/bin/env bash
# Builds the SwiftPM executable (release) and wraps it into "M4DGS Studio.app".
# Usage: ./build_app.sh [debug|release]   → apps/macbook/build/M4DGS Studio.app
set -euo pipefail
cd "$(dirname "$0")"
CONFIG="${1:-release}"
swift build -c "$CONFIG" --product M4DGSStudio
BIN="$(swift build -c "$CONFIG" --show-bin-path)/M4DGSStudio"
APP="build/M4DGS Studio.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BIN" "$APP/Contents/MacOS/M4DGSStudio"
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>M4DGS Studio</string>
  <key>CFBundleDisplayName</key><string>M4DGS Studio</string>
  <key>CFBundleIdentifier</key><string>dev.m4dgs.studio</string>
  <key>CFBundleExecutable</key><string>M4DGSStudio</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>0.1.0</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>LSMinimumSystemVersion</key><string>15.0</string>
  <key>LSApplicationCategoryType</key><string>public.app-category.developer-tools</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSPrincipalClass</key><string>NSApplication</string>
  <key>NSLocalNetworkUsageDescription</key><string>M4DGS Studio receives captures streamed from the M4DGS Capture iPhone app over your local network.</string>
  <key>NSBonjourServices</key><array><string>_m4dgs-live._tcp</string></array>
  <key>CFBundleDocumentTypes</key>
  <array>
    <dict>
      <key>CFBundleTypeName</key><string>4DGS Scene Checkpoint</string>
      <key>CFBundleTypeExtensions</key><array><string>safetensors</string></array>
      <key>CFBundleTypeRole</key><string>Viewer</string>
      <key>LSHandlerRank</key><string>Alternate</string>
    </dict>
  </array>
</dict>
</plist>
PLIST
# ad-hoc sign so Gatekeeper / TCC treat it as a stable app identity locally
codesign --force --sign - "$APP" >/dev/null 2>&1 || echo "warning: ad-hoc codesign failed (app still runnable)"
echo "built: $(pwd)/$APP"
echo "run:   open \"$APP\" --args \"$(pwd)/Fixtures/test_scene.safetensors\" --rollout \"$(pwd)/Fixtures/test_scene.rollout.json\""
