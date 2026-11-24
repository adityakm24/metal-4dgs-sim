// Host-side checks for the pure capture-format logic in M4DGSCapture/Core.
// Built and run on macOS by ../run_core_checks.sh (no simulator needed).
//
// Usage: core_checks <output_dir>
//   Writes sample files (capture.json, depth/confidence PNGs, points.ply) into <output_dir>
//   so the script can cross-check them with Python/OpenCV the way the loader will read them.

import CoreGraphics
import Foundation
import ImageIO
import simd

var failures = 0
@MainActor func check(_ cond: @autoclosure () -> Bool, _ msg: String, file: StaticString = #file, line: UInt = #line) {
    if cond() { return }
    failures += 1
    print("FAIL (line \(line)): \(msg)")
}
func approx(_ a: Double, _ b: Double, _ eps: Double = 1e-4) -> Bool { abs(a - b) <= eps }
func approx(_ a: [[Double]], _ b: [[Double]], _ eps: Double = 1e-4) -> Bool {
    a.count == b.count && zip(a, b).allSatisfy { $0.count == $1.count && zip($0, $1).allSatisfy { approx($0, $1, eps) } }
}

let outDir = URL(fileURLWithPath: CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : NSTemporaryDirectory())
try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

// MARK: - 4x4 row-major serialization
do {
    // Distinct entries: element (row r, col c) = 10*r + c. simd init takes columns.
    var m = simd_float4x4()
    for c in 0..<4 { for r in 0..<4 { m[c][r] = Float(10 * r + c) } }
    let rows = MatrixSerialization.rowMajor(m)
    let expected: [[Double]] = (0..<4).map { r in (0..<4).map { c in Double(10 * r + c) } }
    check(rows == expected, "rowMajor(4x4) wrong: \(rows)")

    // Real pose: 90° about +y, translated by (1, 2, 3). Translation must land in the last COLUMN of each row.
    let rot = simd_float4x4(simd_quatf(angle: .pi / 2, axis: SIMD3<Float>(0, 1, 0)))
    var pose = rot
    pose.columns.3 = SIMD4<Float>(1, 2, 3, 1)
    let p = MatrixSerialization.rowMajor(pose)
    check(approx(p, [[0, 0, 1, 1], [0, 1, 0, 2], [-1, 0, 0, 3], [0, 0, 0, 1]]), "rowMajor(pose) wrong: \(p)")
    // Row-major product check: R_rowmajor · (0,0,1) should equal R·z = (+1, 0, 0) for this rotation.
    let rz = (0..<3).map { p[$0][2] }
    check(approx(rz[0], 1) && approx(rz[1], 0) && approx(rz[2], 0), "rotated z axis wrong: \(rz)")
}

// MARK: - Intrinsics (3x3 row-major + scaling)
do {
    let fx: Float = 1450, fy: Float = 1451, cx: Float = 960.5, cy: Float = 719.5
    let k = simd_float3x3(columns: (SIMD3<Float>(fx, 0, 0), SIMD3<Float>(0, fy, 0), SIMD3<Float>(cx, cy, 1)))
    check(approx(MatrixSerialization.rowMajor(k), [[1450, 0, 960.5], [0, 1451, 719.5], [0, 0, 1]]),
          "rowMajor(K) wrong: \(MatrixSerialization.rowMajor(k))")
    let full = PixelSize(width: 1920, height: 1440)
    let half = full.scaled(by: 0.5)
    check(half == PixelSize(width: 960, height: 720), "half size wrong: \(half)")
    let kh = MatrixSerialization.scaleIntrinsics(k, from: full, to: half)
    check(approx(MatrixSerialization.rowMajor(kh), [[725, 0, 480.25], [0, 725.5, 359.75], [0, 0, 1]]),
          "scaled K wrong: \(MatrixSerialization.rowMajor(kh))")
    let k1 = MatrixSerialization.scaleIntrinsics(k, from: full, to: full)
    check(approx(MatrixSerialization.rowMajor(k1), MatrixSerialization.rowMajor(k)), "identity scaling changed K")
}

// MARK: - Naming / time
do {
    check(CaptureFormat.imagePath(0) == "images/000000.jpg", "imagePath(0)")
    check(CaptureFormat.depthPath(42) == "depth/000042.png", "depthPath(42)")
    check(CaptureFormat.confidencePath(1234567) == "confidence/1234567.png", "confidencePath(big)")
    let date = Date(timeIntervalSince1970: 1_795_515_300) // 2026-11-24T10:15:00Z
    check(CaptureFormat.iso8601UTC(date) == "2026-11-24T10:15:00Z", "iso8601: \(CaptureFormat.iso8601UTC(date))")
    check(CaptureFormat.directoryName(for: date, timeZone: TimeZone(identifier: "UTC")!) == "capture_20261124_101500",
          "dirname: \(CaptureFormat.directoryName(for: date, timeZone: TimeZone(identifier: "UTC")!))")
    check(TrackingStatus.limited("excessiveMotion").jsonString == "limited:excessiveMotion", "tracking string")
}

// MARK: - Depth conversion
do {
    check(DepthConversion.millimeters(fromMeters: 1.2345) == 1235 || DepthConversion.millimeters(fromMeters: 1.2345) == 1234, "1.2345 m")
    check(DepthConversion.millimeters(fromMeters: 0.5) == 500, "0.5 m")
    check(DepthConversion.millimeters(fromMeters: .nan) == 0, "NaN")
    check(DepthConversion.millimeters(fromMeters: .infinity) == 0, "inf")
    check(DepthConversion.millimeters(fromMeters: -1) == 0, "negative")
    check(DepthConversion.millimeters(fromMeters: 0) == 0, "zero")
    check(DepthConversion.millimeters(fromMeters: 100) == 65535, "clamp")
}

// MARK: - Keyframe selection
do {
    var s = KeyframeSelector(settings: KeyframeSettings())
    func pose(x: Float = 0, yawDeg: Float = 0) -> simd_float4x4 {
        var m = simd_float4x4(simd_quatf(angle: yawDeg * .pi / 180, axis: SIMD3<Float>(0, 1, 0)))
        m.columns.3 = SIMD4<Float>(x, 0, 0, 1)
        return m
    }
    var t = 0.0
    var d = s.evaluate(timestamp: t, pose: pose(), trackingNormal: true)
    check(d == .keep, "first frame should keep, got \(d)")
    s.markKept(timestamp: t, pose: pose())

    t += 0.05; d = s.evaluate(timestamp: t, pose: pose(x: 0.002), trackingNormal: true)
    check(d == .tooSoon, "too soon, got \(d)")
    t += 0.10; d = s.evaluate(timestamp: t, pose: pose(x: 0.01), trackingNormal: true)
    check(d == .tooClose, "too close, got \(d)")
    t += 0.10; d = s.evaluate(timestamp: t, pose: pose(x: 0.06), trackingNormal: true)
    check(d == .keep, "translation keep, got \(d)")
    s.markKept(timestamp: t, pose: pose(x: 0.06))
    t += 0.15; d = s.evaluate(timestamp: t, pose: pose(x: 0.06, yawDeg: 6), trackingNormal: true)
    check(d == .keep, "rotation keep, got \(d) (ω=\(s.angularSpeed))")
    s.markKept(timestamp: t, pose: pose(x: 0.06, yawDeg: 6))

    // 3° in 1/60 s ≈ 3.14 rad/s → motion blur.
    t += 1.0 / 60; d = s.evaluate(timestamp: t, pose: pose(x: 0.2, yawDeg: 9), trackingNormal: true)
    check(d == .motionBlur, "motion blur, got \(d)")
    check(abs(s.angularSpeed - Float(3 * Double.pi / 180 * 60)) < 0.01, "angular speed \(s.angularSpeed)")

    t += 0.5; d = s.evaluate(timestamp: t, pose: pose(x: 0.5, yawDeg: 9), trackingNormal: false)
    check(d == .trackingNotNormal, "tracking, got \(d)")

    var bad = pose(); bad.columns.3.x = .nan
    t += 0.5; d = s.evaluate(timestamp: t, pose: bad, trackingNormal: true)
    check(d == .trackingNotNormal, "non-finite pose, got \(d)")

    check(abs(KeyframeSelector.rotationAngle(pose(yawDeg: 0.1), pose(yawDeg: 0.2)) - 0.1 * .pi / 180) < 1e-5, "small angle")
    check(abs(KeyframeSelector.rotationAngle(pose(), pose(yawDeg: 179)) - 179 * .pi / 180) < 1e-4, "large angle")
}

// MARK: - capture.json
do {
    let k = simd_float3x3(columns: (SIMD3<Float>(1450, 0, 0), SIMD3<Float>(0, 1450, 0), SIMD3<Float>(960, 720, 1)))
    var pose = matrix_identity_float4x4
    pose.columns.3 = SIMD4<Float>(0.25, -0.5, 1.5, 1)
    let full = PixelSize(width: 1920, height: 1440), half = full.scaled(by: 0.5)
    let f0 = FrameRecordBuilder.make(index: 0, timestamp: 1000.25, firstTimestamp: 1000.25, hasDepth: true, hasConfidence: true,
                                     cameraTransform: pose, intrinsics: k, cameraResolution: full, storedSize: half,
                                     tracking: .normal, exposureDuration: 0.008, angularSpeed: 0.1)
    let f1 = FrameRecordBuilder.make(index: 1, timestamp: 1000.75, firstTimestamp: 1000.25, hasDepth: false, hasConfidence: false,
                                     cameraTransform: pose, intrinsics: k, cameraResolution: full, storedSize: half,
                                     tracking: .limited("relocalizing"), exposureDuration: .nan, angularSpeed: nil)
    let noLidar = CaptureManifest(deviceModel: "iPhone17,3", systemVersion: "18.2", createdAt: "2026-11-24T10:15:00Z",
                                  imageWidth: half.width, imageHeight: half.height, depthWidth: nil, depthHeight: nil,
                                  hasLidar: false, frames: [f1])
    let data = try noLidar.jsonData()
    let obj = try JSONSerialization.jsonObject(with: data) as! [String: Any]
    let expectedTop: Set<String> = ["format", "version", "device_model", "system_version", "created_at", "world_convention",
                                    "camera_convention", "image_width", "image_height", "depth_width", "depth_height",
                                    "has_lidar", "frames"]
    check(Set(obj.keys) == expectedTop, "top-level keys: \(obj.keys.sorted())")
    check(obj["format"] as? String == "m4dgs-arkit-capture", "format")
    check(obj["version"] as? Int == 1, "version")
    check(obj["depth_width"] is NSNull && obj["depth_height"] is NSNull, "depth size should be null without LiDAR")
    check(obj["has_lidar"] as? Bool == false, "has_lidar")
    let fr = (obj["frames"] as! [[String: Any]])[0]
    let expectedFrame: Set<String> = ["index", "timestamp", "image", "depth", "confidence", "camera_to_world", "intrinsics",
                                      "tracking_state", "exposure_duration"]
    check(Set(fr.keys) == expectedFrame, "frame keys (no blur score): \(fr.keys.sorted())")
    check(fr["depth"] is NSNull && fr["confidence"] is NSNull, "depth/confidence should be explicit null")
    check(fr["tracking_state"] as? String == "limited:relocalizing", "tracking_state")
    check(approx(fr["timestamp"] as! Double, 0.5), "relative timestamp")
    check(fr["exposure_duration"] as? Double == 0, "non-finite exposure sanitized")
    check(!String(decoding: data, as: UTF8.self).contains("\\/"), "slashes should not be escaped")

    let withLidar = CaptureManifest(deviceModel: "iPhone17,1", systemVersion: "18.2", createdAt: "2026-11-24T10:15:00Z",
                                    imageWidth: half.width, imageHeight: half.height, depthWidth: 256, depthHeight: 192,
                                    hasLidar: true, frames: [f0, f1])
    let data2 = try withLidar.jsonData()
    try data2.write(to: outDir.appendingPathComponent("capture.json"))
    let obj2 = try JSONSerialization.jsonObject(with: data2) as! [String: Any]
    let fr0 = (obj2["frames"] as! [[String: Any]])[0]
    check(fr0["image"] as? String == "images/000000.jpg", "image path")
    check(fr0["depth"] as? String == "depth/000000.png", "depth path")
    check(fr0["confidence"] as? String == "confidence/000000.png", "confidence path")
    check(approx(fr0["camera_to_world"] as! [[Double]], [[1, 0, 0, 0.25], [0, 1, 0, -0.5], [0, 0, 1, 1.5], [0, 0, 0, 1]]),
          "camera_to_world: \(fr0["camera_to_world"]!)")
    check(approx(fr0["intrinsics"] as! [[Double]], [[725, 0, 480], [0, 725, 360], [0, 0, 1]]), "intrinsics: \(fr0["intrinsics"]!)")
    check(approx(fr0["timestamp"] as! Double, 0), "t0")
    check(approx(fr0["motion_blur_score"] as! Double, 0.1, 1e-6), "blur score")
    let decoded = try JSONDecoder().decode(CaptureManifest.self, from: data2)
    check(decoded == withLidar, "Codable round trip")
}

// MARK: - PNG encoders (decoded again via ImageIO; Python/OpenCV re-checks the files)
do {
    let size = PixelSize(width: 4, height: 3)
    let depth: [UInt16] = [0, 1, 255, 256, 1000, 1234, 4095, 4096, 12345, 30000, 65534, 65535]
    let png16 = ImageEncoding.pngGray16(depth, size: size)!
    try FileManager.default.createDirectory(at: outDir.appendingPathComponent("depth"), withIntermediateDirectories: true)
    try png16.write(to: outDir.appendingPathComponent("depth/000000.png"))
    let src = CGImageSourceCreateWithData(png16 as CFData, nil)!
    let img = CGImageSourceCreateImageAtIndex(src, 0, nil)!
    check(img.width == 4 && img.height == 3 && img.bitsPerComponent == 16, "png16 decoded as \(img.width)x\(img.height)x\(img.bitsPerComponent)")
    check(String(decoding: png16.prefix(8).dropFirst(1).prefix(3), as: UTF8.self) == "PNG", "png16 signature")
    // IHDR: bit depth at byte 24, color type at byte 25 (0 = grayscale).
    check(png16[24] == 16 && png16[25] == 0, "png16 IHDR bitDepth=\(png16[24]) colorType=\(png16[25])")

    let conf: [UInt8] = [0, 1, 2, 2, 1, 0, 0, 1, 2, 2, 2, 2]
    let png8 = ImageEncoding.pngGray8(conf, size: size)!
    try FileManager.default.createDirectory(at: outDir.appendingPathComponent("confidence"), withIntermediateDirectories: true)
    try png8.write(to: outDir.appendingPathComponent("confidence/000000.png"))
    check(png8[24] == 8 && png8[25] == 0, "png8 IHDR bitDepth=\(png8[24]) colorType=\(png8[25])")
    check(ImageEncoding.pngGray16([1, 2], size: size) == nil, "size mismatch rejected")
}

// MARK: - PLY
do {
    let ply = PointCloudPLY.ascii([SIMD3<Float>(1, 2, 3), SIMD3<Float>(-0.5, 0.25, 1e-6)])
    try ply.write(to: outDir.appendingPathComponent("points.ply"), atomically: true, encoding: .ascii)
    let lines = ply.split(separator: "\n")
    check(lines.first == "ply" && lines.contains("element vertex 2") && lines.contains("end_header"), "ply header")
    check(lines.last == "-0.500000 0.250000 0.000001", "ply row: \(lines.last ?? "")")
}

// MARK: - Live link wire format (LiveLinkChecks.swift)
try runLiveLinkChecks(outDir: outDir)

if failures == 0 {
    print("core checks: all passed (samples in \(outDir.path))")
    exit(0)
} else {
    print("core checks: \(failures) failure(s)")
    exit(1)
}
