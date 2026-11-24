import Foundation
import simd

// Pure, platform-independent pieces of the ARKit capture format (docs/formats.md §2).
// This file must not import ARKit/UIKit so it can be compiled and checked on macOS
// (see apps/iphone/Tests/run_core_checks.sh).

enum CaptureFormat {
    static let formatName = "m4dgs-arkit-capture"
    static let version = 1
    static let worldConvention = "arkit"
    static let cameraConvention = "arkit"
    static let manifestFileName = "capture.json"
    static let pointCloudFileName = "points.ply"
    static let imagesDir = "images"
    static let depthDir = "depth"
    static let confidenceDir = "confidence"
    static let directoryPrefix = "capture_"

    /// `capture_YYYYMMDD_HHMMSS` (local time).
    static func directoryName(for date: Date, timeZone: TimeZone = .current) -> String {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = timeZone
        f.dateFormat = "yyyyMMdd_HHmmss"
        return directoryPrefix + f.string(from: date)
    }

    /// `2026-11-24T10:15:00Z`
    static func iso8601UTC(_ date: Date) -> String {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime]
        f.timeZone = TimeZone(identifier: "UTC")
        return f.string(from: date)
    }

    static func imagePath(_ index: Int) -> String { "\(imagesDir)/\(frameStem(index)).jpg" }
    static func depthPath(_ index: Int) -> String { "\(depthDir)/\(frameStem(index)).png" }
    static func confidencePath(_ index: Int) -> String { "\(confidenceDir)/\(frameStem(index)).png" }

    static func frameStem(_ index: Int) -> String {
        let s = String(index)
        return s.count >= 6 ? s : String(repeating: "0", count: 6 - s.count) + s
    }
}

struct PixelSize: Equatable, Hashable, Sendable {
    var width: Int
    var height: Int

    /// Integer (floor) scaling, at least 1 px.
    func scaled(by s: Double) -> PixelSize {
        PixelSize(width: max(1, Int((Double(width) * s).rounded(.down))),
                  height: max(1, Int((Double(height) * s).rounded(.down))))
    }
}

/// ARKit tracking state, mirrored without importing ARKit.
enum TrackingStatus: Equatable, Hashable, Sendable {
    case normal
    case limited(String)   // "initializing" | "excessiveMotion" | "insufficientFeatures" | "relocalizing" | "unknown"
    case notAvailable

    /// `"normal" | "limited:<reason>" | "notAvailable"` (docs/formats.md §2)
    var jsonString: String {
        switch self {
        case .normal: return "normal"
        case .limited(let reason): return "limited:\(reason)"
        case .notAvailable: return "notAvailable"
        }
    }
}

enum MatrixSerialization {
    /// simd matrices are column-major (`m[column][row]`); returns `rows[row][column]`.
    static func rowMajor(_ m: simd_float4x4) -> [[Double]] {
        (0..<4).map { r in (0..<4).map { c in Double(m[c][r]) } }
    }

    /// simd matrices are column-major (`m[column][row]`); returns `rows[row][column]`.
    static func rowMajor(_ m: simd_float3x3) -> [[Double]] {
        (0..<3).map { r in (0..<3).map { c in Double(m[c][r]) } }
    }

    /// Rescale a pinhole intrinsics matrix from the resolution it was defined for to the stored image
    /// resolution: `K' = diag(sx, sy, 1) · K` (fx, skew, cx scale with x; fy, cy with y). With the
    /// "pixel centers at integer + 0.5" convention this is exact for any resize of the full frame.
    static func scaleIntrinsics(_ k: simd_float3x3, from src: PixelSize, to dst: PixelSize) -> simd_float3x3 {
        guard src.width > 0, src.height > 0 else { return k }
        let sx = Float(Double(dst.width) / Double(src.width))
        let sy = Float(Double(dst.height) / Double(src.height))
        return simd_float3x3(diagonal: SIMD3<Float>(sx, sy, 1)) * k
    }

    static func isFinite(_ m: simd_float4x4) -> Bool {
        for c in 0..<4 { for r in 0..<4 where !m[c][r].isFinite { return false } }
        return true
    }
}

/// One entry of `capture.json` → `frames`.
struct FrameRecord: Codable, Equatable, Sendable {
    var index: Int
    var timestamp: Double
    var image: String
    var depth: String?
    var confidence: String?
    var cameraToWorld: [[Double]]
    var intrinsics: [[Double]]
    var trackingState: String
    var exposureDuration: Double
    var motionBlurScore: Double?

    enum CodingKeys: String, CodingKey {
        case index, timestamp, image, depth, confidence
        case cameraToWorld = "camera_to_world"
        case intrinsics
        case trackingState = "tracking_state"
        case exposureDuration = "exposure_duration"
        case motionBlurScore = "motion_blur_score"
    }

    // Explicit `null`s for depth/confidence (the contract says "or null"), rather than omitting keys.
    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(index, forKey: .index)
        try c.encode(timestamp, forKey: .timestamp)
        try c.encode(image, forKey: .image)
        try c.encode(depth, forKey: .depth)               // Optional → null
        try c.encode(confidence, forKey: .confidence)     // Optional → null
        try c.encode(cameraToWorld, forKey: .cameraToWorld)
        try c.encode(intrinsics, forKey: .intrinsics)
        try c.encode(trackingState, forKey: .trackingState)
        try c.encode(exposureDuration, forKey: .exposureDuration)
        try c.encodeIfPresent(motionBlurScore, forKey: .motionBlurScore)
    }
}

/// `capture.json`.
struct CaptureManifest: Codable, Equatable, Sendable {
    var format: String = CaptureFormat.formatName
    var version: Int = CaptureFormat.version
    var deviceModel: String
    var systemVersion: String
    var createdAt: String
    var worldConvention: String = CaptureFormat.worldConvention
    var cameraConvention: String = CaptureFormat.cameraConvention
    var imageWidth: Int
    var imageHeight: Int
    var depthWidth: Int?
    var depthHeight: Int?
    var hasLidar: Bool
    var frames: [FrameRecord]

    enum CodingKeys: String, CodingKey {
        case format, version
        case deviceModel = "device_model"
        case systemVersion = "system_version"
        case createdAt = "created_at"
        case worldConvention = "world_convention"
        case cameraConvention = "camera_convention"
        case imageWidth = "image_width"
        case imageHeight = "image_height"
        case depthWidth = "depth_width"
        case depthHeight = "depth_height"
        case hasLidar = "has_lidar"
        case frames
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(format, forKey: .format)
        try c.encode(version, forKey: .version)
        try c.encode(deviceModel, forKey: .deviceModel)
        try c.encode(systemVersion, forKey: .systemVersion)
        try c.encode(createdAt, forKey: .createdAt)
        try c.encode(worldConvention, forKey: .worldConvention)
        try c.encode(cameraConvention, forKey: .cameraConvention)
        try c.encode(imageWidth, forKey: .imageWidth)
        try c.encode(imageHeight, forKey: .imageHeight)
        try c.encode(depthWidth, forKey: .depthWidth)     // Optional → null
        try c.encode(depthHeight, forKey: .depthHeight)   // Optional → null
        try c.encode(hasLidar, forKey: .hasLidar)
        try c.encode(frames, forKey: .frames)
    }

    func jsonData() throws -> Data {
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        return try enc.encode(self)
    }
}

/// Builds one `FrameRecord` from raw ARKit-native values.
enum FrameRecordBuilder {
    /// - Parameters:
    ///   - cameraTransform: `ARFrame.camera.transform` (camera-to-world, ARKit conventions).
    ///   - intrinsics: `ARFrame.camera.intrinsics`, defined for `cameraResolution` (`camera.imageResolution`).
    ///   - storedSize: resolution of the JPEG actually written.
    static func make(index: Int,
                     timestamp: Double,
                     firstTimestamp: Double,
                     hasDepth: Bool,
                     hasConfidence: Bool,
                     cameraTransform: simd_float4x4,
                     intrinsics: simd_float3x3,
                     cameraResolution: PixelSize,
                     storedSize: PixelSize,
                     tracking: TrackingStatus,
                     exposureDuration: Double,
                     angularSpeed: Double?) -> FrameRecord {
        let k = MatrixSerialization.scaleIntrinsics(intrinsics, from: cameraResolution, to: storedSize)
        return FrameRecord(
            index: index,
            timestamp: timestamp - firstTimestamp,
            image: CaptureFormat.imagePath(index),
            depth: hasDepth ? CaptureFormat.depthPath(index) : nil,
            confidence: hasConfidence ? CaptureFormat.confidencePath(index) : nil,
            cameraToWorld: MatrixSerialization.rowMajor(cameraTransform),
            intrinsics: MatrixSerialization.rowMajor(k),
            trackingState: tracking.jsonString,
            exposureDuration: exposureDuration.isFinite ? exposureDuration : 0,
            motionBlurScore: angularSpeed.flatMap { $0.isFinite ? $0 : nil })
    }
}

/// ARKit `sceneDepth` (Float32 meters) → 16-bit millimeters, 0 = invalid.
enum DepthConversion {
    static func millimeters(fromMeters m: Float) -> UInt16 {
        guard m.isFinite, m > 0 else { return 0 }
        let mm = (m * 1000).rounded()
        if mm >= 65535 { return 65535 }
        return UInt16(mm)
    }
}

/// ASCII PLY with `x y z` vertices (world frame).
enum PointCloudPLY {
    static func ascii(_ points: [SIMD3<Float>]) -> String {
        var s = "ply\nformat ascii 1.0\nelement vertex \(points.count)\n"
        s += "property float x\nproperty float y\nproperty float z\nend_header\n"
        s.reserveCapacity(s.utf8.count + points.count * 32)
        for p in points {
            s += String(format: "%.6f %.6f %.6f\n", p.x, p.y, p.z)
        }
        return s
    }
}
