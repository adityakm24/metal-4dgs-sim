import CoreImage
import CoreVideo
import Foundation
import os

struct CaptureSummary: Sendable {
    var directory: URL
    var frameCount: Int
    var hasLidar: Bool
    var pointCount: Int
    var writeErrors: Int
}

enum CaptureError: LocalizedError {
    case noFrames
    case notRecording
    case directoryExists(String)

    var errorDescription: String? {
        switch self {
        case .noFrames: return "No keyframes were captured, so nothing was saved. Move the phone slowly while tracking is normal."
        case .notRecording: return "Not recording."
        case .directoryExists(let name): return "\(name) already exists."
        }
    }
}

/// Encodes and writes keyframes on its own serial background queue, then writes points.ply and
/// capture.json on `finish`. Thread-safety: `tryReserve`/`cancelReservation`/`enqueue`/`finish` may be
/// called from any thread; everything else is confined to `queue`.
final class CaptureWriter: @unchecked Sendable {
    let directory: URL
    private let metadata: CaptureMetadata
    private let settings: CaptureSettings
    private let queue = DispatchQueue(label: "com.example.m4dgscapture.writer", qos: .utility)
    private let inFlight = OSAllocatedUnfairLock(initialState: 0)
    private let maxInFlight: Int
    private let ciContext = CIContext(options: [.cacheIntermediates: false])
    private let sRGB = CGColorSpace(name: CGColorSpace.sRGB)!

    // Confined to `queue`.
    private var frames: [FrameRecord] = []
    private var firstTimestamp: TimeInterval?
    private var imageSize: PixelSize?
    private var depthSize: PixelSize?
    private var createdDepthDirs = false
    private var writeErrors = 0

    init(directory: URL, metadata: CaptureMetadata, settings: CaptureSettings, maxInFlight: Int = 6) throws {
        let fm = FileManager.default
        if fm.fileExists(atPath: directory.path) { throw CaptureError.directoryExists(directory.lastPathComponent) }
        try fm.createDirectory(at: directory.appendingPathComponent(CaptureFormat.imagesDir), withIntermediateDirectories: true)
        self.directory = directory
        self.metadata = metadata
        self.settings = settings
        self.maxInFlight = maxInFlight
    }

    /// Reserve a slot for one frame. Returns false when the writer is backed up (the caller drops the frame).
    func tryReserve() -> Bool {
        inFlight.withLock { n in
            guard n < maxInFlight else { return false }
            n += 1
            return true
        }
    }

    func cancelReservation() {
        inFlight.withLock { $0 -= 1 }
    }

    /// Enqueue a frame previously reserved with `tryReserve`.
    func enqueue(_ packet: FramePacket) {
        queue.async {
            defer { self.inFlight.withLock { $0 -= 1 } }
            autoreleasepool { self.write(packet) }
        }
    }

    func finish(points: [SIMD3<Float>], completion: @escaping @Sendable (Result<CaptureSummary, Error>) -> Void) {
        queue.async {
            completion(Result { try self.finalize(points: points) })
        }
    }

    // MARK: - Queue-confined

    private func write(_ p: FramePacket) {
        let index = frames.count
        let srcSize = PixelSize(width: CVPixelBufferGetWidth(p.image), height: CVPixelBufferGetHeight(p.image))
        let dstSize = settings.halfResolution ? srcSize.scaled(by: 0.5) : srcSize
        if let imageSize, imageSize != dstSize {
            // Resolution changed mid-capture (should not happen with a fixed video format); keep the file
            // set consistent with the single image_width/height in capture.json.
            writeErrors += 1
            return
        }
        guard let jpeg = encodeJPEG(p.image, size: dstSize, quality: settings.jpegQuality) else {
            writeErrors += 1
            return
        }
        do {
            try jpeg.write(to: directory.appendingPathComponent(CaptureFormat.imagePath(index)))
            var hasDepth = false, hasConfidence = false
            if let depth = p.depthMillimeters, let size = p.depthSize, depthSize == nil || depthSize == size {
                try ensureDepthDirs()
                if let png = ImageEncoding.pngGray16(depth, size: size) {
                    try png.write(to: directory.appendingPathComponent(CaptureFormat.depthPath(index)))
                    hasDepth = true
                    depthSize = size
                    if let conf = p.confidence, let cpng = ImageEncoding.pngGray8(conf, size: size) {
                        try cpng.write(to: directory.appendingPathComponent(CaptureFormat.confidencePath(index)))
                        hasConfidence = true
                    }
                }
            }
            let t0 = firstTimestamp ?? p.timestamp
            firstTimestamp = t0
            imageSize = dstSize
            frames.append(FrameRecordBuilder.make(
                index: index,
                timestamp: p.timestamp,
                firstTimestamp: t0,
                hasDepth: hasDepth,
                hasConfidence: hasConfidence,
                cameraTransform: p.cameraTransform,
                intrinsics: p.intrinsics,
                cameraResolution: p.cameraResolution,
                storedSize: dstSize,
                tracking: p.tracking,
                exposureDuration: p.exposureDuration,
                angularSpeed: Double(p.angularSpeed)))
        } catch {
            writeErrors += 1
        }
    }

    private func ensureDepthDirs() throws {
        guard !createdDepthDirs else { return }
        let fm = FileManager.default
        try fm.createDirectory(at: directory.appendingPathComponent(CaptureFormat.depthDir), withIntermediateDirectories: true)
        try fm.createDirectory(at: directory.appendingPathComponent(CaptureFormat.confidenceDir), withIntermediateDirectories: true)
        createdDepthDirs = true
    }

    /// YCbCr capturedImage → (optionally downscaled) sRGB JPEG, same (landscape sensor) orientation.
    private func encodeJPEG(_ buffer: CVPixelBuffer, size: PixelSize, quality: Double) -> Data? {
        var image = CIImage(cvPixelBuffer: buffer)
        let extent = image.extent
        if Int(extent.width) != size.width || Int(extent.height) != size.height {
            let sx = CGFloat(size.width) / extent.width, sy = CGFloat(size.height) / extent.height
            image = image.transformed(by: CGAffineTransform(scaleX: sx, y: sy), highQualityDownsample: true)
        }
        let rect = CGRect(x: 0, y: 0, width: size.width, height: size.height)
        guard let cg = ciContext.createCGImage(image, from: rect, format: .RGBA8, colorSpace: sRGB) else { return nil }
        return ImageEncoding.jpeg(cg, quality: quality)
    }

    private func finalize(points: [SIMD3<Float>]) throws -> CaptureSummary {
        let fm = FileManager.default
        guard !frames.isEmpty, let imageSize else {
            try? fm.removeItem(at: directory)
            throw CaptureError.noFrames
        }
        if !points.isEmpty {
            try PointCloudPLY.ascii(points).write(to: directory.appendingPathComponent(CaptureFormat.pointCloudFileName),
                                                  atomically: true, encoding: .ascii)
        }
        let hasLidar = depthSize != nil
        let manifest = CaptureManifest(
            deviceModel: metadata.deviceModel,
            systemVersion: metadata.systemVersion,
            createdAt: metadata.createdAt,
            imageWidth: imageSize.width,
            imageHeight: imageSize.height,
            depthWidth: depthSize?.width,
            depthHeight: depthSize?.height,
            hasLidar: hasLidar,
            frames: frames)
        try manifest.jsonData().write(to: directory.appendingPathComponent(CaptureFormat.manifestFileName), options: .atomic)
        return CaptureSummary(directory: directory, frameCount: frames.count, hasLidar: hasLidar,
                              pointCount: points.count, writeErrors: writeErrors)
    }
}
