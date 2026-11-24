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

/// Receives the exact bytes the writer saved locally (called on the writer's serial queue, in frame order).
/// Implementations must return quickly and never block (the live link only enqueues).
protocol CaptureWriterSink: AnyObject, Sendable {
    /// A keyframe was written and appended to capture.json's `frames`.
    func writerDidWrite(frame: FrameRecord, blobs: LiveFrameBlobs)
    /// points.ply (if any) and capture.json were written; `manifest` is the capture.json file content.
    func writerDidFinish(points: LivePoints?, manifest: Data)
    /// The recording was discarded (no keyframes).
    func writerDidDiscard()
    /// Finalizing failed (the local directory is kept as an incomplete capture).
    func writerDidFail(_ error: Error)
}

enum CaptureError: LocalizedError {
    case noFrames
    case noFramesKeptFiles
    case notRecording
    case directoryExists(String)

    var errorDescription: String? {
        switch self {
        case .noFrames: return "No keyframes were captured, so nothing was saved. Move the phone slowly while tracking is normal."
        case .noFramesKeptFiles: return "No keyframes were recorded successfully; the folder was kept as an incomplete capture because it contains files."
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
    private let sink: (any CaptureWriterSink)?
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

    init(directory: URL, metadata: CaptureMetadata, settings: CaptureSettings, sink: (any CaptureWriterSink)? = nil,
         maxInFlight: Int = 6) throws {
        let fm = FileManager.default
        if fm.fileExists(atPath: directory.path) { throw CaptureError.directoryExists(directory.lastPathComponent) }
        try fm.createDirectory(at: directory.appendingPathComponent(CaptureFormat.imagesDir), withIntermediateDirectories: true)
        self.directory = directory
        self.metadata = metadata
        self.settings = settings
        self.sink = sink
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
            let result = Result { try self.finalize(points: points) }
            if case .failure(let error) = result {
                switch error {
                case CaptureError.noFrames, CaptureError.noFramesKeptFiles: self.sink?.writerDidDiscard()
                default: self.sink?.writerDidFail(error)
                }
            }
            completion(result)
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
            var depthPNG: Data?, confidencePNG: Data?
            if let depth = p.depthMillimeters, let size = p.depthSize, depthSize == nil || depthSize == size {
                try ensureDepthDirs()
                if let png = ImageEncoding.pngGray16(depth, size: size) {
                    try png.write(to: directory.appendingPathComponent(CaptureFormat.depthPath(index)))
                    hasDepth = true
                    depthPNG = png
                    depthSize = size
                    if let conf = p.confidence, let cpng = ImageEncoding.pngGray8(conf, size: size) {
                        try cpng.write(to: directory.appendingPathComponent(CaptureFormat.confidencePath(index)))
                        hasConfidence = true
                        confidencePNG = cpng
                    }
                }
            }
            let t0 = firstTimestamp ?? p.timestamp
            firstTimestamp = t0
            imageSize = dstSize
            let record = FrameRecordBuilder.make(
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
                angularSpeed: Double(p.angularSpeed))
            frames.append(record)
            // Same bytes that were just written to disk.
            sink?.writerDidWrite(frame: record, blobs: LiveFrameBlobs(image: jpeg, depth: depthPNG, confidence: confidencePNG))
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
        guard !frames.isEmpty, let imageSize else {
            // Only ever delete a directory that holds no files (just the empty images/depth/confidence folders
            // this writer created). Anything else is kept and shows up as an incomplete capture.
            if Self.containsNoFiles(directory) {
                try? FileManager.default.removeItem(at: directory)
                throw CaptureError.noFrames
            }
            throw CaptureError.noFramesKeptFiles
        }
        var livePoints: LivePoints?
        if !points.isEmpty {
            let ply = Data(PointCloudPLY.ascii(points).utf8)   // ASCII; identical to the former String.write(.ascii)
            try ply.write(to: directory.appendingPathComponent(CaptureFormat.pointCloudFileName), options: .atomic)
            livePoints = LivePoints(count: points.count, ply: ply)
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
        let manifestData = try manifest.jsonData()
        try manifestData.write(to: directory.appendingPathComponent(CaptureFormat.manifestFileName), options: .atomic)
        sink?.writerDidFinish(points: livePoints, manifest: manifestData)
        return CaptureSummary(directory: directory, frameCount: frames.count, hasLidar: hasLidar,
                              pointCount: points.count, writeErrors: writeErrors)
    }

    /// True when `url` contains no regular files at any depth (empty sub-directories are fine).
    static func containsNoFiles(_ url: URL) -> Bool {
        let keys: [URLResourceKey] = [.isDirectoryKey]
        guard let e = FileManager.default.enumerator(at: url, includingPropertiesForKeys: keys, options: [],
                                                     errorHandler: { _, _ in true }) else { return false }
        for case let item as URL in e {
            if (try? item.resourceValues(forKeys: Set(keys)))?.isDirectory != true { return false }
        }
        return true
    }
}
