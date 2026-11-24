import ARKit
import CoreVideo
import Foundation
import simd

extension TrackingStatus {
    init(_ state: ARCamera.TrackingState) {
        switch state {
        case .normal:
            self = .normal
        case .notAvailable:
            self = .notAvailable
        case .limited(let reason):
            switch reason {
            case .initializing: self = .limited("initializing")
            case .excessiveMotion: self = .limited("excessiveMotion")
            case .insufficientFeatures: self = .limited("insufficientFeatures")
            case .relocalizing: self = .limited("relocalizing")
            @unknown default: self = .limited("unknown")
            }
        }
    }
}

/// Everything the writer needs from one ARFrame, fully copied so the ARFrame (and ARKit's
/// pixel-buffer pool) is released as soon as the delegate callback returns.
struct FramePacket: @unchecked Sendable {   // CVPixelBuffer is owned exclusively by the packet
    var image: CVPixelBuffer                 // private copy of capturedImage (YCbCr 420f)
    var depthMillimeters: [UInt16]?
    var confidence: [UInt8]?
    var depthSize: PixelSize?
    var timestamp: TimeInterval
    var cameraTransform: simd_float4x4
    var intrinsics: simd_float3x3
    var cameraResolution: PixelSize          // resolution `intrinsics` refers to
    var tracking: TrackingStatus
    var exposureDuration: TimeInterval
    var angularSpeed: Float
}

enum FrameCopy {
    /// sceneDepth.depthMap (Float32 meters) → millimeters (0 = invalid).
    static func depthMillimeters(_ buffer: CVPixelBuffer) -> (PixelSize, [UInt16])? {
        guard CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_DepthFloat32 else { return nil }
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return nil }
        let w = CVPixelBufferGetWidth(buffer), h = CVPixelBufferGetHeight(buffer)
        let bpr = CVPixelBufferGetBytesPerRow(buffer)
        var out = [UInt16](repeating: 0, count: w * h)
        out.withUnsafeMutableBufferPointer { dst in
            for y in 0..<h {
                let row = (base + y * bpr).assumingMemoryBound(to: Float32.self)
                for x in 0..<w { dst[y * w + x] = DepthConversion.millimeters(fromMeters: row[x]) }
            }
        }
        return (PixelSize(width: w, height: h), out)
    }

    /// sceneDepth.confidenceMap (UInt8 ARConfidenceLevel 0/1/2).
    static func confidence(_ buffer: CVPixelBuffer) -> (PixelSize, [UInt8])? {
        guard CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_OneComponent8 else { return nil }
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return nil }
        let w = CVPixelBufferGetWidth(buffer), h = CVPixelBufferGetHeight(buffer)
        let bpr = CVPixelBufferGetBytesPerRow(buffer)
        var out = [UInt8](repeating: 0, count: w * h)
        out.withUnsafeMutableBytes { dst in
            for y in 0..<h {
                memcpy(dst.baseAddress! + y * w, base + y * bpr, w)
            }
        }
        return (PixelSize(width: w, height: h), out)
    }
}

/// Copies pixel buffers into buffers from our own pool (confined to the caller's serial queue).
final class PixelBufferCopier {
    private struct Key: Equatable { var width: Int; var height: Int; var format: OSType }
    private var pool: CVPixelBufferPool?
    private var key: Key?

    func copy(_ src: CVPixelBuffer) -> CVPixelBuffer? {
        let k = Key(width: CVPixelBufferGetWidth(src), height: CVPixelBufferGetHeight(src),
                    format: CVPixelBufferGetPixelFormatType(src))
        if pool == nil || key != k {
            let attrs: [String: Any] = [
                kCVPixelBufferWidthKey as String: k.width,
                kCVPixelBufferHeightKey as String: k.height,
                kCVPixelBufferPixelFormatTypeKey as String: k.format,
                kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),
            ]
            var newPool: CVPixelBufferPool?
            guard CVPixelBufferPoolCreate(nil, nil, attrs as CFDictionary, &newPool) == kCVReturnSuccess else { return nil }
            pool = newPool
            key = k
        }
        guard let pool else { return nil }
        var out: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(nil, pool, &out) == kCVReturnSuccess, let dst = out else { return nil }

        CVPixelBufferLockBaseAddress(src, .readOnly)
        CVPixelBufferLockBaseAddress(dst, [])
        defer {
            CVPixelBufferUnlockBaseAddress(dst, [])
            CVPixelBufferUnlockBaseAddress(src, .readOnly)
        }
        if CVPixelBufferIsPlanar(src) {
            let planes = CVPixelBufferGetPlaneCount(src)
            guard planes == CVPixelBufferGetPlaneCount(dst) else { return nil }
            for p in 0..<planes {
                guard let s = CVPixelBufferGetBaseAddressOfPlane(src, p),
                      let d = CVPixelBufferGetBaseAddressOfPlane(dst, p) else { return nil }
                copyRows(src: s, srcStride: CVPixelBufferGetBytesPerRowOfPlane(src, p),
                         dst: d, dstStride: CVPixelBufferGetBytesPerRowOfPlane(dst, p),
                         rows: CVPixelBufferGetHeightOfPlane(src, p))
            }
        } else {
            guard let s = CVPixelBufferGetBaseAddress(src), let d = CVPixelBufferGetBaseAddress(dst) else { return nil }
            copyRows(src: s, srcStride: CVPixelBufferGetBytesPerRow(src),
                     dst: d, dstStride: CVPixelBufferGetBytesPerRow(dst),
                     rows: CVPixelBufferGetHeight(src))
        }
        // Keep YCbCr matrix / color primaries / transfer function so CoreImage converts correctly.
        CVBufferPropagateAttachments(src, dst)
        return dst
    }

    private func copyRows(src: UnsafeMutableRawPointer, srcStride: Int, dst: UnsafeMutableRawPointer, dstStride: Int, rows: Int) {
        if srcStride == dstStride {
            memcpy(dst, src, srcStride * rows)
        } else {
            let n = min(srcStride, dstStride)
            for r in 0..<rows { memcpy(dst + r * dstStride, src + r * srcStride, n) }
        }
    }
}
