import ARKit
import Foundation
import simd

/// Snapshot of engine state for the HUD (posted to the main actor at ~10 Hz).
struct EngineStatus: Sendable, Equatable {
    var tracking: TrackingStatus = .notAvailable
    var depthActive = false
    var angularSpeed: Float = 0
    var maxAngularSpeed: Float = SettingsDefault.maxAngularSpeed.asFloat
    var isRecording = false
    var elapsed: TimeInterval = 0
    var keptFrames = 0
    var droppedFrames = 0
    var featurePoints = 0
    var lastDecision: KeyframeDecision = .keep
}

private extension Double { var asFloat: Float { Float(self) } }

/// Owns the ARSession. All ARSession delegate callbacks and all mutable state below are confined to
/// the private serial `queue` (set as the session's delegateQueue), so ARFrames are processed off the
/// main thread and never retained past the callback.
final class CaptureEngine: NSObject, ARSessionDelegate, @unchecked Sendable {
    static var isSupported: Bool { ARWorldTrackingConfiguration.isSupported }
    static var supportsLiDAR: Bool { ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) }

    let session = ARSession()
    private let queue = DispatchQueue(label: "com.example.m4dgscapture.session", qos: .userInteractive)
    private let onStatus: @Sendable (EngineStatus) -> Void
    private let onError: @Sendable (String) -> Void

    // MARK: Confined to `queue`
    private final class Recording {
        let writer: CaptureWriter
        var startTimestamp: TimeInterval?
        var points: [UInt64: SIMD3<Float>] = [:]
        var kept = 0
        var dropped = 0
        init(writer: CaptureWriter) { self.writer = writer }
    }
    private var selector = KeyframeSelector()
    private var recording: Recording?
    private var status = EngineStatus()
    private var lastStatusPost: TimeInterval = -1
    private let copier = PixelBufferCopier()

    init(onStatus: @escaping @Sendable (EngineStatus) -> Void, onError: @escaping @Sendable (String) -> Void) {
        self.onStatus = onStatus
        self.onError = onError
        super.init()
        session.delegate = self
        session.delegateQueue = queue
    }

    /// (Re)start the session. Re-running without reset options keeps the world map / tracking.
    func run(autoFocus: Bool) {
        let config = ARWorldTrackingConfiguration()
        if Self.supportsLiDAR {
            config.frameSemantics.insert(.sceneDepth)
        }
        config.isAutoFocusEnabled = autoFocus
        config.planeDetection = []
        config.environmentTexturing = .none
        session.run(config)
    }

    func pause() {
        session.pause()
    }

    /// Image size the writer will store (and the LiDAR depth size) for the live-link `hello`, taken from the
    /// session's current frame; falls back to the configured video format before the first frame arrives.
    func streamGeometry(halfResolution: Bool) -> (image: PixelSize, depth: PixelSize?) {
        var image: PixelSize
        var depth: PixelSize?
        if let frame = session.currentFrame {
            image = PixelSize(width: CVPixelBufferGetWidth(frame.capturedImage), height: CVPixelBufferGetHeight(frame.capturedImage))
            if let map = frame.sceneDepth?.depthMap {
                depth = PixelSize(width: CVPixelBufferGetWidth(map), height: CVPixelBufferGetHeight(map))
            }
        } else {
            let r = (session.configuration as? ARWorldTrackingConfiguration)?.videoFormat.imageResolution
                ?? ARWorldTrackingConfiguration().videoFormat.imageResolution
            image = PixelSize(width: Int(r.width.rounded()), height: Int(r.height.rounded()))
        }
        if halfResolution { image = image.scaled(by: 0.5) }
        return (image, depth)
    }

    func startRecording(writer: CaptureWriter, settings: CaptureSettings) {
        queue.async {
            self.selector.settings = settings.keyframe
            self.selector.resetKeyframes()
            self.recording = Recording(writer: writer)
            self.status.isRecording = true
            self.status.elapsed = 0
            self.status.keptFrames = 0
            self.status.droppedFrames = 0
            self.status.featurePoints = 0
            self.status.maxAngularSpeed = settings.keyframe.maxAngularSpeed
            self.onStatus(self.status)
        }
    }

    /// Stops recording; frames already queued are still written, then points.ply and capture.json.
    func stopRecording(completion: @escaping @Sendable (Result<CaptureSummary, Error>) -> Void) {
        queue.async {
            guard let rec = self.recording else {
                completion(.failure(CaptureError.notRecording))
                return
            }
            self.recording = nil
            self.status.isRecording = false
            self.onStatus(self.status)
            let points = rec.points.sorted { $0.key < $1.key }.map(\.value)
            rec.writer.finish(points: points, completion: completion)
        }
    }

    // MARK: ARSessionDelegate (on `queue`)

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let camera = frame.camera
        let tracking = TrackingStatus(camera.trackingState)
        let pose = camera.transform
        let t = frame.timestamp
        let decision = selector.evaluate(timestamp: t, pose: pose, trackingNormal: tracking == .normal)

        status.tracking = tracking
        status.depthActive = frame.sceneDepth != nil
        status.angularSpeed = selector.angularSpeed

        if let rec = recording {
            if rec.startTimestamp == nil { rec.startTimestamp = t }
            status.elapsed = t - (rec.startTimestamp ?? t)

            if tracking == .normal, let raw = frame.rawFeaturePoints {
                let ids = raw.identifiers, pts = raw.points
                for i in 0..<min(ids.count, pts.count) {
                    rec.points[ids[i]] = pts[i]   // dedupe by identifier; keep the latest estimate
                }
            }

            status.lastDecision = decision
            if decision == .keep {
                if rec.writer.tryReserve() {
                    if let packet = makePacket(frame, tracking: tracking) {
                        rec.writer.enqueue(packet)
                        selector.markKept(timestamp: t, pose: pose)
                        rec.kept += 1
                    } else {
                        rec.writer.cancelReservation()
                        rec.dropped += 1
                    }
                } else {
                    rec.dropped += 1   // writer backed up: drop rather than stall the session
                }
            }
            status.keptFrames = rec.kept
            status.droppedFrames = rec.dropped
            status.featurePoints = rec.points.count
        }

        if t - lastStatusPost >= 0.1 || lastStatusPost < 0 {
            lastStatusPost = t
            onStatus(status)
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        onError(error.localizedDescription)
    }

    func sessionWasInterrupted(_ session: ARSession) {
        status.tracking = .notAvailable
        onStatus(status)
    }

    // MARK: Private

    private func makePacket(_ frame: ARFrame, tracking: TrackingStatus) -> FramePacket? {
        guard let image = copier.copy(frame.capturedImage) else { return nil }
        var depthMM: [UInt16]?
        var confidence: [UInt8]?
        var depthSize: PixelSize?
        if let depth = frame.sceneDepth, let (size, mm) = FrameCopy.depthMillimeters(depth.depthMap) {
            depthMM = mm
            depthSize = size
            if let confMap = depth.confidenceMap, let (csize, conf) = FrameCopy.confidence(confMap), csize == size {
                confidence = conf
            }
        }
        let camera = frame.camera
        return FramePacket(
            image: image,
            depthMillimeters: depthMM,
            confidence: confidence,
            depthSize: depthSize,
            timestamp: frame.timestamp,
            cameraTransform: camera.transform,
            intrinsics: camera.intrinsics,
            cameraResolution: PixelSize(width: Int(camera.imageResolution.width.rounded()),
                                        height: Int(camera.imageResolution.height.rounded())),
            tracking: tracking,
            exposureDuration: camera.exposureDuration,
            angularSpeed: selector.angularSpeed)
    }
}
