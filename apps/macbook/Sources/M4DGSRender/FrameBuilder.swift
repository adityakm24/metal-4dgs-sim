import Foundation
import simd
import ImageIO
import CoreGraphics
import UniformTypeIdentifiers
import M4DGSCore

public struct OverlayOptions: Equatable, Sendable {
    public var boxes = true
    public var actorTracks = true
    public var egoTrack = true
    public var originalTracks = true
    public var egoFrustum = true
    public var selected: String?
    public init() {}
    public static var none: OverlayOptions {
        var o = OverlayOptions(); o.boxes = false; o.actorTracks = false; o.egoTrack = false
        o.originalTracks = false; o.egoFrustum = false
        return o
    }
}

public enum CameraChoice: Sendable {
    case ego
    case free(OrbitCamera)
}

public enum OverlayColors {
    public static let box: SIMD4<Float> = [0.2, 0.85, 1.0, 0.95]
    public static let selected: SIMD4<Float> = [1.0, 0.85, 0.1, 1.0]
    public static let track: SIMD4<Float> = [0.2, 0.85, 1.0, 0.8]
    public static let original: SIMD4<Float> = [0.6, 0.6, 0.6, 0.6]
    public static let ego: SIMD4<Float> = [0.25, 1.0, 0.35, 0.9]
    public static let egoLogged: SIMD4<Float> = [0.6, 0.9, 0.6, 0.45]
    public static let collision: SIMD4<Float> = [1.0, 0.15, 0.15, 1.0]
}

/// Shared by the GUI and the headless renderer: turns scene state into a `RenderFrame`.
public struct FrameBuilder {
    public var graph: SceneGraph
    /// Effective actor tracks (edited and/or rollout-simulated). Missing entries use the graph's track.
    public var tracks: [String: [TrackKey]] = [:]
    /// Pre-edit tracks, drawn in grey when they differ from `tracks`.
    public var originalTracks: [String: [TrackKey]] = [:]
    /// Effective ego track (rollout steps or logged ego).
    public var egoTrack: [TrackKey]?
    /// Logged ego track drawn faintly when a rollout ego track is active.
    public var loggedEgoTrack: [TrackKey]?
    public var collidingActors: Set<String> = []
    public var overlays = OverlayOptions()
    public var background: SIMD3<Double>?

    public init(graph: SceneGraph) { self.graph = graph }

    public static let defaultCameraToEgo = RigidTransform(rowMajor: [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 1.5], [0, 0, 0, 1]])
    public var egoIntrinsics: Intrinsics {
        graph.ego?.intrinsics ?? Intrinsics(fx: 300, fy: 300, cx: 256, cy: 144, width: 512, height: 288)
    }
    public var cameraToEgo: RigidTransform { graph.ego?.cameraToEgo ?? Self.defaultCameraToEgo }

    public func sceneFrame(at t: Double) -> SceneFrame {
        SceneFrame.evaluate(graph: graph, time: t, trackOverrides: tracks, egoTrack: egoTrack)
    }

    /// Ego camera at native image resolution.
    public func egoCamera(at t: Double) -> CameraView {
        CameraView.ego(egoPose: sceneFrame(at: t).egoPose, cameraToEgo: cameraToEgo, intrinsics: egoIntrinsics)
    }

    public func build(time t: Double, camera: CameraChoice, targetWidth W: Int, targetHeight H: Int) -> RenderFrame {
        let sf = sceneFrame(at: t)
        let cam: CameraView
        let rect: SIMD4<Double>
        let egoCam = CameraView.ego(egoPose: sf.egoPose, cameraToEgo: cameraToEgo, intrinsics: egoIntrinsics)
        switch camera {
        case .ego:
            let (k, r) = egoIntrinsics.fitted(targetWidth: W, targetHeight: H)
            cam = CameraView(camToWorld: egoCam.camToWorld, intrinsics: k)
            rect = r
        case .free(let orbit):
            cam = CameraView(camToWorld: orbit.camToWorld,
                             intrinsics: .fromFov(verticalFov: orbit.verticalFov, width: W, height: H))
            rect = SIMD4(0, 0, Double(W), Double(H))
        }
        var f = RenderFrame(camera: cam, imageRect: rect, time: t, nodePoses: sf.nodePoses, nodeVisible: sf.nodeVisible)
        f.background = background ?? graph.backgroundColor ?? .zero
        f.lines = lines(sf, egoCamera: egoCam, isEgoView: { if case .ego = camera { return true } else { return false } }())
        return f
    }

    func lines(_ sf: SceneFrame, egoCamera: CameraView, isEgoView: Bool) -> [LineSegment] {
        var out: [LineSegment] = []
        let o = overlays
        let ground = 0.05
        for (i, n) in graph.nodes.enumerated() where n.isActor {
            let size = n.size ?? [4.5, 1.9, 1.6]
            let drop = size.z * 0.5 - ground
            let sel = n.name == o.selected
            if o.originalTracks, let orig = originalTracks[n.name], let cur = tracks[n.name], orig != cur {
                out += Overlay.track(orig, drop: drop, color: OverlayColors.original, width: 1.5)
            }
            if o.actorTracks || sel {
                let tr = tracks[n.name] ?? n.track
                out += Overlay.track(tr, drop: drop, color: sel ? OverlayColors.selected : OverlayColors.track, width: sel ? 3 : 2)
            }
            if o.boxes && sf.nodeVisible[i] {
                let c = collidingActors.contains(n.name) ? OverlayColors.collision : (sel ? OverlayColors.selected : OverlayColors.box)
                out += Overlay.box(pose: sf.nodePoses[i], size: size, color: c, width: sel ? 3 : 2)
            }
        }
        if o.egoTrack {
            if let logged = loggedEgoTrack { out += Overlay.track(logged, drop: -ground, color: OverlayColors.egoLogged, width: 1.5) }
            let et = egoTrack ?? graph.ego?.track ?? []
            out += Overlay.track(et, drop: -ground, color: OverlayColors.ego, width: 2.5)
        }
        if !isEgoView {
            if o.boxes {
                let size = graph.ego?.size ?? [4.5, 1.9, 1.6]
                var p = sf.egoPose
                p.pos += p.up * (size.z * 0.5)
                out += Overlay.box(pose: p, size: size, color: OverlayColors.ego, width: 2)
            }
            if o.egoFrustum { out += Overlay.frustum(egoCamera, depth: 3, color: OverlayColors.ego) }
        }
        return out
    }
}

public enum PNGWriter {
    public static func write(rgba: [UInt8], width: Int, height: Int, to url: URL) throws {
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        let data = Data(rgba) as CFData
        guard let provider = CGDataProvider(data: data),
              let img = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
                                space: cs, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
              let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            throw NSError(domain: "PNGWriter", code: 1, userInfo: [NSLocalizedDescriptionKey: "cannot create PNG \(url.path)"])
        }
        CGImageDestinationAddImage(dest, img, nil)
        guard CGImageDestinationFinalize(dest) else {
            throw NSError(domain: "PNGWriter", code: 2, userInfo: [NSLocalizedDescriptionKey: "cannot write PNG \(url.path)"])
        }
    }
}
