import SwiftUI
import AppKit
import Observation
import simd
import UniformTypeIdentifiers
import M4DGSCore
import M4DGSRender

enum CameraMode: String, CaseIterable, Identifiable {
    case ego = "Ego"
    case free = "Free"
    var id: String { rawValue }
}

@Observable @MainActor
final class AppModel {
    // MARK: scene + edits
    var scene: LoadedScene?
    /// Scene graph the edits are applied on top of (checkpoint graph, or the loaded `.edits.json`).
    var baseGraph: SceneGraphDocument?
    var edits: [String: TrackEdit] = [:] {
        didSet { if edits != oldValue { recomputeTracks(); dirty = true } }
    }
    private(set) var editedTracks: [String: [TrackKey]] = [:]
    var dirty = false
    var selectedNode: String?
    var statusMessage: String?
    var errorMessage: String?
    var isLoading = false

    // MARK: time
    var time: Double = 0
    var playing = false
    var playbackSpeed: Double = 1
    var loop = true

    // MARK: view
    var cameraMode: CameraMode = .ego
    var orbit = OrbitCamera()
    var followEgo = true
    var overlays = OverlayOptions()
    var showInspector = true
    var useSceneBackground = true
    var customBackground = Color.black

    // MARK: rollout
    var rollout: Rollout?
    var rolloutURL: URL?
    var playRollout = true

    // MARK: stats (written from the render loop)
    var gpuMs: Double = 0
    var fps: Double = 0
    var visibleSplats = 0

    @ObservationIgnored let renderer: SplatRenderer?
    @ObservationIgnored var sceneGeneration = 0
    @ObservationIgnored var lastTick: CFTimeInterval?

    init() {
        do { renderer = try SplatRenderer() } catch {
            renderer = nil
            errorMessage = "Metal renderer unavailable: \(error)"
        }
    }

    var graph: SceneGraph? { baseGraph?.graph }
    var actors: [SceneNode] { graph?.actors ?? [] }
    var rolloutActive: Bool { playRollout && rollout != nil && !(rollout?.steps.isEmpty ?? true) }

    var timeRange: ClosedRange<Double> {
        if rolloutActive, let r = rollout { return r.timeRange }
        return graph?.timeRange ?? 0...1
    }

    var currentStep: RolloutStep? {
        guard let r = rollout, let i = r.stepIndex(at: time) else { return nil }
        return r.steps[i]
    }

    var editsURL: URL? { scene.map { $0.editsApplied ?? LoadedScene.editsURL(for: $0.url) } }

    var title: String {
        guard let s = scene else { return "M4DGS Studio" }
        return s.url.lastPathComponent + (dirty ? " — edited" : "")
    }

    // MARK: loading

    func openScenePanel() {
        let p = NSOpenPanel()
        p.allowedContentTypes = [UTType(filenameExtension: "safetensors") ?? .data]
        p.allowsOtherFileTypes = true
        p.message = "Open a 4DGS scene checkpoint (.safetensors)"
        if p.runModal() == .OK, let u = p.url { openScene(u) }
    }

    func openRolloutPanel() {
        let p = NSOpenPanel()
        p.allowedContentTypes = [.json]
        p.message = "Open a closed-loop rollout (*.rollout.json)"
        if p.runModal() == .OK, let u = p.url { openRollout(u) }
    }

    func openScene(_ url: URL, applyEdits: Bool = true, then: (() -> Void)? = nil) {
        isLoading = true
        statusMessage = "Loading \(url.lastPathComponent)…"
        Task {
            do {
                let s = try await Task.detached(priority: .userInitiated) {
                    try LoadedScene(url: url, applyEdits: applyEdits)
                }.value
                try self.renderer?.setScene(s.cloud)
                self.scene = s
                self.sceneGeneration += 1
                self.baseGraph = s.loaded
                self.edits = [:]
                self.recomputeTracks()
                self.dirty = false
                self.selectedNode = s.graph.actors.first?.name
                self.time = s.graph.timeRange.lowerBound
                if let ego = s.graph.ego?.track.first?.pose {
                    self.orbit = OrbitCamera(target: ego.pos + ego.forward * 10, yaw: .pi + 0.6, pitch: 0.5, distance: 35)
                }
                self.statusMessage = "Loaded \(s.cloud.count) Gaussians" + (s.editsApplied.map { " + edits \($0.lastPathComponent)" } ?? "")
                self.errorMessage = nil
                NSDocumentController.shared.noteNewRecentDocumentURL(url)
                then?()
            } catch {
                self.errorMessage = "Could not open \(url.lastPathComponent): \(error)"
            }
            self.isLoading = false
        }
    }

    func openRollout(_ url: URL) {
        do {
            let r = try Rollout(url: url)
            rollout = r
            rolloutURL = url
            playRollout = true
            time = r.timeRange.lowerBound
            statusMessage = "Rollout: \(r.steps.count) steps"
            if scene == nil, let s = r.resolveScene(relativeTo: url) { openScene(s) }
        } catch {
            errorMessage = "Could not open rollout: \(error)"
        }
    }

    // MARK: edits

    func recomputeTracks() {
        guard let g = graph else { editedTracks = [:]; return }
        var out: [String: [TrackKey]] = [:]
        for (name, e) in edits where !e.isIdentity {
            if let n = g.node(named: name) { out[name] = TrackEditor.apply(e, to: n.track, timeRange: g.timeRange) }
        }
        editedTracks = out
    }

    func binding(for actor: String) -> Binding<TrackEdit> {
        Binding(get: { self.edits[actor] ?? TrackEdit() }, set: { self.edits[actor] = $0 })
    }

    func isEdited(_ name: String) -> Bool { editedTracks[name] != nil }

    func resetEdits(_ actor: String? = nil) {
        if let a = actor { edits[a] = nil } else { edits = [:] }
    }

    func saveEdits() {
        guard let base = baseGraph, let url = editsURL else { return }
        do {
            let data = try base.jsonData(replacingTracks: editedTracks)
            try data.write(to: url, options: .atomic)
            dirty = false
            statusMessage = "Saved \(url.lastPathComponent)"
        } catch {
            errorMessage = "Save failed: \(error)"
        }
    }

    /// Reloads the checkpoint ignoring any `.edits.json` sidecar (the sidecar file is left on disk).
    func revertToCheckpoint() {
        guard let s = scene else { return }
        let t = time
        openScene(s.url, applyEdits: false) { self.time = t; self.statusMessage = "Reverted to checkpoint scene graph (sidecar not applied)" }
    }

    // MARK: per-frame

    func tick(now: CFTimeInterval) {
        defer { lastTick = now }
        guard let last = lastTick else { return }
        let dt = min(now - last, 0.1)
        guard playing, scene != nil else { return }
        let r = timeRange
        var t = time + dt * playbackSpeed
        if t > r.upperBound {
            if loop { t = r.lowerBound } else { t = r.upperBound; playing = false }
        }
        time = t
    }

    func step(_ dir: Double) {
        let dt = rolloutActive ? (rollout?.dt ?? 0.1) : 0.1
        time = min(max(time + dir * dt, timeRange.lowerBound), timeRange.upperBound)
    }

    func frameBuilder() -> FrameBuilder? {
        guard let s = scene, let g = baseGraph?.graph else { return nil }
        // node order must match the uploaded cloud (built from the loaded graph)
        var fb = FrameBuilder(graph: g.nodes.map(\.name) == s.graph.nodes.map(\.name) ? g : s.graph)
        var tracks = editedTracks
        var originals: [String: [TrackKey]] = [:]
        for name in editedTracks.keys { originals[name] = g.node(named: name)?.track }
        if rolloutActive, let r = rollout {
            for (k, v) in r.actorTracks { tracks[k] = v }
            fb.egoTrack = r.egoTrack
            fb.loggedEgoTrack = g.ego?.track
            if let st = currentStep {
                fb.collidingActors = Set(st.events.filter { $0.hasPrefix("collision") }.compactMap {
                    $0.split(separator: ":").dropFirst().first.map(String.init)
                })
            }
        }
        fb.tracks = tracks
        fb.originalTracks = originals
        var o = overlays
        o.selected = selectedNode
        fb.overlays = o
        if !useSceneBackground, let c = NSColor(customBackground).usingColorSpace(.sRGB) {
            fb.background = SIMD3(Double(c.redComponent), Double(c.greenComponent), Double(c.blueComponent))
        }
        return fb
    }

    /// Switches to the free camera starting from the current ego view.
    func enterFreeCamera() {
        guard cameraMode == .ego else { return }
        if let fb = frameBuilder() {
            orbit = OrbitCamera.matching(fb.egoCamera(at: time).camToWorld, distance: 12)
        }
        followEgo = false
        cameraMode = .free
    }

    /// Selects the actor whose projected box contains the clicked pixel (nearest first).
    func pick(pixel: SIMD2<Double>, camera: CameraView) {
        guard let fb = frameBuilder() else { return }
        let sf = fb.sceneFrame(at: time)
        var best: (String, Double)?
        for (i, n) in fb.graph.nodes.enumerated() where n.isActor && sf.nodeVisible[i] {
            let h = (n.size ?? [4.5, 1.9, 1.6]) * 0.5
            var lo = SIMD2<Double>(.infinity, .infinity), hi = -lo
            var depth = Double.infinity
            var ok = true
            for c in 0..<8 {
                let s = SIMD3<Double>(c & 1 == 0 ? -1 : 1, c & 2 == 0 ? -1 : 1, c & 4 == 0 ? -1 : 1)
                let p = camera.project(sf.nodePoses[i].apply(s * h))
                if !(p.z > 0) { ok = false; break }
                lo = simd_min(lo, SIMD2(p.x, p.y)); hi = simd_max(hi, SIMD2(p.x, p.y)); depth = min(depth, p.z)
            }
            if ok, pixel.x >= lo.x - 4, pixel.x <= hi.x + 4, pixel.y >= lo.y - 4, pixel.y <= hi.y + 4,
               depth < (best?.1 ?? .infinity) {
                best = (n.name, depth)
            }
        }
        if let b = best { selectedNode = b.0 }
    }
}
