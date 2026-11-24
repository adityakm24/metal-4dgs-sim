import SwiftUI
import AppKit
import Observation
import M4DGSCore

struct CompletedCapture: Identifiable, Hashable {
    var id: String { directory.path }
    var name: String
    var directory: URL
    var frames: Int
}

/// State behind the Live Capture window: receiver status, the capture in flight, finished captures and the
/// reconstruction job. Settings are written through to `StudioSettings` (UserDefaults).
@Observable @MainActor
final class LiveCaptureModel {
    enum CaptureState: Equatable { case idle, receiving, saved, cancelled, dropped, failed(String) }

    // MARK: receiver
    var listening = false
    var port: UInt16?
    var serviceName: String?
    var listenerError: String?

    // MARK: current capture
    var endpoint: String?
    var device: String?
    var captureName: String?
    var captureDirectory: URL?
    var captureState: CaptureState = .idle
    var framesReceived = 0
    var bytesReceived = 0
    var thumbnail: NSImage?
    var lastFrameInfo: String?
    /// ARKit camera centers (x, z) and forward direction (x, z) of the frames received so far (top-down, +y up).
    var trajectory: [(pos: SIMD2<Double>, fwd: SIMD2<Double>)] = []
    var warnings: [String] = []
    var events: [String] = []

    // MARK: captures + reconstruction
    var completed: [CompletedCapture] = []
    var selectedCapture: CompletedCapture?
    var iterations: Int = StudioSettings.iterations {
        didSet { StudioSettings.defaults.set(iterations, forKey: StudioSettings.iterationsKey) }
    }
    var reconstructing = false
    var reconstructCapture: String?
    var progress: Double = 0
    var lastProgress: TrainProgress?
    var log: [String] = []
    var reconstructResult: String?
    var lastCheckpoint: URL?

    // MARK: settings
    var repoRootPath: String = StudioSettings.defaults.string(forKey: StudioSettings.repoRootKey) ?? "" {
        didSet { StudioSettings.set(StudioSettings.repoRootKey, path: repoRootPath); receiver.capturesFolder = capturesFolder }
    }
    var capturesFolderPath: String = StudioSettings.defaults.string(forKey: StudioSettings.capturesFolderKey) ?? "" {
        didSet { StudioSettings.set(StudioSettings.capturesFolderKey, path: capturesFolderPath); receiver.capturesFolder = capturesFolder }
    }

    var repoRoot: URL? { _ = repoRootPath; return StudioSettings.repoRoot }
    var capturesFolder: URL { _ = capturesFolderPath; _ = repoRootPath; return StudioSettings.capturesFolder }
    var detectedRepoRoot: URL? { StudioSettings.detectRepoRoot() }

    @ObservationIgnored private let receiver: LiveReceiver
    @ObservationIgnored private var job: ReconstructJob?
    @ObservationIgnored private var currentConnection: Int?
    @ObservationIgnored private var lastThumbTime: CFTimeInterval = 0
    @ObservationIgnored private var pendingThumb: Data?
    @ObservationIgnored private var thumbScheduled = false
    /// Called with a finished checkpoint (opens it in the viewer).
    @ObservationIgnored var openCheckpoint: ((URL) -> Void)?

    init() {
        receiver = LiveReceiver(capturesFolder: StudioSettings.capturesFolder)
        receiver.onEvent = { [weak self] e in
            DispatchQueue.main.async { MainActor.assumeIsolated { self?.handle(e) } }
        }
    }

    // MARK: receiver control

    func startListening() {
        guard !listening else { return }
        listenerError = nil
        receiver.capturesFolder = capturesFolder
        do {
            try receiver.start()
            listening = true
        } catch {
            listenerError = "\(error)"
        }
    }

    func stopListening() {
        receiver.stop()
        listening = false
        port = nil
    }

    private func note(_ s: String) {
        let f = DateFormatter.localizedString(from: Date(), dateStyle: .none, timeStyle: .medium)
        events.append("\(f)  \(s)")
        if events.count > 200 { events.removeFirst(events.count - 200) }
    }

    private func handle(_ e: LiveReceiver.Event) {
        switch e {
        case .listening(let p, let name):
            listening = true; port = p; serviceName = serviceName ?? name
            note("Listening on port \(p)")
        case .serviceRegistered(let name):
            serviceName = name
            note("Bonjour: \"\(name)\" (\(LiveLink.serviceType))")
        case .listenerFailed(let s):
            listenerError = s; note("Listener: \(s)")
            if !s.hasPrefix("waiting") { listening = false; port = nil }
        case .stopped:
            listening = false; port = nil; note("Stopped listening")
        case .connected(let id, let ep):
            note("#\(id) connected from \(ep)")
            if captureState != .receiving {
                currentConnection = id
                endpoint = ep
            }
        case .protocolError(let id, let s):
            note("#\(id) error: \(s)")
            if id == currentConnection { captureState = .failed(s) }
        case .disconnected(let id, let state, let partial):
            note("#\(id) disconnected" + (partial.map { " — partial capture in \($0.lastPathComponent)" } ?? ""))
            if id == currentConnection, state == .receiving || state == .failed || state == .awaitingHello {
                if case .failed = captureState {} else if captureState == .receiving { captureState = .dropped }
            }
        case .session(let id, let se):
            handleSession(id, se)
        }
    }

    private func handleSession(_ id: Int, _ e: LiveSessionEvent) {
        switch e {
        case .started(let h, let dir):
            currentConnection = id
            captureName = dir.lastPathComponent
            captureDirectory = dir
            device = [h.deviceModel, h.systemVersion.map { "iOS \($0)" }].compactMap { $0 }.joined(separator: " · ")
                + ((h.imageWidth).map { "  ·  \($0)×\(h.imageHeight ?? 0)" } ?? "") + (h.hasLidar == true ? " · LiDAR" : "")
            captureState = .receiving
            framesReceived = 0; bytesReceived = 0; trajectory = []; thumbnail = nil; warnings = []; lastFrameInfo = nil
            note("#\(id) capture \(h.captureName) started → \(dir.path)")
        case .frame(let f):
            guard id == currentConnection else { return }
            framesReceived = f.framesReceived
            bytesReceived = f.bytesReceived
            if let m = f.cameraToWorld {
                trajectory.append((SIMD2(m[0][3], m[2][3]), SIMD2(-m[0][2], -m[2][2])))
            }
            lastFrameInfo = String(format: "#%d  t=%.2f s  %@", f.index ?? -1, f.timestamp ?? .nan, f.trackingState ?? "")
            if let d = f.imageData { scheduleThumbnail(d) }
        case .points(let n, let b):
            note("#\(id) points: \(n.map(String.init) ?? "?") (\(ByteCountFormatter.string(fromByteCount: Int64(b), countStyle: .file)))")
        case .saved(let dir, let n, let w):
            warnings = w
            if id == currentConnection { captureState = .saved }
            let c = CompletedCapture(name: dir.lastPathComponent, directory: dir, frames: n)
            completed.removeAll { $0.id == c.id }
            completed.insert(c, at: 0)
            selectedCapture = c
            note("#\(id) saved \(n) frames → \(dir.path)")
        case .cancelled(let dir):
            if id == currentConnection { captureState = .cancelled }
            note("#\(id) cancelled on the phone; deleted \(dir.lastPathComponent)")
        case .ignored(let s):
            note("#\(id) ignored \(s)")
        }
    }

    /// Decodes at most ~8 thumbnails per second (always the latest frame).
    private func scheduleThumbnail(_ d: Data) {
        pendingThumb = d
        guard !thumbScheduled else { return }
        thumbScheduled = true
        let wait = max(0, 0.125 - (CACurrentMediaTime() - lastThumbTime))
        DispatchQueue.main.asyncAfter(deadline: .now() + wait) { [weak self] in
            MainActor.assumeIsolated {
                guard let self else { return }
                self.thumbScheduled = false
                self.lastThumbTime = CACurrentMediaTime()
                if let d = self.pendingThumb { self.thumbnail = NSImage(data: d) }
                self.pendingThumb = nil
            }
        }
    }

    // MARK: captures on disk

    func addExistingCapturePanel() {
        let p = NSOpenPanel()
        p.canChooseDirectories = true
        p.canChooseFiles = false
        p.directoryURL = capturesFolder
        p.message = "Choose a capture directory (containing capture.json)"
        guard p.runModal() == .OK, let u = p.url else { return }
        guard FileManager.default.fileExists(atPath: u.appendingPathComponent("capture.json").path) else {
            reconstructResult = "\(u.lastPathComponent) has no capture.json"
            return
        }
        let n = ((try? JSONSerialization.jsonObject(with: Data(contentsOf: u.appendingPathComponent("capture.json")))) as? [String: Any])
            .flatMap { ($0["frames"] as? [Any])?.count } ?? 0
        let c = CompletedCapture(name: u.lastPathComponent, directory: u, frames: n)
        completed.removeAll { $0.id == c.id }
        completed.insert(c, at: 0)
        selectedCapture = c
    }

    func choosePath(directoryFor keyPath: ReferenceWritableKeyPath<LiveCaptureModel, String>, message: String) {
        let p = NSOpenPanel()
        p.canChooseDirectories = true
        p.canChooseFiles = false
        p.canCreateDirectories = true
        p.message = message
        if p.runModal() == .OK, let u = p.url { self[keyPath: keyPath] = u.path }
    }

    // MARK: reconstruction

    func reconstruct(_ c: CompletedCapture) {
        guard !reconstructing else { return }
        guard let repo = repoRoot else {
            reconstructResult = "Repo root not set (Settings → repo containing scripts/train_4dgs.py)"
            return
        }
        let cfg = ReconstructJob.Config(repoRoot: repo, captureDir: c.directory, iterations: max(1, iterations))
        let j = ReconstructJob(cfg)
        let iters = cfg.iterations
        log = ["$ " + cfg.commandLine]
        progress = 0; lastProgress = nil; reconstructResult = nil
        j.onLine = { [weak self] line, p in
            DispatchQueue.main.async {
                MainActor.assumeIsolated {
                    guard let self else { return }
                    self.log.append(line)
                    if self.log.count > 4000 { self.log.removeFirst(self.log.count - 4000) }
                    if let p { self.lastProgress = p; self.progress = min(1, Double(p.iteration) / Double(max(iters, 1))) }
                }
            }
        }
        j.onExit = { [weak self] status, cancelled in
            DispatchQueue.main.async {
                MainActor.assumeIsolated {
                    guard let self else { return }
                    self.reconstructing = false
                    self.job = nil
                    let ok = status == 0 && FileManager.default.fileExists(atPath: cfg.output.path)
                    if cancelled {
                        self.reconstructResult = "Cancelled"
                    } else if ok {
                        self.progress = 1
                        self.lastCheckpoint = cfg.output
                        self.reconstructResult = "Done → \(cfg.output.lastPathComponent)"
                        self.openCheckpoint?(cfg.output)
                    } else {
                        self.reconstructResult = "Failed (exit \(status)) — see log"
                    }
                    self.log.append(self.reconstructResult ?? "")
                }
            }
        }
        do {
            try j.start()
            job = j
            reconstructing = true
            reconstructCapture = c.name
        } catch {
            reconstructResult = "\(error)"
            log.append("error: \(error)")
        }
    }

    func cancelReconstruction() { job?.cancel() }
}
