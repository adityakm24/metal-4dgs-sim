import ARKit
import Observation
import SwiftUI
import UIKit

/// Main-actor model for the capture screen. Wraps `CaptureEngine` (which is nil when ARKit world
/// tracking is unavailable, e.g. in the Simulator).
@MainActor
@Observable
final class CaptureController {
    let isSupported: Bool
    let hasLiDAR: Bool
    private(set) var status = EngineStatus()
    private(set) var isRecording = false
    private(set) var isFinishing = false
    var message: String?
    /// Live link state of the current (or last) capture's stream; nil when not streaming.
    private(set) var liveStatus: LiveStreamSnapshot?
    let browser = LiveLinkBrowser()

    @ObservationIgnored private var engine: CaptureEngine?
    @ObservationIgnored private var isRunning = false
    /// Streams still running (a stream may outlive its recording while it finishes sending).
    @ObservationIgnored private var streamers: [UUID: LiveStreamer] = [:]
    @ObservationIgnored private var currentStreamID: UUID?
    @ObservationIgnored private var backgroundTask: UIBackgroundTaskIdentifier = .invalid

    var session: ARSession? { engine?.session }

    init() {
        let supported = CaptureEngine.isSupported
        isSupported = supported
        hasLiDAR = supported && CaptureEngine.supportsLiDAR
        guard supported else { return }
        engine = CaptureEngine(
            onStatus: { [weak self] s in
                Task { @MainActor in self?.status = s }
            },
            onError: { [weak self] text in
                Task { @MainActor in self?.message = "AR session error: \(text)" }
            })
    }

    func resume() {
        guard let engine else { return }
        engine.run(autoFocus: CaptureSettings.load().autoFocus)
        isRunning = true
    }

    func pause() {
        guard let engine, isRunning, !isRecording, !isFinishing else { return }
        engine.pause()
        isRunning = false
    }

    func toggleRecording() {
        if isRecording { stopRecording() } else { startRecording() }
    }

    func startRecording() {
        guard let engine, !isRecording, !isFinishing else { return }
        let now = Date()
        let settings = CaptureSettings.load()
        let directory = CaptureStore.capturesDirectory.appendingPathComponent(CaptureFormat.directoryName(for: now), isDirectory: true)
        let metadata = CaptureMetadata(deviceModel: DeviceInfo.modelIdentifier,
                                       systemVersion: UIDevice.current.systemVersion,
                                       createdAt: CaptureFormat.iso8601UTC(now))
        var note: String?
        let streamer = makeStreamer(directory: directory, metadata: metadata, settings: settings, note: &note)
        do {
            let writer = try CaptureWriter(directory: directory, metadata: metadata, settings: settings, sink: streamer)
            if !isRunning { resume() }
            engine.startRecording(writer: writer, settings: settings)
            isRecording = true
            message = note
            UIApplication.shared.isIdleTimerDisabled = true
            if let streamer {
                streamers[streamer.id] = streamer
                currentStreamID = streamer.id
                liveStatus = LiveStreamSnapshot(macName: streamer.macName)
                streamer.start()
            } else {
                currentStreamID = nil
                liveStatus = nil
            }
        } catch {
            streamer?.abandon()
            message = "Could not start capture: \(error.localizedDescription)"
        }
    }

    /// A streamer for this capture when "Stream to Mac" is on and a Mac is selected. Local saving never depends on it.
    private func makeStreamer(directory: URL, metadata: CaptureMetadata, settings: CaptureSettings,
                              note: inout String?) -> LiveStreamer? {
        let live = LiveSettings.load()
        guard live.enabled, let engine else { return nil }
        guard !live.macName.isEmpty else {
            note = "Stream to Mac is on but no Mac is selected (Settings). Recording on this iPhone only."
            return nil
        }
        let geometry = engine.streamGeometry(halfResolution: settings.halfResolution)
        let hello = LiveHello(captureName: directory.lastPathComponent,
                              deviceModel: metadata.deviceModel,
                              systemVersion: metadata.systemVersion,
                              imageWidth: geometry.image.width,
                              imageHeight: geometry.image.height,
                              depthWidth: geometry.depth?.width,
                              depthHeight: geometry.depth?.height,
                              hasLidar: geometry.depth != nil)
        let id = UUID()
        do {
            return try LiveStreamer(id: id, macName: live.macName, endpoint: browser.endpoint(for: live.macName),
                                    directory: directory, hello: hello) { [weak self] snap in
                // Main-queue FIFO keeps the updates in order (the terminal state is always last).
                DispatchQueue.main.async {
                    MainActor.assumeIsolated { self?.liveUpdate(id, snap) }
                }
            }
        } catch {
            note = "Could not start streaming to \(live.macName): \(error.localizedDescription)"
            return nil
        }
    }

    private func liveUpdate(_ id: UUID, _ snap: LiveStreamSnapshot) {
        if id == currentStreamID {
            liveStatus = snap
        } else if snap.phase.isTerminal {
            // A previous capture's stream finished while a new recording is running.
            message = [message, "Previous capture — \(snap.hudText)"].compactMap { $0 }.joined(separator: "\n")
        }
        if snap.phase.isTerminal {
            streamers[id] = nil
            endBackgroundWorkIfIdle()
        }
    }

    /// Keep the app alive long enough to write capture.json (and finish the stream) when it is backgrounded.
    private func beginBackgroundWork() {
        guard backgroundTask == .invalid else { return }
        backgroundTask = UIApplication.shared.beginBackgroundTask(withName: "Finish capture") { [weak self] in
            MainActor.assumeIsolated { self?.endBackgroundWork() }
        }
    }

    private func endBackgroundWorkIfIdle() {
        if !isRecording && !isFinishing && streamers.isEmpty { endBackgroundWork() }
    }

    private func endBackgroundWork() {
        guard backgroundTask != .invalid else { return }
        UIApplication.shared.endBackgroundTask(backgroundTask)
        backgroundTask = .invalid
    }

    func stopRecording() {
        guard let engine, isRecording else { return }
        isRecording = false
        isFinishing = true
        beginBackgroundWork()
        UIApplication.shared.isIdleTimerDisabled = false
        engine.stopRecording { [weak self] result in
            Task { @MainActor in self?.finished(result) }
        }
    }

    private func finished(_ result: Result<CaptureSummary, Error>) {
        isFinishing = false
        defer { endBackgroundWorkIfIdle() }
        switch result {
        case .success(let s):
            var text = "Saved \(s.directory.lastPathComponent): \(s.frameCount) frames"
            text += s.hasLidar ? ", LiDAR depth" : ""
            text += ", \(s.pointCount) points"
            if s.writeErrors > 0 { text += " (\(s.writeErrors) frames failed to write)" }
            message = text
        case .failure(let error):
            message = error.localizedDescription
        }
    }
}
