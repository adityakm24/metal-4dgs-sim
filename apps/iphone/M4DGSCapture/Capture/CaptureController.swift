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

    @ObservationIgnored private var engine: CaptureEngine?
    @ObservationIgnored private var isRunning = false

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
        do {
            let writer = try CaptureWriter(directory: directory, metadata: metadata, settings: settings)
            if !isRunning { resume() }
            engine.startRecording(writer: writer, settings: settings)
            isRecording = true
            message = nil
            UIApplication.shared.isIdleTimerDisabled = true
        } catch {
            message = "Could not start capture: \(error.localizedDescription)"
        }
    }

    func stopRecording() {
        guard let engine, isRecording else { return }
        isRecording = false
        isFinishing = true
        UIApplication.shared.isIdleTimerDisabled = false
        engine.stopRecording { [weak self] result in
            Task { @MainActor in self?.finished(result) }
        }
    }

    private func finished(_ result: Result<CaptureSummary, Error>) {
        isFinishing = false
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
