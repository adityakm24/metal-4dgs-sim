import SwiftUI

struct CaptureScreen: View {
    @State private var controller = CaptureController()
    @State private var sheet: Sheet?
    @AppStorage(SettingsKey.showFeaturePoints) private var showFeaturePoints = SettingsDefault.showFeaturePoints
    @Environment(\.scenePhase) private var scenePhase

    enum Sheet: String, Identifiable {
        case captures, settings
        var id: String { rawValue }
    }

    var body: some View {
        ZStack {
            if controller.isSupported, let session = controller.session {
                ARPreview(session: session, showFeaturePoints: showFeaturePoints)
                    .ignoresSafeArea()
                VStack(spacing: 8) {
                    HUDView(status: controller.status, isRecording: controller.isRecording, hasLiDAR: controller.hasLiDAR)
                    if let message = controller.message {
                        MessageBanner(text: message) { controller.message = nil }
                    }
                    Spacer()
                    controls
                }
                .padding()
            } else {
                UnsupportedView()
                VStack {
                    Spacer()
                    HStack(spacing: 16) {
                        Button { sheet = .captures } label: { Label("Captures", systemImage: "folder") }
                        Button { sheet = .settings } label: { Label("Settings", systemImage: "gearshape") }
                    }
                    .buttonStyle(.bordered)
                    .padding(.bottom, 32)
                }
            }
        }
        .sheet(item: $sheet, onDismiss: { controller.resume() }) { which in
            NavigationStack {
                switch which {
                case .captures: CapturesListView()
                case .settings: SettingsView(hasLiDAR: controller.hasLiDAR, arSupported: controller.isSupported)
                }
            }
        }
        .onChange(of: sheet) { _, newValue in
            if newValue != nil { controller.pause() }
        }
        .onAppear { controller.resume() }
        .onDisappear { controller.pause() }
        .onChange(of: scenePhase) { _, phase in
            switch phase {
            case .background:
                controller.stopRecording()   // finalize capture.json instead of losing the recording
                controller.pause()
            case .active:
                if sheet == nil { controller.resume() }
            default:
                break
            }
        }
    }

    private var controls: some View {
        HStack {
            Button { sheet = .captures } label: {
                Image(systemName: "folder").font(.title2).frame(width: 56, height: 56)
            }
            .disabled(controller.isRecording || controller.isFinishing)
            .accessibilityLabel("Captures")

            Spacer()

            Button { controller.toggleRecording() } label: {
                ZStack {
                    Circle().strokeBorder(.white, lineWidth: 4).frame(width: 78, height: 78)
                    if controller.isFinishing {
                        ProgressView().tint(.white)
                    } else if controller.isRecording {
                        RoundedRectangle(cornerRadius: 6).fill(.red).frame(width: 32, height: 32)
                    } else {
                        Circle().fill(.red).frame(width: 64, height: 64)
                    }
                }
            }
            .disabled(controller.isFinishing)
            .accessibilityLabel(controller.isRecording ? "Stop recording" : "Start recording")

            Spacer()

            Button { sheet = .settings } label: {
                Image(systemName: "gearshape").font(.title2).frame(width: 56, height: 56)
            }
            .disabled(controller.isRecording || controller.isFinishing)
            .accessibilityLabel("Settings")
        }
        .foregroundStyle(.white)
        .padding(.horizontal, 8)
    }
}

struct HUDView: View {
    let status: EngineStatus
    let isRecording: Bool
    let hasLiDAR: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Circle().fill(trackingColor).frame(width: 10, height: 10)
                Text(trackingText).font(.subheadline.weight(.semibold))
                Spacer()
                Text(hasLiDAR ? (status.depthActive ? "LiDAR: yes" : "LiDAR: starting") : "LiDAR: no")
                    .font(.caption.weight(.medium))
                    .foregroundStyle(hasLiDAR ? .green : .secondary)
            }
            HStack(spacing: 14) {
                if isRecording {
                    Label(Self.formatElapsed(status.elapsed), systemImage: "record.circle")
                        .foregroundStyle(.red)
                }
                Text("Keyframes: \(status.keptFrames)")
                if status.droppedFrames > 0 { Text("Dropped: \(status.droppedFrames)").foregroundStyle(.orange) }
                Spacer()
                Text(String(format: "ω %.2f rad/s", status.angularSpeed))
                    .foregroundStyle(status.angularSpeed > status.maxAngularSpeed ? .orange : .secondary)
            }
            .font(.caption.monospacedDigit())
            if isRecording {
                Text("Last frame: \(decisionText)  ·  points: \(status.featurePoints)")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            if let warning {
                Label(warning, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.yellow)
            }
        }
        .padding(12)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
        .environment(\.colorScheme, .dark)
    }

    private var trackingText: String {
        switch status.tracking {
        case .normal: return "Tracking: normal"
        case .notAvailable: return "Tracking: not available"
        case .limited(let reason): return "Tracking limited: \(Self.humanize(reason))"
        }
    }

    private var trackingColor: Color {
        switch status.tracking {
        case .normal: return .green
        case .limited: return .yellow
        case .notAvailable: return .red
        }
    }

    private var warning: String? {
        switch status.tracking {
        case .normal:
            return status.angularSpeed > status.maxAngularSpeed ? "Turning too fast — frames rejected (motion blur)" : nil
        case .limited(let reason):
            switch reason {
            case "excessiveMotion": return "Slow down — move the phone smoothly."
            case "insufficientFeatures": return "Point at a textured, well-lit area."
            case "initializing": return "Move the phone sideways a little to initialize tracking."
            case "relocalizing": return "Relocalizing — return to an area you already scanned."
            default: return "Tracking is limited; frames are skipped."
            }
        case .notAvailable:
            return "Tracking unavailable; frames are skipped."
        }
    }

    private var decisionText: String {
        switch status.lastDecision {
        case .keep: return "kept"
        case .trackingNotNormal: return "skipped (tracking)"
        case .motionBlur: return "rejected (motion blur)"
        case .tooSoon: return "waiting (interval)"
        case .tooClose: return "waiting (move more)"
        }
    }

    static func humanize(_ reason: String) -> String {
        switch reason {
        case "excessiveMotion": return "excessive motion"
        case "insufficientFeatures": return "insufficient features"
        default: return reason
        }
    }

    static func formatElapsed(_ t: Double) -> String {
        let total = max(0, t)
        let m = Int(total) / 60
        let s = total - Double(m * 60)
        return String(format: "%02d:%04.1f", m, s)
    }
}

struct MessageBanner: View {
    let text: String
    let onClose: () -> Void

    var body: some View {
        HStack(alignment: .top) {
            Text(text).font(.footnote)
            Spacer()
            Button(action: onClose) { Image(systemName: "xmark.circle.fill") }
                .accessibilityLabel("Dismiss")
        }
        .padding(10)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
        .environment(\.colorScheme, .dark)
    }
}

struct UnsupportedView: View {
    var body: some View {
        ContentUnavailableView {
            Label("ARKit not available", systemImage: "arkit")
        } description: {
            Text("This device (or the Simulator) doesn't support ARKit world tracking, so it can't record captures. Run M4DGS Capture on an iPhone. You can still browse and share existing captures.")
        }
    }
}
