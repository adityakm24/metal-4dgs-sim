import SwiftUI
import AppKit

@MainActor enum Shared {
    static let model = AppModel()
    static let live: LiveCaptureModel = {
        let l = LiveCaptureModel()
        l.openCheckpoint = { url in
            Shared.model.openScene(url)
            // bring a viewer window forward
            NSApp.windows.first { $0.identifier?.rawValue != LiveCaptureWindow.id && $0.canBecomeMain }?.makeKeyAndOrderFront(nil)
        }
        return l
    }()
}

enum LiveCaptureWindow { static let id = "live-capture" }

/// Live menu (⌘L opens the Live Capture window).
struct LiveCommands: Commands {
    @Environment(\.openWindow) private var openWindow
    var body: some Commands {
        CommandMenu("Live") {
            Button("Live Capture…") { openWindow(id: LiveCaptureWindow.id) }.keyboardShortcut("l")
        }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        // Running as a bare SwiftPM executable: become a regular foreground app with a Dock icon.
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
        // `M4DGSStudio [scene.safetensors] [--rollout file.rollout.json] [--live]`
        let args = Array(CommandLine.arguments.dropFirst())
        var scene: String?, rollout: String?
        var i = 0
        while i < args.count {
            if args[i] == "--rollout", i + 1 < args.count { rollout = args[i + 1]; i += 2; continue }
            // Cocoa `-key value` (NSArgumentDomain, e.g. `-capturesFolder /tmp/x`) overrides a setting for this run
            if args[i].hasPrefix("-"), !args[i].hasPrefix("--"), i + 1 < args.count { i += 2; continue }
            if !args[i].hasPrefix("-") && scene == nil { scene = args[i] }
            i += 1
        }
        let m = Shared.model
        if let s = scene {
            m.openScene(URL(fileURLWithPath: s)) {
                if let r = rollout { m.openRollout(URL(fileURLWithPath: r)) }
            }
        } else if let r = rollout {
            m.openRollout(URL(fileURLWithPath: r))
        }
    }
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}

struct StudioApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    @State private var model = Shared.model
    @State private var live = Shared.live

    var body: some Scene {
        WindowGroup("M4DGS Studio") {
            ContentView(model: model)
                .frame(minWidth: 1100, minHeight: 640)
        }
        .defaultSize(width: 1500, height: 900)
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("Open Scene…") { model.openScenePanel() }.keyboardShortcut("o")
                Button("Open Rollout…") { model.openRolloutPanel() }.keyboardShortcut("o", modifiers: [.command, .shift])
            }
            CommandGroup(replacing: .saveItem) {
                Button("Save Edits") { model.saveEdits() }.keyboardShortcut("s").disabled(model.scene == nil)
                Button("Reset All Edits") { model.resetEdits() }.disabled(model.edits.isEmpty)
                Button("Revert to Checkpoint (ignore .edits.json)") { model.revertToCheckpoint() }.disabled(model.scene == nil)
            }
            CommandMenu("Playback") {
                Button(model.playing ? "Pause" : "Play") { model.playing.toggle() }.keyboardShortcut("p", modifiers: [.command])
                Button("Step Back") { model.step(-1) }.keyboardShortcut(.leftArrow, modifiers: [.command])
                Button("Step Forward") { model.step(1) }.keyboardShortcut(.rightArrow, modifiers: [.command])
                Divider()
                Button("Ego Camera") { model.cameraMode = .ego }.keyboardShortcut("1", modifiers: [.command])
                Button("Free Camera") { model.enterFreeCamera() }.keyboardShortcut("2", modifiers: [.command])
            }
            LiveCommands()
        }

        Window("Live Capture", id: LiveCaptureWindow.id) {
            LiveCaptureView(model: live)
        }
        .defaultSize(width: 1100, height: 760)

        Settings {
            LiveSettingsView(model: live)
        }
    }
}
