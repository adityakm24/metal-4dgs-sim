import SwiftUI
import AppKit

@MainActor enum Shared {
    static let model = AppModel()
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        // Running as a bare SwiftPM executable: become a regular foreground app with a Dock icon.
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
        // `M4DGSStudio [scene.safetensors] [--rollout file.rollout.json]`
        let args = Array(CommandLine.arguments.dropFirst())
        var scene: String?, rollout: String?
        var i = 0
        while i < args.count {
            if args[i] == "--rollout", i + 1 < args.count { rollout = args[i + 1]; i += 2; continue }
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
        }
    }
}
