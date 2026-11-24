import SwiftUI
import M4DGSCore
import M4DGSRender

struct ContentView: View {
    @Bindable var model: AppModel
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        NavigationSplitView {
            SidebarView(model: model)
                .navigationSplitViewColumnWidth(min: 270, ideal: 310, max: 420)
        } detail: {
            VStack(spacing: 0) {
                ZStack(alignment: .topLeading) {
                    MetalViewport(model: model)
                    if model.scene != nil {
                        HUDView(model: model).padding(10).allowsHitTesting(false)
                    } else {
                        EmptyStateView(model: model).frame(maxWidth: .infinity, maxHeight: .infinity)
                    }
                }
                .frame(minWidth: 480, minHeight: 300)
                Divider()
                TimelineBar(model: model)
            }
        }
        .inspector(isPresented: $model.showInspector) {
            InspectorView(model: model).inspectorColumnWidth(min: 260, ideal: 310, max: 440)
        }
        .navigationTitle(model.title)
        .onAppear {
            if CommandLine.arguments.contains("--live") { openWindow(id: LiveCaptureWindow.id) }
        }
        .toolbar {
            ToolbarItemGroup(placement: .navigation) {
                Button { model.openScenePanel() } label: { Label("Open Scene", systemImage: "cube.transparent") }
                    .help("Open a scene checkpoint (⌘O)")
                Button { model.openRolloutPanel() } label: { Label("Open Rollout", systemImage: "chart.xyaxis.line") }
                    .help("Open a rollout log (⇧⌘O)")
                Button { openWindow(id: LiveCaptureWindow.id) } label: { Label("Live Capture", systemImage: "iphone.radiowaves.left.and.right") }
                    .help("Receive a capture streamed from the iPhone app (⌘L)")
            }
            ToolbarItem(placement: .principal) {
                Picker("Camera", selection: Binding(get: { model.cameraMode }, set: { m in
                    if m == .free { model.enterFreeCamera() } else { model.cameraMode = .ego }
                })) {
                    ForEach(CameraMode.allCases) { Text("\($0.rawValue) camera").tag($0) }
                }
                .pickerStyle(.segmented)
                .frame(width: 220)
            }
            ToolbarItemGroup(placement: .primaryAction) {
                Button { model.saveEdits() } label: { Label("Save Edits", systemImage: "square.and.arrow.down") }
                    .disabled(model.scene == nil)
                    .help("Write <scene>.edits.json (⌘S)")
                Button { model.showInspector.toggle() } label: { Label("Inspector", systemImage: "sidebar.right") }
            }
        }
        .alert("Error", isPresented: Binding(get: { model.errorMessage != nil }, set: { if !$0 { model.errorMessage = nil } })) {
            Button("OK") { model.errorMessage = nil }
        } message: { Text(model.errorMessage ?? "") }
    }
}

struct EmptyStateView: View {
    let model: AppModel
    var body: some View {
        VStack(spacing: 14) {
            Image(systemName: "cube.transparent").font(.system(size: 44)).foregroundStyle(.secondary)
            Text("M4DGS Studio").font(.title2.weight(.semibold))
            Text("Open a 4DGS scene checkpoint (.safetensors) or a rollout (.rollout.json).")
                .foregroundStyle(.secondary)
            HStack {
                Button("Open Scene…") { model.openScenePanel() }.keyboardShortcut("o")
                Button("Open Rollout…") { model.openRolloutPanel() }
            }
            if model.isLoading { ProgressView().controlSize(.small) }
        }
        .padding(30)
    }
}

// MARK: - HUD

struct HUDView: View {
    let model: AppModel
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            VStack(alignment: .leading, spacing: 2) {
                Text("\(model.cameraMode == .ego ? "Ego camera" : (model.followEgo ? "Free camera · following ego" : "Free camera"))  ·  t = \(model.time, specifier: "%.2f") s")
                Text("GPU \(model.gpuMs, specifier: "%.2f") ms  ·  \(model.visibleSplats.formatted()) / \((model.scene?.cloud.count ?? 0).formatted()) splats  ·  \(model.fps, specifier: "%.0f") fps")
                if model.rolloutActive { Text("Rollout playback").foregroundStyle(.cyan) }
            }
            .font(.system(.caption, design: .monospaced))
            .padding(.horizontal, 8).padding(.vertical, 5)
            .background(.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 6))
            .foregroundStyle(.white)
            if model.rolloutActive, let st = model.currentStep, st.hasCollision {
                Label(st.events.filter { $0.hasPrefix("collision") }.joined(separator: ", ").uppercased(),
                      systemImage: "exclamationmark.triangle.fill")
                    .font(.system(.callout, design: .monospaced).weight(.bold))
                    .padding(.horizontal, 10).padding(.vertical, 6)
                    .background(.red.opacity(0.85), in: RoundedRectangle(cornerRadius: 6))
                    .foregroundStyle(.white)
            }
        }
    }
}

// MARK: - Sidebar: nodes + trajectory editor

struct SidebarView: View {
    @Bindable var model: AppModel

    var body: some View {
        List(selection: $model.selectedNode) {
            if let s = model.scene, let g = model.graph {
                Section("Scene") {
                    LabeledContent("File", value: s.url.lastPathComponent)
                    LabeledContent("Gaussians", value: s.cloud.count.formatted())
                    LabeledContent("SH degree", value: "\(g.shDegree)")
                    LabeledContent("Time", value: String(format: "%.2f – %.2f s", g.timeRange.lowerBound, g.timeRange.upperBound))
                    if let e = s.editsApplied {
                        LabeledContent("Edits", value: e.lastPathComponent).foregroundStyle(.orange)
                    }
                }
                Section("Nodes") {
                    ForEach(g.nodes, id: \.name) { n in
                        NodeRow(node: n, count: s.cloud.nodeRanges[n.name]?.count ?? 0, edited: model.isEdited(n.name))
                            .tag(n.name)
                    }
                }
                if let sel = model.selectedNode, let node = g.node(named: sel), node.isActor {
                    Section("Trajectory editor — \(sel)") {
                        TrackEditorForm(model: model, actor: sel)
                    }
                }
                Section("Edits") {
                    HStack {
                        Button("Save") { model.saveEdits() }.keyboardShortcut("s")
                        Button("Reset all") { model.resetEdits() }.disabled(model.edits.isEmpty)
                        Button("Revert") { model.revertToCheckpoint() }
                            .help("Reload the checkpoint without applying the .edits.json sidecar")
                    }
                    if let u = model.editsURL {
                        Text(u.lastPathComponent + (model.dirty ? " (unsaved changes)" : ""))
                            .font(.caption).foregroundStyle(model.dirty ? .orange : .secondary)
                    }
                    if let m = model.statusMessage { Text(m).font(.caption).foregroundStyle(.secondary) }
                }
            } else {
                Text(model.isLoading ? "Loading…" : "No scene loaded").foregroundStyle(.secondary)
            }
        }
        .listStyle(.sidebar)
    }
}

struct NodeRow: View {
    let node: SceneNode
    let count: Int
    let edited: Bool
    var body: some View {
        HStack {
            Image(systemName: node.isActor ? "car.fill" : "square.3.layers.3d")
                .foregroundStyle(node.isActor ? .cyan : .secondary)
            VStack(alignment: .leading, spacing: 1) {
                Text(node.name)
                Text([node.category ?? node.kind, "\(count.formatted()) G", node.track.isEmpty ? nil : "\(node.track.count) keys"]
                    .compactMap { $0 }.joined(separator: " · "))
                    .font(.caption2).foregroundStyle(.secondary)
            }
            Spacer()
            if edited {
                Text("edited").font(.caption2.weight(.semibold))
                    .padding(.horizontal, 5).padding(.vertical, 1)
                    .background(.orange.opacity(0.25), in: Capsule())
            }
        }
    }
}

struct TrackEditorForm: View {
    let model: AppModel
    let actor: String

    var body: some View {
        let e = model.binding(for: actor)
        let tr = model.timeRange
        VStack(alignment: .leading, spacing: 8) {
            ValueSlider(title: "Lateral offset", value: e.lateral, range: -6...6, unit: "m", help: "along the actor's left axis")
            ValueSlider(title: "Longitudinal", value: e.longitudinal, range: -40...40, unit: "m", help: "along the actor's forward axis")
            ValueSlider(title: "Time shift", value: e.timeShift, range: -5...5, unit: "s", help: "positive = arrives later")
            ValueSlider(title: "Speed scale", value: e.speedScale, range: 0...3, unit: "×", help: "along the same path")
            Toggle("Cut-in", isOn: e.cutInEnabled)
            if e.wrappedValue.cutInEnabled {
                ValueSlider(title: "Cut-in offset", value: e.cutInOffset, range: -6...6, unit: "m", help: "extra lateral offset reached")
                ValueSlider(title: "Start", value: e.cutInStart, range: tr.lowerBound...max(tr.upperBound, tr.lowerBound + 0.1), unit: "s", help: nil)
                ValueSlider(title: "Duration", value: e.cutInDuration, range: 0.2...6, unit: "s", help: nil)
                Toggle("Align heading to path", isOn: e.alignHeading)
                Button("Start cut-in at current time") { e.wrappedValue.cutInStart = model.time }
                    .controlSize(.small)
            }
            HStack {
                Button("Reset actor") { model.resetEdits(actor) }.disabled(model.edits[actor] == nil)
                Spacer()
                Text(model.isEdited(actor) ? "\(model.editedTracks[actor]?.count ?? 0) keys" : "unchanged")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 4)
    }
}

struct ValueSlider: View {
    let title: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    let unit: String
    let help: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(title).font(.callout)
                Spacer()
                TextField("", value: $value, format: .number.precision(.fractionLength(2)))
                    .multilineTextAlignment(.trailing)
                    .frame(width: 62)
                    .textFieldStyle(.roundedBorder)
                    .controlSize(.small)
                Text(unit).font(.caption).foregroundStyle(.secondary).frame(width: 14, alignment: .leading)
            }
            Slider(value: $value, in: range).controlSize(.small)
        }
        .help(help ?? title)
    }
}

// MARK: - Timeline

struct TimelineBar: View {
    @Bindable var model: AppModel

    var body: some View {
        let r = model.timeRange
        HStack(spacing: 10) {
            Button { model.step(-1) } label: { Image(systemName: "backward.frame.fill") }.buttonStyle(.borderless)
            Button { model.playing.toggle() } label: {
                Image(systemName: model.playing ? "pause.fill" : "play.fill").frame(width: 18)
            }
            .buttonStyle(.borderless)
            Button { model.step(1) } label: { Image(systemName: "forward.frame.fill") }.buttonStyle(.borderless)
            Text(String(format: "%6.2f s", model.time)).font(.system(.body, design: .monospaced)).frame(width: 72)
            ZStack(alignment: .bottom) {
                Slider(value: Binding(get: { min(max(model.time, r.lowerBound), r.upperBound) },
                                      set: { model.time = $0 }),
                       in: r.lowerBound...max(r.upperBound, r.lowerBound + 1e-3))
                EventMarks(model: model, range: r).frame(height: 6).offset(y: 8).allowsHitTesting(false)
            }
            Text(String(format: "%.2f s", r.upperBound)).font(.caption).foregroundStyle(.secondary)
            Picker("", selection: $model.playbackSpeed) {
                ForEach([0.1, 0.25, 0.5, 1, 2, 4], id: \.self) { Text("\($0, specifier: "%g")×").tag($0) }
            }
            .frame(width: 72)
            Toggle("Loop", isOn: $model.loop).toggleStyle(.checkbox)
        }
        .padding(.horizontal, 12).padding(.vertical, 8)
        .disabled(model.scene == nil)
    }
}

/// Ticks under the timeline for rollout events (red = collision).
struct EventMarks: View {
    let model: AppModel
    let range: ClosedRange<Double>
    var body: some View {
        Canvas { ctx, size in
            guard model.rolloutActive, let r = model.rollout else { return }
            let span = max(range.upperBound - range.lowerBound, 1e-6)
            let inset: CGFloat = 10
            for s in r.steps where !s.events.isEmpty {
                let x = inset + CGFloat((s.t - range.lowerBound) / span) * (size.width - 2 * inset)
                ctx.fill(Path(CGRect(x: x - 1.5, y: 0, width: 3, height: size.height)),
                         with: .color(s.hasCollision ? .red : .yellow))
            }
        }
    }
}

// MARK: - Inspector: view options + rollout panel

struct InspectorView: View {
    @Bindable var model: AppModel

    var body: some View {
        Form {
            Section("View") {
                Picker("Camera", selection: Binding(get: { model.cameraMode }, set: { m in
                    if m == .free { model.enterFreeCamera() } else { model.cameraMode = .ego }
                })) { ForEach(CameraMode.allCases) { Text($0.rawValue).tag($0) } }
                Toggle("Follow ego (free camera)", isOn: $model.followEgo)
                Toggle("Actor boxes", isOn: $model.overlays.boxes)
                Toggle("Actor tracks", isOn: $model.overlays.actorTracks)
                Toggle("Pre-edit tracks", isOn: $model.overlays.originalTracks)
                Toggle("Ego track", isOn: $model.overlays.egoTrack)
                Toggle("Ego frustum", isOn: $model.overlays.egoFrustum)
                Toggle("Scene background colour", isOn: $model.useSceneBackground)
                if !model.useSceneBackground { ColorPicker("Background", selection: $model.customBackground, supportsOpacity: false) }
                Text("Drag: orbit · ⌥/⇧-drag or right-drag: pan · scroll/pinch: zoom · WASD/QE: move · click: select actor · space: play · ←/→: step · 1/2: ego/free · F: follow")
                    .font(.caption2).foregroundStyle(.secondary)
            }
            if let r = model.rollout {
                RolloutPanel(model: model, rollout: r)
            }
        }
        .formStyle(.grouped)
    }
}

struct RolloutPanel: View {
    @Bindable var model: AppModel
    let rollout: Rollout

    var body: some View {
        Section("Rollout") {
            Toggle("Play rollout (ego + actors from log)", isOn: $model.playRollout)
            if let u = model.rolloutURL { LabeledContent("File", value: u.lastPathComponent) }
            LabeledContent("Policy", value: rollout.policy ?? "—")
            LabeledContent("Steps", value: "\(rollout.steps.count) @ \(rollout.dt.map { String(format: "%.3g s", $0) } ?? "?")")
            ForEach(rollout.perturbations, id: \.self) { Text($0).font(.caption.monospaced()) }
        }
        if let st = model.currentStep {
            Section("Step t = \(String(format: "%.2f", st.t)) s") {
                LabeledContent("Speed", value: st.speed.map { String(format: "%.2f m/s (%.1f km/h)", $0, $0 * 3.6) } ?? "—")
                LabeledContent("Steer", value: st.steer.map { String(format: "%+.3f", $0) } ?? "—")
                LabeledContent("Accel", value: st.accel.map { String(format: "%+.2f m/s²", $0) } ?? "—")
                if !st.events.isEmpty {
                    ForEach(st.events, id: \.self) { e in
                        Label(e, systemImage: e.hasPrefix("collision") ? "exclamationmark.triangle.fill" : "flag.fill")
                            .foregroundStyle(e.hasPrefix("collision") ? .red : .orange)
                    }
                }
            }
            Section("Latency (ms)") {
                let maxL = max(st.totalLatencyMs, 1)
                ForEach(st.latencyMs, id: \.0) { k, v in
                    HStack {
                        Text(k).frame(width: 58, alignment: .leading)
                        GeometryReader { g in
                            RoundedRectangle(cornerRadius: 2).fill(.cyan.opacity(0.7))
                                .frame(width: max(2, g.size.width * v / maxL))
                        }.frame(height: 8)
                        Text(String(format: "%6.2f", v)).font(.caption.monospaced())
                    }
                }
                LabeledContent("Total", value: String(format: "%.2f ms (%.0f Hz)", st.totalLatencyMs, 1000 / max(st.totalLatencyMs, 1e-3)))
            }
        }
        Section("Events") {
            let evs = rollout.steps.filter { !$0.events.isEmpty }
            if evs.isEmpty { Text("none").foregroundStyle(.secondary) }
            ForEach(Array(evs.prefix(50).enumerated()), id: \.offset) { _, s in
                Button {
                    model.time = s.t; model.playing = false
                } label: {
                    HStack {
                        Text(String(format: "%.2f s", s.t)).font(.caption.monospaced())
                        Text(s.events.joined(separator: ", ")).foregroundStyle(s.hasCollision ? .red : .orange)
                    }
                }
                .buttonStyle(.plain)
            }
        }
        Section("Metrics") {
            ForEach(rollout.metrics, id: \.0) { k, v in
                LabeledContent(k, value: v.description)
                    .foregroundStyle(k == "collision" && v.description == "true" ? .red : .primary)
            }
        }
    }
}
