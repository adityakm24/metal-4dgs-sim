import SwiftUI
import AppKit
import simd
import M4DGSCore

/// "Live Capture" window: receiver status, the capture streaming in from the iPhone (thumbnail, counters,
/// top-down trajectory) and Reconstruct for finished captures.
struct LiveCaptureView: View {
    @Bindable var model: LiveCaptureModel

    var body: some View {
        HSplitView {
            VStack(alignment: .leading, spacing: 12) {
                ReceiverHeader(model: model)
                Divider()
                CurrentCaptureSection(model: model)
                Spacer(minLength: 0)
            }
            .padding(14)
            .frame(minWidth: 420, idealWidth: 520)

            VStack(alignment: .leading, spacing: 12) {
                CapturesSection(model: model)
                Divider()
                ReconstructSection(model: model)
            }
            .padding(14)
            .frame(minWidth: 380, idealWidth: 480)
        }
        .frame(minWidth: 860, minHeight: 600)
        .onAppear { model.startListening() }
    }
}

private func formatBytes(_ n: Int) -> String { ByteCountFormatter.string(fromByteCount: Int64(n), countStyle: .file) }

struct ReceiverHeader: View {
    @Bindable var model: LiveCaptureModel
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Circle().fill(model.listening ? Color.green : Color.secondary).frame(width: 10, height: 10)
                Text(model.listening ? "Listening" : "Not listening").font(.headline)
                if let p = model.port { Text("port \(String(p))").font(.callout.monospaced()).foregroundStyle(.secondary) }
                Spacer()
                Toggle("Receiver", isOn: Binding(get: { model.listening }, set: { $0 ? model.startListening() : model.stopListening() }))
                    .toggleStyle(.switch).labelsHidden()
            }
            if let n = model.serviceName {
                Text("Bonjour \(LiveLink.serviceType) · \"\(n)\"").font(.caption).foregroundStyle(.secondary)
            }
            Text("Captures → \(model.capturesFolder.path)").font(.caption).foregroundStyle(.secondary)
                .lineLimit(1).truncationMode(.middle).help(model.capturesFolder.path)
            if let e = model.listenerError { Text(e).font(.caption).foregroundStyle(.red) }
        }
    }
}

struct CurrentCaptureSection: View {
    @Bindable var model: LiveCaptureModel

    var stateText: (String, Color) {
        switch model.captureState {
        case .idle: return ("Waiting for an iPhone — start a capture in M4DGS Capture and pick this Mac", .secondary)
        case .receiving: return ("Receiving", .green)
        case .saved: return ("Capture complete", .blue)
        case .cancelled: return ("Cancelled on the phone (partial files deleted)", .orange)
        case .dropped: return ("Connection dropped — capture.partial.json left in the folder", .orange)
        case .failed(let s): return ("Error: \(s)", .red)
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(stateText.0).font(.callout.weight(.medium)).foregroundStyle(stateText.1)
            Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 4) {
                GridRow { Text("Device").foregroundStyle(.secondary); Text(model.device ?? "—") }
                GridRow { Text("From").foregroundStyle(.secondary); Text(model.endpoint ?? "—").font(.callout.monospaced()) }
                GridRow { Text("Capture").foregroundStyle(.secondary); Text(model.captureName ?? "—").textSelection(.enabled) }
                GridRow {
                    Text("Received").foregroundStyle(.secondary)
                    Text("\(model.framesReceived) frames · \(formatBytes(model.bytesReceived))").monospacedDigit()
                }
                if let f = model.lastFrameInfo {
                    GridRow { Text("Last frame").foregroundStyle(.secondary); Text(f).font(.callout.monospaced()) }
                }
            }
            .font(.callout)
            HStack(alignment: .top, spacing: 10) {
                ZStack {
                    RoundedRectangle(cornerRadius: 6).fill(.black.opacity(0.85))
                    if let img = model.thumbnail {
                        Image(nsImage: img).resizable().aspectRatio(contentMode: .fit)
                    } else {
                        Image(systemName: "camera").font(.largeTitle).foregroundStyle(.secondary)
                    }
                }
                .frame(minWidth: 200, maxWidth: .infinity, minHeight: 150, maxHeight: 240)
                TrajectoryPlot(points: model.trajectory)
                    .frame(minWidth: 180, maxWidth: .infinity, minHeight: 150, maxHeight: 240)
            }
            ForEach(model.warnings, id: \.self) { Label($0, systemImage: "exclamationmark.triangle").font(.caption).foregroundStyle(.orange) }
            Text("Events").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 1) {
                        ForEach(Array(model.events.enumerated()), id: \.offset) { i, s in
                            Text(s).font(.caption.monospaced()).textSelection(.enabled).id(i)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(minHeight: 80)
                .background(Color(nsColor: .textBackgroundColor).opacity(0.5), in: RoundedRectangle(cornerRadius: 4))
                .onChange(of: model.events.count) { _, n in if n > 0 { proxy.scrollTo(n - 1, anchor: .bottom) } }
            }
        }
    }
}

/// Top-down (ARKit +y up) plot of camera centers: x to the right, z downwards, equal aspect.
struct TrajectoryPlot: View {
    let points: [(pos: SIMD2<Double>, fwd: SIMD2<Double>)]

    var body: some View {
        Canvas { ctx, size in
            let rect = CGRect(origin: .zero, size: size)
            ctx.fill(Path(roundedRect: rect, cornerRadius: 6), with: .color(Color(nsColor: .textBackgroundColor)))
            guard !points.isEmpty else {
                ctx.draw(Text("camera trajectory (top-down)").font(.caption).foregroundStyle(.secondary),
                         at: CGPoint(x: size.width / 2, y: size.height / 2))
                return
            }
            var lo = points[0].pos, hi = lo
            for p in points { lo = simd_min(lo, p.pos); hi = simd_max(hi, p.pos) }
            let span = max(hi.x - lo.x, hi.y - lo.y, 0.5)
            let pad: CGFloat = 18
            let s = Double(min(size.width, size.height) - 2 * pad) / span
            let c = (lo + hi) * 0.5
            func map(_ p: SIMD2<Double>) -> CGPoint {
                CGPoint(x: size.width / 2 + CGFloat((p.x - c.x) * s), y: size.height / 2 + CGFloat((p.y - c.y) * s))
            }
            var path = Path()
            path.move(to: map(points[0].pos))
            for p in points.dropFirst() { path.addLine(to: map(p.pos)) }
            ctx.stroke(path, with: .color(.cyan), lineWidth: 1.5)
            for p in points { ctx.fill(Path(ellipseIn: CGRect(origin: map(p.pos), size: .zero).insetBy(dx: -2, dy: -2)), with: .color(.cyan)) }
            ctx.fill(Path(ellipseIn: CGRect(origin: map(points[0].pos), size: .zero).insetBy(dx: -4, dy: -4)), with: .color(.green))
            if let last = points.last {
                let a = map(last.pos)
                let f = last.fwd / max(simd_length(last.fwd), 1e-9)
                let b = CGPoint(x: a.x + CGFloat(f.x) * 22, y: a.y + CGFloat(f.y) * 22)
                var arrow = Path(); arrow.move(to: a); arrow.addLine(to: b)
                ctx.stroke(arrow, with: .color(.orange), lineWidth: 2.5)
                ctx.fill(Path(ellipseIn: CGRect(origin: a, size: .zero).insetBy(dx: -4, dy: -4)), with: .color(.orange))
            }
            // scale bar (1, 0.5, 2, 5 … m)
            let nice = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100].first { $0 * s > 40 } ?? 100
            let y = size.height - 8
            var bar = Path(); bar.move(to: CGPoint(x: 10, y: y)); bar.addLine(to: CGPoint(x: 10 + nice * s, y: y))
            ctx.stroke(bar, with: .color(.secondary), lineWidth: 2)
            ctx.draw(Text("\(nice, specifier: "%g") m").font(.caption2).foregroundStyle(.secondary),
                     at: CGPoint(x: 10 + nice * s / 2, y: y - 8))
            ctx.draw(Text("x → · z ↓ (top-down)").font(.caption2).foregroundStyle(.secondary),
                     at: CGPoint(x: size.width - 60, y: 10))
        }
    }
}

struct CapturesSection: View {
    @Bindable var model: LiveCaptureModel
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Completed captures").font(.headline)
                Spacer()
                Button("Add Folder…") { model.addExistingCapturePanel() }.controlSize(.small)
            }
            List(model.completed, selection: $model.selectedCapture) { c in
                HStack {
                    VStack(alignment: .leading) {
                        Text(c.name)
                        Text("\(c.frames) frames").font(.caption).foregroundStyle(.secondary)
                    }
                    Spacer()
                    Button { NSWorkspace.shared.activateFileViewerSelecting([c.directory]) } label: { Image(systemName: "folder") }
                        .buttonStyle(.borderless).help("Show in Finder")
                }
                .tag(c)
            }
            .frame(minHeight: 90, maxHeight: 160)
            .overlay { if model.completed.isEmpty { Text("none yet").foregroundStyle(.secondary) } }
        }
    }
}

struct ReconstructSection: View {
    @Bindable var model: LiveCaptureModel
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Reconstruct").font(.headline)
                Spacer()
                Text("Iterations")
                TextField("", value: $model.iterations, format: .number.grouping(.never))
                    .frame(width: 70).textFieldStyle(.roundedBorder)
                    .disabled(model.reconstructing)
            }
            HStack {
                if model.reconstructing {
                    Button(role: .cancel) { model.cancelReconstruction() } label: { Label("Cancel", systemImage: "stop.fill") }
                } else {
                    Button {
                        if let c = model.selectedCapture { model.reconstruct(c) }
                    } label: { Label("Reconstruct", systemImage: "cube.transparent") }
                    .buttonStyle(.borderedProminent)
                    .disabled(model.selectedCapture == nil)
                }
                if let c = model.reconstructing ? model.reconstructCapture : model.selectedCapture?.name {
                    Text(c).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                }
                Spacer()
                if let ck = model.lastCheckpoint, !model.reconstructing {
                    Button("Open in Viewer") { model.openCheckpoint?(ck) }.controlSize(.small)
                }
            }
            if model.reconstructing || model.progress > 0 {
                ProgressView(value: model.progress) {
                    if let p = model.lastProgress {
                        Text("it \(p.iteration) / \(model.iterations)" + (p.loss.map { String(format: " · loss %.4f", $0) } ?? "")
                             + (p.psnr.map { String(format: " · PSNR %.2f dB", $0) } ?? "")
                             + (p.gaussians.map { " · \($0.formatted()) Gaussians" } ?? ""))
                            .font(.caption.monospaced())
                    } else if model.reconstructing {
                        Text("starting…").font(.caption)
                    }
                }
            }
            if let r = model.reconstructResult { Text(r).font(.caption).foregroundStyle(.secondary).textSelection(.enabled) }
            ScrollViewReader { proxy in
                ScrollView([.vertical, .horizontal]) {
                    LazyVStack(alignment: .leading, spacing: 0) {
                        ForEach(Array(model.log.enumerated()), id: \.offset) { i, s in
                            Text(s).font(.system(size: 10.5, design: .monospaced)).fixedSize().id(i)
                        }
                    }
                    .padding(6)
                    .textSelection(.enabled)
                }
                .frame(maxWidth: .infinity, minHeight: 160, maxHeight: .infinity)
                .background(Color(nsColor: .textBackgroundColor), in: RoundedRectangle(cornerRadius: 4))
                .onChange(of: model.log.count) { _, n in if n > 0 { proxy.scrollTo(n - 1, anchor: .bottomLeading) } }
            }
            Text("Repo: \(model.repoRoot?.path ?? "not found — set it in Settings (⌘,)")")
                .font(.caption2).foregroundStyle(model.repoRoot == nil ? .red : .secondary).lineLimit(1).truncationMode(.middle)
        }
    }
}

/// Settings (⌘,): repo root + captures folder, persisted in UserDefaults; empty = auto-detected default.
struct LiveSettingsView: View {
    @Bindable var model: LiveCaptureModel
    var body: some View {
        Form {
            Section("Python repo (contains scripts/train_4dgs.py)") {
                HStack {
                    TextField("auto: \(model.detectedRepoRoot?.path ?? "not found")", text: $model.repoRootPath)
                    Button("Choose…") { model.choosePath(directoryFor: \.repoRootPath, message: "Choose the metal-4dgs-sim repo root") }
                }
                Text("Using: \(model.repoRoot?.path ?? "—")").font(.caption).foregroundStyle(.secondary)
            }
            Section("Captures folder (live captures are written here)") {
                HStack {
                    TextField("auto: \(StudioSettings.defaultCapturesFolder(repo: model.repoRoot).path)", text: $model.capturesFolderPath)
                    Button("Choose…") { model.choosePath(directoryFor: \.capturesFolderPath, message: "Choose the folder for live captures") }
                }
                Text("Using: \(model.capturesFolder.path)").font(.caption).foregroundStyle(.secondary)
            }
            Section("Reconstruction") {
                TextField("Iterations", value: $model.iterations, format: .number.grouping(.never))
            }
        }
        .formStyle(.grouped)
        .frame(width: 560)
        .padding()
    }
}
