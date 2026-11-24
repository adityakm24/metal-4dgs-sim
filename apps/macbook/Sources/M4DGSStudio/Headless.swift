import Foundation
import Metal
import M4DGSCore
import M4DGSRender

/// `M4DGSStudio --render <scene.safetensors> --time 2.0 --out frame.png [--bench 200] ...`
enum Headless {
    static let usage = """
    usage: M4DGSStudio --render <scene.safetensors> [--time T] [--out frame.png] [--bench N]
                       [--width W --height H] [--overlays] [--rollout file.rollout.json]
                       [--orbit yaw,pitch,dist] [--no-edits] [--edits file.edits.json]
                       [--perturb "actor_1:lateral_offset=3.5m;actor_2:lateral_offset=-1.8m,t=2.5s"] [--save-edits]
    Renders the ego camera view offscreen (rgba16Float) and writes a PNG; --bench reports GPU times.
    """

    static func run(_ args: [String]) -> Int32 {
        var opts: [String: String] = [:]
        var flags: Set<String> = []
        var i = 0
        let valued: Set<String> = ["--render", "--time", "--out", "--bench", "--width", "--height", "--rollout", "--edits", "--orbit", "--perturb"]
        while i < args.count {
            let a = args[i]
            if valued.contains(a), i + 1 < args.count { opts[a] = args[i + 1]; i += 2 } else { flags.insert(a); i += 1 }
        }
        guard let scenePath = opts["--render"] else { print(usage); return 2 }
        do {
            let t0 = Date()
            let scene = try LoadedScene(url: URL(fileURLWithPath: scenePath),
                                        applyEdits: !flags.contains("--no-edits"),
                                        editsOverride: opts["--edits"].map { URL(fileURLWithPath: $0) })
            let loadMs = Date().timeIntervalSince(t0) * 1000
            let g = scene.graph
            print(String(format: "scene: %@  gaussians=%d  sh_degree=%d  nodes=%d  time_range=[%.2f, %.2f]  load=%.0f ms",
                         scene.url.lastPathComponent, scene.cloud.count, g.shDegree, g.nodes.count,
                         g.timeRange.lowerBound, g.timeRange.upperBound, loadMs))
            if let e = scene.editsApplied { print("applied edits: \(e.path)") }

            var fb = FrameBuilder(graph: g)
            fb.overlays = flags.contains("--overlays") ? OverlayOptions() : .none
            if let ps = opts["--perturb"] {
                var edited: [String: [TrackKey]] = [:]
                for item in ps.split(separator: ";") {
                    guard let (name, e) = parsePerturbation(String(item)), let n = g.node(named: name) else {
                        print("ignoring perturbation '\(item)'"); continue
                    }
                    edited[name] = TrackEditor.apply(e, to: n.track, timeRange: g.timeRange)
                    print("perturbation \(name): \(e)")
                }
                fb.tracks = edited
                if flags.contains("--save-edits") {
                    let url = opts["--edits"].map { URL(fileURLWithPath: $0) } ?? LoadedScene.editsURL(for: scene.url)
                    try scene.loaded.jsonData(replacingTracks: edited).write(to: url, options: .atomic)
                    print("saved edits: \(url.path)")
                }
            }
            if let rp = opts["--rollout"] {
                let r = try Rollout(url: URL(fileURLWithPath: rp))
                fb.tracks = r.actorTracks
                fb.egoTrack = r.egoTrack
                fb.loggedEgoTrack = g.ego?.track
                let tq = Double(opts["--time"] ?? "") ?? r.timeRange.lowerBound
                if let si = r.stepIndex(at: tq) {
                    let st = r.steps[si]
                    fb.collidingActors = Set(st.events.filter { $0.hasPrefix("collision") }
                        .compactMap { $0.split(separator: ":").dropFirst().first.map(String.init) })
                    print(String(format: "rollout step t=%.2f speed=%.2f steer=%+.3f accel=%+.2f events=%@",
                                 st.t, st.speed ?? .nan, st.steer ?? .nan, st.accel ?? .nan, st.events.description))
                }
                print("rollout: \(r.steps.count) steps, policy=\(r.policy ?? "?")")
            }
            let renderer = try SplatRenderer()
            try renderer.setScene(scene.cloud)
            print("device: \(renderer.device.name)")

            let k = fb.egoIntrinsics
            let W = Int(opts["--width"] ?? "") ?? k.width
            let H = Int(opts["--height"] ?? "") ?? k.height
            let t = Double(opts["--time"] ?? "") ?? g.timeRange.lowerBound
            var cam: CameraChoice = .ego
            if let o = opts["--orbit"] {
                let v = o.split(separator: ",").compactMap { Double($0) }
                let ego = fb.sceneFrame(at: t).egoPose
                cam = .free(OrbitCamera(target: ego.pos + ego.forward * 8, yaw: (v.first ?? 200) * .pi / 180,
                                        pitch: (v.count > 1 ? v[1] : 35) * .pi / 180, distance: v.count > 2 ? v[2] : 30))
            }
            let frame = fb.build(time: t, camera: cam, targetWidth: W, targetHeight: H)
            let fmtEnv = ProcessInfo.processInfo.environment["M4DGS_TARGET_FORMAT"]
            let target = renderer.makeTarget(width: W, height: H, format: fmtEnv == "bgra8" ? .bgra8Unorm : .rgba16Float)
            let ms = try renderer.renderSync(frame, target: target)
            print(String(format: "rendered %dx%d at t=%.3f s: visible splats=%d/%d, gpu=%.3f ms (cold)",
                         W, H, t, renderer.visibleCount, renderer.gaussianCount, ms))
            if let out = opts["--out"] {
                let px = renderer.readRGBA8(target)
                try PNGWriter.write(rgba: px, width: W, height: H, to: URL(fileURLWithPath: out))
                print("wrote \(out)")
            }
            if let nStr = opts["--bench"], let n = Int(nStr), n > 0 {
                let tm = try renderer.benchmark(frame, target: target, iterations: n)
                func fmt(_ name: String, _ xs: [Double]) -> String {
                    name.padding(toLength: 11, withPad: " ", startingAt: 0) + String(format: "P50 %.3f ms  P90 %.3f ms  min %.3f ms  (n=%d)",
                           percentile(xs, 50), percentile(xs, 90), xs.min() ?? .nan, xs.count)
                }
                print("benchmark \(W)x\(H), \(renderer.gaussianCount) gaussians, \(renderer.visibleCount) visible:")
                print("  " + fmt("frame", tm.totalMs))
                print("  " + fmt("preprocess", tm.preprocessMs))
                print("  " + fmt("sort", tm.sortMs))
                print("  " + fmt("raster", tm.rasterMs))
                let sustained = try renderer.benchmarkSustained(frame, target: target, iterations: n)
                print(String(format: "  sustained  %.3f ms/frame GPU (back-to-back, 3 in flight; = %.0f fps GPU-bound)",
                             sustained, 1000 / sustained))
                print(String(format: "P50_GPU_MS=%.3f  (per-frame gpuEndTime-gpuStartTime, serial submission)", percentile(tm.totalMs, 50)))
            }
            return 0
        } catch {
            fputs("error: \(error)\n", stderr)
            return 1
        }
    }
}
