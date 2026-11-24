import Foundation
import M4DGSCore

/// `M4DGSStudio --receive [--out DIR] [--once] [--port P] [--name NAME]` — headless live-link receiver.
/// `M4DGSStudio --reconstruct <capture dir> [--iterations N] [--output ckpt.safetensors] [--repo DIR]` — runs the
/// same reconstruction job as the Live Capture panel's Reconstruct button.
enum HeadlessLive {
    static let usage = """
    usage: M4DGSStudio --receive [--out DIR] [--once] [--port P] [--name "Bonjour name"]
             Advertise _m4dgs-live._tcp and write incoming captures to DIR/<capture_name>/
             (default: the captures folder setting, e.g. <repo>/data/live). --once: exit after the first
             capture connection ends (status 0 if it was saved, 1 otherwise).
           M4DGSStudio --reconstruct <capture dir> [--iterations N] [--output file.safetensors] [--repo DIR]
             Run <repo>/.venv/bin/python <repo>/scripts/train_4dgs.py on a capture
             (default output: <repo>/checkpoints/<capture name>.safetensors).
    """

    static func parse(_ args: [String], valued: Set<String>) -> ([String: String], Set<String>, [String]) {
        var opts: [String: String] = [:], flags: Set<String> = [], positional: [String] = []
        var i = 0
        while i < args.count {
            let a = args[i]
            if valued.contains(a), i + 1 < args.count { opts[a] = args[i + 1]; i += 2 }
            else if a.hasPrefix("--") { flags.insert(a); i += 1 }
            else { positional.append(a); i += 1 }
        }
        return (opts, flags, positional)
    }

    static func log(_ s: String) {
        let ts = ISO8601DateFormatter.string(from: Date(), timeZone: .current, formatOptions: [.withTime, .withColonSeparatorInTime])
        print("[\(ts)] \(s)")
        fflush(stdout)
    }

    static func bytes(_ n: Int) -> String { ByteCountFormatter.string(fromByteCount: Int64(n), countStyle: .file) }

    // MARK: --receive

    static func receive(_ args: [String]) -> Int32 {
        let (opts, flags, _) = parse(args, valued: ["--out", "--port", "--name"])
        let out = opts["--out"].map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath, isDirectory: true) }
            ?? StudioSettings.capturesFolder
        let once = flags.contains("--once")
        let port = UInt16(opts["--port"] ?? "0") ?? 0
        let receiver = LiveReceiver(capturesFolder: out.standardizedFileURL)
        receiver.onEvent = { e in
            switch e {
            case .listening(let p, let name):
                log("listening on port \(p), Bonjour \(LiveLink.serviceType) \"\(name)\"; captures → \(out.path)")
                print("LISTENING_PORT=\(p)"); fflush(stdout)
            case .serviceRegistered(let name): log("Bonjour service registered as \"\(name)\"")
            case .listenerFailed(let s): log("listener: \(s)")
            case .stopped: log("stopped")
            case .connected(let id, let ep): log("#\(id) connection from \(ep)")
            case .protocolError(let id, let s): log("#\(id) error: \(s)")
            case .session(let id, let se):
                switch se {
                case .started(let h, let dir):
                    log("#\(id) hello: \(h.captureName) from \(h.deviceModel ?? "?") iOS \(h.systemVersion ?? "?"), "
                        + "\(h.imageWidth ?? 0)x\(h.imageHeight ?? 0), lidar=\(h.hasLidar.map(String.init) ?? "?") → \(dir.path)")
                case .frame(let f):
                    log("#\(id) frame \(f.index ?? -1) \(f.imagePath ?? "") (\(f.framesReceived) frames, \(bytes(f.bytesReceived)))")
                case .points(let n, let b): log("#\(id) points: \(n.map(String.init) ?? "?") (\(bytes(b)))")
                case .saved(let dir, let n, let w):
                    for x in w { log("#\(id) warning: \(x)") }
                    log("#\(id) saved \(n) frames → \(dir.path)")
                    print("SAVED_CAPTURE=\(dir.path)"); fflush(stdout)
                case .cancelled(let dir): log("#\(id) cancelled; deleted \(dir.path)")
                case .ignored(let s): log("#\(id) ignored \(s)")
                }
            case .disconnected(let id, let state, let partial):
                log("#\(id) disconnected (\(state))" + (partial.map { "; partial capture left in \($0.path)" } ?? ""))
                if once { exit(state == .saved ? 0 : 1) }
            }
        }
        do {
            try receiver.start(port: port, serviceName: opts["--name"] ?? LiveReceiver.macName)
        } catch {
            fputs("error: could not start listener: \(error)\n", stderr)
            return 1
        }
        // Ctrl-C / SIGTERM: close connections (incomplete captures keep capture.partial.json), then exit.
        signal(SIGINT, SIG_IGN); signal(SIGTERM, SIG_IGN)
        var sources: [DispatchSourceSignal] = []
        for sig in [SIGINT, SIGTERM] {
            let s = DispatchSource.makeSignalSource(signal: sig, queue: .main)
            s.setEventHandler {
                receiver.stop()
                receiver.queue.async { DispatchQueue.main.async { exit(130) } }
            }
            s.resume()
            sources.append(s)
        }
        withExtendedLifetime(sources) { dispatchMain() }
    }

    // MARK: --reconstruct

    static func reconstruct(_ args: [String]) -> Int32 {
        let (opts, _, _) = parse(args, valued: ["--reconstruct", "--iterations", "--output", "--repo"])
        guard let cap = opts["--reconstruct"] else { print(usage); return 2 }
        let repo = opts["--repo"].map { URL(fileURLWithPath: $0) } ?? StudioSettings.repoRoot
        guard let repo else { fputs("error: repo root not found (pass --repo)\n", stderr); return 2 }
        let cfg = ReconstructJob.Config(repoRoot: repo, captureDir: URL(fileURLWithPath: cap).standardizedFileURL,
                                        iterations: Int(opts["--iterations"] ?? "") ?? StudioSettings.iterations,
                                        output: opts["--output"].map { URL(fileURLWithPath: $0) })
        let job = ReconstructJob(cfg)
        let iters = cfg.iterations
        job.onLine = { line, p in
            if let p { print(String(format: "%@   [%3.0f%%]", line, 100 * Double(p.iteration) / Double(max(iters, 1)))) }
            else { print(line) }
            fflush(stdout)
        }
        let done = DispatchSemaphore(value: 0)
        var status: Int32 = 1
        job.onExit = { st, cancelled in status = cancelled ? 130 : st; done.signal() }
        log("running: \(cfg.commandLine)")
        do { try job.start() } catch {
            fputs("error: \(error)\n", stderr)
            return 1
        }
        signal(SIGINT, SIG_IGN)
        let s = DispatchSource.makeSignalSource(signal: SIGINT, queue: .global())
        s.setEventHandler { job.cancel() }
        s.resume()
        done.wait()
        s.cancel()
        let ok = status == 0 && FileManager.default.fileExists(atPath: cfg.output.path)
        log(ok ? "reconstruction finished: \(cfg.output.path)" : "reconstruction failed (exit \(status))")
        if ok { print("CHECKPOINT=\(cfg.output.path)") }
        return ok ? 0 : (status == 0 ? 1 : status)
    }
}
