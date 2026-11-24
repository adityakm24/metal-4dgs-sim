import Foundation
import M4DGSCore

/// Runs `<repo>/.venv/bin/python <repo>/scripts/train_4dgs.py --data_dir <capture> --iterations N --output <ckpt>`
/// as a child process, streaming its merged stdout/stderr line by line. Used by the Live Capture panel's
/// Reconstruct button and by `M4DGSStudio --reconstruct`.
final class ReconstructJob: @unchecked Sendable {
    struct Config {
        var repoRoot: URL
        var captureDir: URL
        var iterations: Int
        var output: URL

        init(repoRoot: URL, captureDir: URL, iterations: Int, output: URL? = nil) {
            self.repoRoot = repoRoot
            self.captureDir = captureDir
            self.iterations = iterations
            self.output = output ?? ReconstructJob.defaultOutput(repoRoot: repoRoot, captureName: captureDir.lastPathComponent)
        }
        var python: URL { repoRoot.appendingPathComponent(".venv/bin/python") }
        var script: URL { repoRoot.appendingPathComponent("scripts/train_4dgs.py") }
        var arguments: [String] {
            [script.path, "--data_dir", captureDir.path, "--iterations", String(iterations), "--output", output.path]
        }
        var commandLine: String { ([python.path] + arguments).map { $0.contains(" ") ? "'\($0)'" : $0 }.joined(separator: " ") }
    }

    enum JobError: Error, CustomStringConvertible {
        case missing(String)
        var description: String { switch self { case .missing(let s): return s } }
    }

    static func defaultOutput(repoRoot: URL, captureName: String) -> URL {
        repoRoot.appendingPathComponent("checkpoints", isDirectory: true).appendingPathComponent("\(captureName).safetensors")
    }

    let config: Config
    /// Called on a background queue, in order, for each output line (and parsed progress if it is a train line).
    var onLine: ((String, TrainProgress?) -> Void)?
    /// Called once, after the process exited and all output was delivered.
    var onExit: ((_ status: Int32, _ cancelled: Bool) -> Void)?

    private let process = Process()
    private let queue = DispatchQueue(label: "m4dgs.reconstruct")
    private var pending = Data()
    private var eof = false
    private var exited = false
    private var cancelled = false
    private var finished = false

    init(_ config: Config) { self.config = config }

    func start() throws {
        let fm = FileManager.default
        guard fm.isExecutableFile(atPath: config.python.path) else {
            throw JobError.missing("Python not found at \(config.python.path) (create the repo's .venv first)")
        }
        guard fm.fileExists(atPath: config.script.path) else { throw JobError.missing("train script not found at \(config.script.path)") }
        guard fm.fileExists(atPath: config.captureDir.appendingPathComponent("capture.json").path)
                || fm.fileExists(atPath: config.captureDir.appendingPathComponent("log.json").path) else {
            throw JobError.missing("\(config.captureDir.path) has no capture.json (incomplete capture?)")
        }
        try fm.createDirectory(at: config.output.deletingLastPathComponent(), withIntermediateDirectories: true)

        process.executableURL = config.python
        process.arguments = config.arguments
        process.currentDirectoryURL = config.repoRoot
        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"
        process.environment = env
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        process.standardInput = FileHandle.nullDevice
        pipe.fileHandleForReading.readabilityHandler = { [weak self] h in
            let d = h.availableData
            self?.queue.async { self?.consume(d) }
            if d.isEmpty { h.readabilityHandler = nil }
        }
        process.terminationHandler = { [weak self] _ in self?.queue.async { self?.exited = true; self?.finishIfDone() } }
        try process.run()
    }

    var isRunning: Bool { process.isRunning }

    /// SIGTERM, then SIGKILL after 5 s if the trainer is still alive.
    func cancel() {
        queue.async { [self] in
            guard !finished, process.isRunning else { return }
            cancelled = true
            process.terminate()
            let pid = process.processIdentifier
            queue.asyncAfter(deadline: .now() + 5) { [self] in
                if process.isRunning { kill(pid, SIGKILL) }
            }
        }
    }

    private func consume(_ d: Data) {
        if d.isEmpty { eof = true; flush(all: true); finishIfDone(); return }
        pending.append(d)
        flush(all: false)
    }

    private func flush(all: Bool) {
        // lines end with \n; tqdm-style progress uses \r
        while let i = pending.firstIndex(where: { $0 == 0x0a || $0 == 0x0d }) {
            let line = String(decoding: pending[pending.startIndex..<i], as: UTF8.self)
            pending.removeSubrange(pending.startIndex...i)
            if !line.isEmpty { onLine?(line, TrainProgress.parse(line)) }
        }
        if all, !pending.isEmpty {
            let line = String(decoding: pending, as: UTF8.self)
            pending = Data()
            onLine?(line, TrainProgress.parse(line))
        }
    }

    private func finishIfDone() {
        guard eof, exited, !finished else { return }
        finished = true
        onExit?(process.terminationStatus, cancelled)
    }
}
