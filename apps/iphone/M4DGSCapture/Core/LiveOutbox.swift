import Foundation

// Pure send-queue policy and HUD state for the live link (no Network import; checked on macOS by Tests/CoreChecks).

/// The ordered message stream of one capture — hello, frame 0 … frame n-1, [points], end (or cancel) — with a
/// bounded in-memory cache of frame bytes.
///
/// Frames are never dropped: when the link falls behind and the cache is full, a frame is only recorded by its
/// `FrameRecord` and its bytes are re-read from the capture directory (the complete local copy) when its turn comes.
/// `rewind()` restarts the stream from `hello` for a new connection (the Mac starts a fresh directory per connection).
struct LiveOutbox: Sendable {
    enum Step: Equatable, Sendable {
        case hello
        /// nil blobs → read the files from disk.
        case frame(FrameRecord, LiveFrameBlobs?)
        case points(LivePoints)
        case end(Data)
        case cancel
    }

    let memoryLimit: Int
    private(set) var frames: [FrameRecord] = []
    private var cache: [Int: LiveFrameBlobs] = [:]
    private(set) var cacheBytes = 0
    /// Frames whose bytes did not fit in memory when they were added (will be streamed from disk).
    private(set) var spilledFrames = 0
    private(set) var points: LivePoints?
    private(set) var manifest: Data?
    private(set) var isCancelled = false
    /// 0 = hello, 1 … frames.count = frames, then points (if any), then end.
    private(set) var position = 0
    private var cancelPending = false
    private(set) var cancelTaken = false

    init(memoryLimit: Int) {
        self.memoryLimit = memoryLimit
    }

    /// Recording stopped (end queued) or discarded (cancel queued): no more frames are accepted.
    var isClosed: Bool { manifest != nil || isCancelled }

    /// Frames already handed out on the current pass.
    var framesTaken: Int { max(0, min(position - 1, frames.count)) }
    var framesPending: Int { isCancelled ? 0 : frames.count - framesTaken }
    var helloTaken: Bool { position > 0 }

    private var tailCount: Int { manifest == nil ? 0 : (points == nil ? 1 : 2) }
    private var stepCount: Int { 1 + frames.count + tailCount }

    /// Everything that will ever be sent on this pass has been handed out.
    var isDrained: Bool {
        if isCancelled { return !cancelPending }
        return manifest != nil && position >= stepCount
    }

    var endTaken: Bool { !isCancelled && manifest != nil && position >= stepCount }

    /// Returns true when the bytes were kept in memory, false when the frame will be re-read from disk.
    @discardableResult
    mutating func addFrame(_ record: FrameRecord, blobs: LiveFrameBlobs) -> Bool {
        guard !isClosed else { return false }
        let i = frames.count
        frames.append(record)
        let n = blobs.byteCount
        if cacheBytes + n <= memoryLimit {
            cache[i] = blobs
            cacheBytes += n
            return true
        }
        spilledFrames += 1
        return false
    }

    mutating func finish(points: LivePoints?, manifest: Data) {
        guard !isClosed else { return }
        self.points = points
        self.manifest = manifest
    }

    /// Recording discarded: drop everything; a `cancel` is only owed if `hello` already went out on this pass.
    mutating func cancel() {
        guard !isCancelled else { return }
        isCancelled = true
        cancelPending = helloTaken
        cache.removeAll()
        cacheBytes = 0
    }

    mutating func takeNext() -> Step? {
        if isCancelled {
            guard cancelPending else { return nil }
            cancelPending = false
            cancelTaken = true
            return .cancel
        }
        guard position < stepCount else { return nil }
        defer { position += 1 }
        if position == 0 { return .hello }
        let fi = position - 1
        if fi < frames.count {
            let blobs = cache.removeValue(forKey: fi)
            if let blobs { cacheBytes -= blobs.byteCount }
            return .frame(frames[fi], blobs)
        }
        if fi == frames.count, let points { return .points(points) }
        return .end(manifest!)
    }

    /// New connection: start over from `hello`. Frames already sent are re-read from disk.
    mutating func rewind() {
        position = 0
        if isCancelled { cancelPending = false }
    }
}

/// What the HUD shows about the live link for one capture.
struct LiveStreamSnapshot: Equatable, Sendable {
    enum Phase: Equatable, Sendable {
        case connecting
        case waiting(String)
        case streaming
        case awaitingSaved
        case reconnecting(attempt: Int, of: Int, reason: String)
        case saved(path: String, frames: Int)
        case cancelled
        case failed(String)

        var isTerminal: Bool {
            switch self {
            case .saved, .cancelled, .failed: return true
            default: return false
            }
        }
    }

    enum Severity: Sendable { case info, good, warning, error }

    var macName: String
    var phase: Phase = .connecting
    /// Kept frames handed to the link so far (m).
    var framesQueued = 0
    /// Frames fully handed to TCP on the current connection (n).
    var framesSent = 0
    /// `frames_received` from the Mac's last `status`.
    var macFrames = 0
    /// Frames that did not fit in the memory queue and are streamed from the local files instead.
    var spilledFrames = 0
    /// Frames that could not be read back from disk — a real gap on the Mac.
    var missingFrames = 0
    var recordingEnded = false

    static let lagThreshold = 10

    var framesBehind: Int { max(0, framesQueued - framesSent) }
    var isLagging: Bool {
        guard !phase.isTerminal else { return false }
        return spilledFrames > 0 || framesBehind > Self.lagThreshold
    }

    var severity: Severity {
        switch phase {
        case .saved: return missingFrames > 0 ? .warning : .good
        case .failed: return .error
        case .reconnecting, .waiting: return .warning
        case .cancelled: return .info
        default: return (isLagging || missingFrames > 0) ? .warning : .info
        }
    }

    var hudText: String {
        let gap = missingFrames > 0 ? " · \(missingFrames) frame\(missingFrames == 1 ? "" : "s") unreadable, missing on Mac" : ""
        switch phase {
        case .connecting:
            return "Mac: connecting to \(macName)…"
        case .waiting(let reason):
            return "Mac: waiting for \(macName) (\(reason))"
        case .streaming:
            var s = recordingEnded ? "Mac: finishing · sending \(framesSent)/\(framesQueued) frames"
                                   : "Mac: connected to \(macName) · sending \(framesSent)/\(framesQueued) frames"
            if isLagging { s += " · lagging, \(framesBehind) behind (kept locally)" }
            return s + gap
        case .awaitingSaved:
            return "Mac: sent \(framesSent)/\(framesQueued) frames, waiting for \(macName) to save…" + gap
        case .reconnecting(let attempt, let of, let reason):
            return "Mac: link lost (\(reason)), reconnecting \(attempt)/\(of)…"
        case .saved(let path, let frames):
            return "Saved on Mac at \(path) (\(frames) frames)" + gap
        case .cancelled:
            return "Mac: recording discarded (cancel sent)"
        case .failed(let reason):
            return "Mac stream failed: \(reason). Mac has \(macFrames)/\(framesQueued) frames; the iPhone copy is complete."
        }
    }
}
