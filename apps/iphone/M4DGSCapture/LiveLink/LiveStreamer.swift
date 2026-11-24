import Foundation
import Network

/// Streams one capture to M4DGS Studio over one TCP connection (docs/formats.md §6).
///
/// * The writer hands over each kept frame's exact saved bytes via `CaptureWriterSink`; that call only enqueues on
///   this object's own serial queue, so it never blocks the writer or the ARSession.
/// * The send queue (`LiveOutbox`) keeps frame bytes in memory up to `memoryLimit`; beyond that frames are re-read
///   from the local capture directory when their turn comes. Frames go out strictly in order and none are dropped;
///   the HUD shows "lagging" while the link is behind.
/// * At most `maxInFlightBytes` are handed to NWConnection at once (completion = `.contentProcessed`).
/// * If the connection drops, it reconnects (up to `maxReconnects` times) and replays the whole capture from `hello`
///   (the Mac starts a fresh directory per connection, leaving the broken one as `capture.partial.json`).
///   When it gives up, the HUD reports how many frames the Mac has; the local capture is unaffected.
///
/// Thread-safety: all mutable state is confined to `queue`.
final class LiveStreamer: CaptureWriterSink, @unchecked Sendable {
    let id: UUID
    let macName: String

    static let memoryLimit = 64 << 20
    static let maxInFlightBytes = 4 << 20
    static let maxReconnects = 4
    static let connectTimeout: TimeInterval = 15
    static let stallTimeout: TimeInterval = 20
    static let savedTimeout: TimeInterval = 30

    private let endpoint: NWEndpoint
    private let directory: URL
    private let helloMessage: Data
    private let queue = DispatchQueue(label: "com.example.m4dgscapture.livelink", qos: .utility)
    private let onUpdate: @Sendable (LiveStreamSnapshot) -> Void

    // MARK: Confined to `queue`
    private var outbox = LiveOutbox(memoryLimit: LiveStreamer.memoryLimit)
    private var connection: NWConnection?
    private var generation = 0
    private var connected = false
    private var inFlightBytes = 0
    private var connectStartedAt = Date()
    private var lastProgress = Date()
    private var endSentAt: Date?
    private var reconnects = 0
    private var decoder = LiveMessageDecoder()
    private var missing: Set<Int> = []
    private var snapshot: LiveStreamSnapshot
    private var lastPosted: LiveStreamSnapshot?
    private var lastPostTime = Date.distantPast
    private var timer: DispatchSourceTimer?
    private var started = false

    init(id: UUID = UUID(), macName: String, endpoint: NWEndpoint, directory: URL, hello: LiveHello,
         onUpdate: @escaping @Sendable (LiveStreamSnapshot) -> Void) throws {
        self.id = id
        self.macName = macName
        self.endpoint = endpoint
        self.directory = directory
        self.helloMessage = try LiveLinkCodec.hello(hello)
        self.onUpdate = onUpdate
        self.snapshot = LiveStreamSnapshot(macName: macName)
    }

    /// Opens the connection (call once, after the writer was created successfully).
    func start() {
        queue.async {
            guard !self.started else { return }
            self.started = true
            self.startTimer()
            self.connect()
            self.post(force: true)
        }
    }

    /// Recording never started (writer creation failed): tear down without sending anything.
    func abandon() {
        queue.async {
            self.timer?.cancel()
            self.timer = nil
            self.connection?.cancel()
            self.connection = nil
        }
    }

    // MARK: - CaptureWriterSink (writer queue)

    func writerDidWrite(frame: FrameRecord, blobs: LiveFrameBlobs) {
        queue.async {
            guard !self.snapshot.phase.isTerminal else {
                self.snapshot.framesQueued += 1   // keep counting so the HUD shows the real gap
                self.post()
                return
            }
            self.outbox.addFrame(frame, blobs: blobs)
            self.pump()
            self.post()
        }
    }

    func writerDidFinish(points: LivePoints?, manifest: Data) {
        queue.async {
            self.outbox.finish(points: points, manifest: manifest)
            self.snapshot.recordingEnded = true
            self.lastProgress = Date()
            self.pump()
            self.post(force: true)
        }
    }

    func writerDidDiscard() {
        queue.async {
            self.outbox.cancel()
            self.snapshot.recordingEnded = true
            if self.outbox.isDrained || !self.connected {
                // Nothing on the wire to cancel (or no connection to send it on).
                self.terminate(.cancelled)
            } else {
                self.pump()
            }
        }
    }

    func writerDidFail(_ error: Error) {
        queue.async {
            // Local finalize failed; leave the Mac with its partial directory.
            self.terminate(.failed("the iPhone could not finish the capture (\(error.localizedDescription))"))
        }
    }

    // MARK: - Connection

    private func connect() {
        generation += 1
        let gen = generation
        connected = false
        inFlightBytes = 0
        connectStartedAt = Date()
        decoder.reset()
        let tcp = NWProtocolTCP.Options()
        tcp.enableKeepalive = true
        tcp.keepaliveIdle = 5
        tcp.keepaliveInterval = 2
        tcp.keepaliveCount = 4
        tcp.connectionTimeout = Int(Self.connectTimeout)
        let params = NWParameters(tls: nil, tcp: tcp)
        let conn = NWConnection(to: endpoint, using: params)
        connection = conn
        conn.stateUpdateHandler = { [weak self] state in
            guard let self else { return }
            // Handlers run on `queue` (the connection was started on it).
            guard gen == self.generation else { return }
            self.connectionStateChanged(state)
        }
        if reconnects == 0 { setPhase(.connecting) }
        conn.start(queue: queue)
    }

    private func connectionStateChanged(_ state: NWConnection.State) {
        switch state {
        case .ready:
            connected = true
            outbox.rewind()        // a fresh connection always starts with hello
            snapshot.framesSent = 0
            snapshot.macFrames = 0
            endSentAt = nil
            lastProgress = Date()
            setPhase(.streaming)
            receive(generation)
            pump()
        case .waiting(let error):
            if connected {
                connectionLost(error.localizedDescription)
            } else {
                setPhase(.waiting(error.localizedDescription))
            }
        case .failed(let error):
            connectionLost(error.localizedDescription)
        case .preparing, .setup, .cancelled:
            break
        @unknown default:
            break
        }
        post(force: true)
    }

    private func connectionLost(_ reason: String) {
        guard !snapshot.phase.isTerminal else { return }
        connection?.stateUpdateHandler = nil
        connection?.cancel()
        connection = nil
        connected = false
        generation += 1            // ignore late callbacks from the old connection
        inFlightBytes = 0
        if outbox.isCancelled {
            // Recording was discarded: nothing worth replaying. (If the cancel did not make it, the Mac keeps an
            // empty partial directory.)
            terminate(.cancelled)
            return
        }
        guard reconnects < Self.maxReconnects else {
            terminate(.failed(reason))
            return
        }
        reconnects += 1
        setPhase(.reconnecting(attempt: reconnects, of: Self.maxReconnects, reason: reason))
        post(force: true)
        let delay = min(8.0, pow(2.0, Double(reconnects - 1)))
        let gen = generation
        queue.asyncAfter(deadline: .now() + delay) { [weak self] in
            guard let self, gen == self.generation, !self.snapshot.phase.isTerminal else { return }
            self.connect()
        }
    }

    private func receive(_ gen: Int) {
        connection?.receive(minimumIncompleteLength: 1, maximumLength: 64 << 10) { [weak self] data, _, isComplete, error in
            guard let self, gen == self.generation else { return }
            if let data, !data.isEmpty {
                self.decoder.append(data)
                do {
                    while let msg = try self.decoder.next() {
                        self.handle(msg)
                        if self.snapshot.phase.isTerminal { return }
                    }
                } catch {
                    self.connectionLost("bad data from Mac: \(error)")
                    return
                }
            }
            if let error {
                self.connectionLost(error.localizedDescription)
            } else if isComplete {
                self.connectionLost("the Mac closed the connection")
            } else {
                self.receive(gen)
            }
        }
    }

    private func handle(_ msg: LiveMessage) {
        switch msg.type {
        case .status:
            if let s = try? msg.decodeHeader(LiveStatus.self) {
                snapshot.macFrames = s.framesReceived
                lastProgress = Date()
                post()
            }
        case .saved:
            let s = try? msg.decodeHeader(LiveSaved.self)
            terminate(.saved(path: s?.path ?? "(unknown path)", frames: s?.frames ?? snapshot.framesSent))
        default:
            break   // unknown / phone→Mac types are ignored
        }
    }

    // MARK: - Sending

    private func pump() {
        guard connected, let conn = connection, !snapshot.phase.isTerminal else { return }
        while inFlightBytes < Self.maxInFlightBytes, let step = outbox.takeNext() {
            var data: Data?
            var kind: SentKind = .other
            switch step {
            case .hello:
                data = helloMessage
            case .frame(let record, let blobs):
                guard let b = blobs ?? readFromDisk(record) else {
                    missing.insert(record.index)
                    snapshot.missingFrames = missing.count
                    continue   // reported in the HUD; capture.json on the Mac will reference the missing files
                }
                data = try? LiveLinkCodec.frame(record, blobs: b)
                if data == nil {
                    missing.insert(record.index)
                    snapshot.missingFrames = missing.count
                    continue
                }
                kind = .frame
            case .points(let p):
                data = try? LiveLinkCodec.points(p)
            case .end(let manifest):
                data = try? LiveLinkCodec.end(manifest: manifest)
                kind = .end
            case .cancel:
                data = try? LiveLinkCodec.cancel()
                kind = .cancel
            }
            guard let data else { continue }
            send(data, kind: kind, on: conn)
        }
        snapshot.spilledFrames = outbox.spilledFrames
        snapshot.framesQueued = outbox.frames.count
    }

    private enum SentKind { case frame, end, cancel, other }

    private func send(_ data: Data, kind: SentKind, on conn: NWConnection) {
        let gen = generation
        let n = data.count
        inFlightBytes += n
        conn.send(content: data, completion: .contentProcessed { [weak self] error in
            guard let self, gen == self.generation else { return }
            self.inFlightBytes -= n
            if let error {
                self.connectionLost(error.localizedDescription)
                return
            }
            self.lastProgress = Date()
            switch kind {
            case .frame:
                self.snapshot.framesSent += 1
            case .end:
                self.endSentAt = Date()
                self.setPhase(.awaitingSaved)
            case .cancel:
                self.terminate(.cancelled)
                return
            case .other:
                break
            }
            self.pump()
            self.post()
        })
    }

    private func readFromDisk(_ r: FrameRecord) -> LiveFrameBlobs? {
        func read(_ rel: String?) -> Data?? {
            guard let rel else { return .some(nil) }
            guard let d = try? Data(contentsOf: directory.appendingPathComponent(rel)) else { return nil }
            return .some(d)
        }
        guard let image = read(r.image), let image, let depth = read(r.depth), let conf = read(r.confidence) else { return nil }
        return LiveFrameBlobs(image: image, depth: depth, confidence: conf)
    }

    // MARK: - State

    private func setPhase(_ p: LiveStreamSnapshot.Phase) {
        guard !snapshot.phase.isTerminal else { return }
        snapshot.phase = p
    }

    private func terminate(_ p: LiveStreamSnapshot.Phase) {
        guard !snapshot.phase.isTerminal else { return }
        snapshot.phase = p
        snapshot.framesQueued = max(snapshot.framesQueued, outbox.frames.count)
        timer?.cancel()
        timer = nil
        generation += 1
        connection?.stateUpdateHandler = nil
        connection?.cancel()     // graceful close; everything we care about was already processed
        connection = nil
        connected = false
        outbox = LiveOutbox(memoryLimit: 0)   // release cached frame bytes; later frames are only counted
        post(force: true)
    }

    private func startTimer() {
        let t = DispatchSource.makeTimerSource(queue: queue)
        t.schedule(deadline: .now() + 1, repeating: 1)
        t.setEventHandler { [weak self] in self?.tick() }
        t.resume()
        timer = t
    }

    private func tick() {
        let now = Date()
        if connection != nil, !connected, now.timeIntervalSince(connectStartedAt) > Self.connectTimeout {
            connectionLost("could not reach \(macName)")
            post()
            return
        }
        switch snapshot.phase {
        case .streaming:
            if inFlightBytes > 0, now.timeIntervalSince(lastProgress) > Self.stallTimeout {
                connectionLost("no progress for \(Int(Self.stallTimeout)) s")
            }
        case .awaitingSaved:
            if let t = endSentAt, now.timeIntervalSince(t) > Self.savedTimeout {
                terminate(.failed("sent all \(snapshot.framesSent) frames but \(macName) did not confirm the save"))
            }
        default:
            break
        }
        post()
    }

    private func post(force: Bool = false) {
        snapshot.spilledFrames = max(snapshot.spilledFrames, outbox.spilledFrames)
        if !snapshot.phase.isTerminal { snapshot.framesQueued = outbox.frames.count }
        let now = Date()
        guard snapshot != lastPosted else { return }
        guard force || snapshot.phase.isTerminal || now.timeIntervalSince(lastPostTime) >= 0.1 else { return }
        lastPosted = snapshot
        lastPostTime = now
        onUpdate(snapshot)
    }
}
