import Foundation

/// The `hello` header (docs/formats.md §6).
public struct LiveHello: @unchecked Sendable {
    public var captureName: String
    public var deviceModel: String?
    public var systemVersion: String?
    public var imageWidth: Int?
    public var imageHeight: Int?
    public var depthWidth: Int?
    public var depthHeight: Int?
    public var hasLidar: Bool?
    /// The header as received (written into `capture.partial.json`).
    public var raw: [String: Any]
}

/// One received keyframe, for UIs (thumbnail + trajectory).
public struct LiveFrameInfo: @unchecked Sendable {
    public var index: Int?
    public var timestamp: Double?
    /// ARKit `camera_to_world`, row-major 4x4.
    public var cameraToWorld: [[Double]]?
    public var imagePath: String?
    /// The JPEG bytes as received (for a thumbnail).
    public var imageData: Data?
    public var trackingState: String?
    public var framesReceived: Int
    public var bytesReceived: Int
}

public enum LiveSessionEvent: @unchecked Sendable {
    case started(hello: LiveHello, directory: URL)
    case frame(LiveFrameInfo)
    case points(count: Int?, bytes: Int)
    case saved(directory: URL, frames: Int, warnings: [String])
    case cancelled(directory: URL)
    case ignored(String)
}

public struct LiveSessionLimits: Sendable {
    public var maxFrames = 100_000
    public var maxCaptureBytes: Int = 64 << 30
    public init() {}
}

/// Receiver-side state machine for one capture connection: consumes decoded messages, writes the §2 capture
/// directory under `capturesFolder/<capture_name>/` as frames arrive, and produces the Mac→phone replies.
/// Not thread-safe: drive it from one queue.
public final class LiveCaptureSession {
    public enum State: Equatable, Sendable { case awaitingHello, receiving, saved, cancelled, failed }

    public let capturesFolder: URL
    public let limits: LiveSessionLimits
    public private(set) var state: State = .awaitingHello
    public private(set) var hello: LiveHello?
    public private(set) var directory: URL?
    public private(set) var framesReceived = 0
    /// Wire bytes of every message handled so far (prefix + header + payload).
    public private(set) var bytesReceived = 0
    public private(set) var warnings: [String] = []

    private var frameObjects: [[String: Any]] = []
    private var pointsPLY: Data?
    private var createdDirs: Set<String> = []
    private var lastPartialWrite = 0
    private let fm = FileManager.default

    public static let partialFileName = "capture.partial.json"
    public static let partialEvery = 10

    public init(capturesFolder: URL, limits: LiveSessionLimits = LiveSessionLimits()) {
        self.capturesFolder = capturesFolder
        self.limits = limits
    }

    public var isFinished: Bool { state == .saved || state == .cancelled || state == .failed }

    /// Handles one message. Returns the replies to send (in order) and the events for the UI.
    /// Throws on protocol violations / unsafe input; the caller should then close the connection
    /// (call `connectionClosed()` to leave `capture.partial.json`).
    public func handle(_ msg: LiveMessage) throws -> (replies: [LiveMessage], events: [LiveSessionEvent]) {
        bytesReceived += LiveLink.prefixLength + msg.header.count + msg.payload.count
        guard let type = msg.type else { return ([], [.ignored("unknown message type \(msg.rawType)")]) }
        if isFinished { return ([], [.ignored("\(type) after the capture finished")]) }
        do {
            switch type {
            case .hello: return try handleHello(msg)
            case .frame: return try handleFrame(msg)
            case .points: return try handlePoints(msg)
            case .end: return try handleEnd(msg)
            case .cancel: return try handleCancel()
            case .status, .saved: return ([], [.ignored("\(type) is a Mac→phone message")])
            }
        } catch {
            state = .failed
            throw error
        }
    }

    /// The connection went away (or was closed after an error). If the capture did not complete, writes
    /// `capture.partial.json` and returns the directory.
    @discardableResult
    public func connectionClosed() -> URL? {
        guard let dir = directory, state == .receiving || state == .failed else { return nil }
        writePartial()
        if state == .receiving { state = .failed }
        return dir
    }

    // MARK: messages

    private func requireReceiving(_ what: LiveMessageType) throws {
        if state == .awaitingHello { throw LiveLinkError.protocolViolation("\(what) before hello") }
    }

    private func handleHello(_ msg: LiveMessage) throws -> ([LiveMessage], [LiveSessionEvent]) {
        guard state == .awaitingHello else { throw LiveLinkError.protocolViolation("second hello") }
        let h = try msg.headerObject()
        let proto = intValue(h["protocol"]) ?? LiveLink.protocolVersion
        guard proto == LiveLink.protocolVersion else { throw LiveLinkError.unsupportedProtocol(proto) }
        guard let rawName = h["capture_name"] as? String else {
            throw LiveLinkError.protocolViolation("hello without capture_name")
        }
        guard let name = LivePaths.sanitizeCaptureName(rawName) else { throw LiveLinkError.unsafePath(rawName) }
        let info = LiveHello(captureName: name, deviceModel: h["device_model"] as? String,
                             systemVersion: h["system_version"] as? String,
                             imageWidth: intValue(h["image_width"]), imageHeight: intValue(h["image_height"]),
                             depthWidth: intValue(h["depth_width"]), depthHeight: intValue(h["depth_height"]),
                             hasLidar: h["has_lidar"] as? Bool, raw: h)
        try fm.createDirectory(at: capturesFolder, withIntermediateDirectories: true)
        let dir = uniqueDirectory(for: name)
        try fm.createDirectory(at: dir, withIntermediateDirectories: false)
        directory = dir
        hello = info
        state = .receiving
        writePartial()
        return ([], [.started(hello: info, directory: dir)])
    }

    private func handleFrame(_ msg: LiveMessage) throws -> ([LiveMessage], [LiveSessionEvent]) {
        try requireReceiving(.frame)
        guard let dir = directory else { throw LiveLinkError.protocolViolation("no capture directory") }
        guard framesReceived < limits.maxFrames else { throw LiveLinkError.protocolViolation("more than \(limits.maxFrames) frames") }
        guard bytesReceived <= limits.maxCaptureBytes else { throw LiveLinkError.protocolViolation("capture exceeds \(limits.maxCaptureBytes) bytes") }
        var h = try msg.headerObject()
        let parts: [(key: String, size: String)] = [("image", "image_bytes"), ("depth", "depth_bytes"), ("confidence", "confidence_bytes")]
        var sizes: [Int] = []
        for p in parts {
            let v = h[p.size]
            let n = (v == nil || v is NSNull) ? 0 : intValue(v)
            guard let n, n >= 0 else { throw LiveLinkError.protocolViolation("frame \(p.size) is not a non-negative integer") }
            sizes.append(n)
        }
        guard sizes.reduce(0, +) == msg.payload.count else {
            throw LiveLinkError.protocolViolation("frame byte counts \(sizes) do not add up to payload_len \(msg.payload.count)")
        }
        guard sizes[0] > 0 else { throw LiveLinkError.protocolViolation("frame without image bytes") }
        // validate every path before writing anything
        var targets: [(String, Range<Int>)] = []
        var off = 0
        for (p, n) in zip(parts, sizes) {
            defer { off += n }
            guard n > 0 else { continue }
            guard let raw = h[p.key] as? String else { throw LiveLinkError.protocolViolation("frame has \(p.size)=\(n) but no \(p.key) path") }
            guard let safe = LivePaths.sanitizeFramePath(raw) else { throw LiveLinkError.unsafePath(raw) }
            targets.append((safe, off..<(off + n)))
        }
        var imageData: Data?
        for (i, (rel, r)) in targets.enumerated() {
            let comps = rel.split(separator: "/")
            let sub = String(comps[0])
            if !createdDirs.contains(sub) {
                try fm.createDirectory(at: dir.appendingPathComponent(sub, isDirectory: true), withIntermediateDirectories: true)
                createdDirs.insert(sub)
            }
            let bytes = msg.payload.subdata(in: r)
            try bytes.write(to: dir.appendingPathComponent(sub).appendingPathComponent(String(comps[1])), options: .atomic)
            if i == 0 { imageData = bytes }
        }
        for p in parts { h.removeValue(forKey: p.size) }
        frameObjects.append(h)
        framesReceived += 1
        if framesReceived - lastPartialWrite >= Self.partialEvery { writePartial() }
        let info = LiveFrameInfo(index: intValue(h["index"]), timestamp: doubleValue(h["timestamp"]),
                                 cameraToWorld: matrix4(h["camera_to_world"]), imagePath: targets.first?.0,
                                 imageData: imageData, trackingState: h["tracking_state"] as? String,
                                 framesReceived: framesReceived, bytesReceived: bytesReceived)
        let status = try LiveMessage(.status, json: ["frames_received": framesReceived, "bytes_received": bytesReceived])
        return ([status], [.frame(info)])
    }

    private func handlePoints(_ msg: LiveMessage) throws -> ([LiveMessage], [LiveSessionEvent]) {
        try requireReceiving(.points)
        let h = (try? msg.headerObject()) ?? [:]
        if !msg.payload.starts(with: Array("ply".utf8)) { warnings.append("points payload does not start with 'ply'") }
        pointsPLY = msg.payload
        return ([], [.points(count: intValue(h["count"]), bytes: msg.payload.count)])
    }

    private func handleEnd(_ msg: LiveMessage) throws -> ([LiveMessage], [LiveSessionEvent]) {
        try requireReceiving(.end)
        guard let dir = directory else { throw LiveLinkError.protocolViolation("no capture directory") }
        let doc = try msg.headerObject()
        if (doc["format"] as? String) != "m4dgs-arkit-capture" {
            warnings.append("capture.json format is \(String(describing: doc["format"] ?? "missing")), expected m4dgs-arkit-capture")
        }
        let frames = doc["frames"] as? [[String: Any]] ?? []
        if frames.isEmpty { warnings.append("capture.json has no frames") }
        var missing = 0
        for f in frames {
            for key in ["image", "depth", "confidence"] {
                guard let raw = f[key] as? String else { continue }
                // capture.json is read by the Python loader: never let it point outside the capture
                guard let safe = LivePaths.sanitizeFramePath(raw) else { throw LiveLinkError.unsafePath(raw) }
                if !fm.fileExists(atPath: dir.appendingPathComponent(safe).path) { missing += 1 }
            }
        }
        if missing > 0 { warnings.append("\(missing) file(s) referenced by capture.json were not received") }
        if frames.count != framesReceived { warnings.append("capture.json lists \(frames.count) frames, received \(framesReceived)") }
        // capture.json is written verbatim (the phone's local copy is byte-identical)
        try msg.header.write(to: dir.appendingPathComponent("capture.json"), options: .atomic)
        if let ply = pointsPLY { try ply.write(to: dir.appendingPathComponent("points.ply"), options: .atomic) }
        try? fm.removeItem(at: dir.appendingPathComponent(Self.partialFileName))
        state = .saved
        let saved = try LiveMessage(.saved, json: ["path": dir.path, "frames": framesReceived])
        return ([saved], [.saved(directory: dir, frames: framesReceived, warnings: warnings)])
    }

    private func handleCancel() throws -> ([LiveMessage], [LiveSessionEvent]) {
        try requireReceiving(.cancel)
        guard let dir = directory else { throw LiveLinkError.protocolViolation("no capture directory") }
        try? fm.removeItem(at: dir)
        state = .cancelled
        return ([], [.cancelled(directory: dir)])
    }

    // MARK: helpers

    private func uniqueDirectory(for name: String) -> URL {
        var url = capturesFolder.appendingPathComponent(name, isDirectory: true)
        var k = 1
        while fm.fileExists(atPath: url.path) {
            url = capturesFolder.appendingPathComponent("\(name)_\(k)", isDirectory: true)
            k += 1
        }
        return url
    }

    private func writePartial() {
        guard let dir = directory else { return }
        lastPartialWrite = framesReceived
        let doc: [String: Any] = [
            "format": "m4dgs-arkit-capture-partial",
            "complete": false,
            "hello": hello?.raw ?? [:],
            "frames_received": framesReceived,
            "frames": frameObjects,
        ]
        if let d = try? JSONSerialization.data(withJSONObject: doc, options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]) {
            try? d.write(to: dir.appendingPathComponent(Self.partialFileName), options: .atomic)
        }
    }
}

func intValue(_ v: Any?) -> Int? {
    switch v {
    case let i as Int: return i
    case let n as NSNumber:
        let d = n.doubleValue
        return d == d.rounded() && abs(d) < 1e15 ? Int(d) : nil
    default: return nil
    }
}

func doubleValue(_ v: Any?) -> Double? { (v as? NSNumber)?.doubleValue }

func matrix4(_ v: Any?) -> [[Double]]? {
    guard let rows = v as? [[Any]], rows.count == 4 else { return nil }
    let m = rows.map { $0.compactMap(doubleValue) }
    return m.allSatisfy({ $0.count == 4 }) ? m : nil
}
