import Foundation

// Live link wire protocol (docs/formats.md §6): iPhone capture app ⇄ M4DGS Studio over TCP.
//
//   offset 0   4 bytes  magic "M4DG"
//   offset 4   1 byte   type
//   offset 5   3 bytes  reserved (0)
//   offset 8   uint32   header_len  (big-endian)
//   offset 12  uint32   payload_len (big-endian)
//   offset 16  header_len bytes   UTF-8 JSON header
//              payload_len bytes  binary payload
//
// Pure Foundation (no Network.framework) so it is unit-testable; the NWListener glue lives in the app target.

public enum LiveLink {
    public static let serviceType = "_m4dgs-live._tcp"
    public static let protocolVersion = 1
    public static let magic: [UInt8] = Array("M4DG".utf8)
    public static let prefixLength = 16
}

public enum LiveMessageType: UInt8, Sendable, CustomStringConvertible {
    case hello = 1, frame = 2, points = 3, end = 4, status = 5, cancel = 6, saved = 7
    public var description: String {
        switch self {
        case .hello: return "hello"
        case .frame: return "frame"
        case .points: return "points"
        case .end: return "end"
        case .status: return "status"
        case .cancel: return "cancel"
        case .saved: return "saved"
        }
    }
}

public struct LiveMessage: Sendable, Equatable {
    /// Raw type byte (unknown types are passed through so the receiver can ignore them).
    public var rawType: UInt8
    /// UTF-8 JSON header bytes, exactly as received.
    public var header: Data
    public var payload: Data

    public var type: LiveMessageType? { LiveMessageType(rawValue: rawType) }

    public init(rawType: UInt8, header: Data, payload: Data = Data()) {
        self.rawType = rawType; self.header = header; self.payload = payload
    }

    public init(_ type: LiveMessageType, header: Data, payload: Data = Data()) {
        self.init(rawType: type.rawValue, header: header, payload: payload)
    }

    /// Builds a message whose header is `object` serialized as JSON.
    public init(_ type: LiveMessageType, json object: Any, payload: Data = Data()) throws {
        let h = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys, .withoutEscapingSlashes])
        self.init(type, header: h, payload: payload)
    }

    /// Header parsed as a JSON object.
    public func headerObject() throws -> [String: Any] {
        guard let o = try? JSONSerialization.jsonObject(with: header), let d = o as? [String: Any] else {
            throw LiveLinkError.badHeaderJSON(type.map(String.init(describing:)) ?? "type \(rawType)")
        }
        return d
    }

    /// Wire encoding (prefix + header + payload).
    public func encoded() -> Data {
        var d = Data(capacity: LiveLink.prefixLength + header.count + payload.count)
        d.append(contentsOf: LiveLink.magic)
        d.append(rawType)
        d.append(contentsOf: [0, 0, 0])
        d.appendBigEndian(UInt32(header.count))
        d.appendBigEndian(UInt32(payload.count))
        d.append(header)
        d.append(payload)
        return d
    }

    public static func == (a: LiveMessage, b: LiveMessage) -> Bool {
        a.rawType == b.rawType && a.header == b.header && a.payload == b.payload
    }
}

public enum LiveLinkError: Error, Equatable, CustomStringConvertible {
    case badMagic([UInt8])
    case headerTooLarge(UInt32, limit: Int)
    case payloadTooLarge(UInt32, limit: Int)
    case badHeaderJSON(String)
    case protocolViolation(String)
    case unsafePath(String)
    case unsupportedProtocol(Int)

    public var description: String {
        switch self {
        case .badMagic(let b): return "bad magic \(b.map { String(format: "%02x", $0) }.joined()) (expected 4d344447 \"M4DG\")"
        case .headerTooLarge(let n, let l): return "header_len \(n) exceeds limit \(l)"
        case .payloadTooLarge(let n, let l): return "payload_len \(n) exceeds limit \(l)"
        case .badHeaderJSON(let t): return "\(t): header is not a JSON object"
        case .protocolViolation(let s): return "protocol violation: \(s)"
        case .unsafePath(let p): return "rejected unsafe path \(p.debugDescription)"
        case .unsupportedProtocol(let v): return "unsupported protocol version \(v) (this Mac speaks \(LiveLink.protocolVersion))"
        }
    }
}

public struct LiveLinkLimits: Sendable {
    /// `end` carries the whole capture.json (≈0.7 KB per frame), so this is generous.
    public var maxHeaderBytes = 32 << 20
    /// One frame (JPEG + depth + confidence) or the points.ply.
    public var maxPayloadBytes = 512 << 20
    public init() {}
    public init(maxHeaderBytes: Int, maxPayloadBytes: Int) {
        self.maxHeaderBytes = maxHeaderBytes; self.maxPayloadBytes = maxPayloadBytes
    }
}

/// Incremental decoder: feed it whatever TCP delivers (any split, any coalescing) and pull complete messages.
/// Malformed prefixes throw as soon as enough bytes are present to tell (the connection should then be dropped).
public struct LiveMessageDecoder: Sendable {
    public var limits: LiveLinkLimits
    private var buffer = Data()
    public private(set) var totalBytes = 0

    public init(limits: LiveLinkLimits = LiveLinkLimits()) { self.limits = limits }

    /// Bytes buffered but not yet returned as a message.
    public var bufferedBytes: Int { buffer.count }

    public mutating func append(_ data: Data) {
        buffer.append(data)
        totalBytes += data.count
    }

    /// Next complete message, or nil if more bytes are needed.
    public mutating func next() throws -> LiveMessage? {
        let n = buffer.count
        if n == 0 { return nil }
        let s = buffer.startIndex
        // reject garbage early, even before the whole 16-byte prefix arrived
        let m = min(n, 4)
        for i in 0..<m where buffer[s + i] != LiveLink.magic[i] {
            throw LiveLinkError.badMagic(Array(buffer[s..<(s + m)]))
        }
        guard n >= LiveLink.prefixLength else { return nil }
        let type = buffer[s + 4]
        let hlen = buffer.readBigEndianUInt32(at: s + 8)
        let plen = buffer.readBigEndianUInt32(at: s + 12)
        if Int(hlen) > limits.maxHeaderBytes { throw LiveLinkError.headerTooLarge(hlen, limit: limits.maxHeaderBytes) }
        if Int(plen) > limits.maxPayloadBytes { throw LiveLinkError.payloadTooLarge(plen, limit: limits.maxPayloadBytes) }
        let total = LiveLink.prefixLength + Int(hlen) + Int(plen)
        guard n >= total else { return nil }
        let h0 = s + LiveLink.prefixLength, p0 = h0 + Int(hlen), end = s + total
        // Data(...) re-bases the slices to zero-based, independent storage
        let msg = LiveMessage(rawType: type, header: Data(buffer[h0..<p0]), payload: Data(buffer[p0..<end]))
        if end == buffer.endIndex { buffer = Data() } else { buffer.removeSubrange(s..<end) }
        return msg
    }

    /// Drains every complete message currently buffered.
    public mutating func drain() throws -> [LiveMessage] {
        var out: [LiveMessage] = []
        while let m = try next() { out.append(m) }
        return out
    }
}

// MARK: - path sanitization

public enum LivePaths {
    /// Subdirectories a frame may write into (docs/formats.md §2).
    public static let frameDirectories: Set<String> = ["images", "depth", "confidence"]
    static let allowed = Set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.")

    /// A safe single directory name for `capture_name`: rejects separators, `.`/`..`, hidden names, control
    /// characters and empty / over-long names; other unusual characters become `_`.
    public static func sanitizeCaptureName(_ raw: String) -> String? {
        let s = raw.trimmingCharacters(in: .whitespaces)
        guard !s.isEmpty, s.count <= 128, !s.hasPrefix("."), s != "..",
              !s.contains("/"), !s.contains("\\"), !s.contains(":"),
              !s.unicodeScalars.contains(where: { $0.value < 0x20 || $0.value == 0x7f }) else { return nil }
        let mapped = String(s.map { allowed.contains($0) ? $0 : "_" })
        return mapped.allSatisfy({ $0 == "_" || $0 == "." }) ? nil : mapped
    }

    /// Validates a frame file path from a header: exactly `<images|depth|confidence>/<file>` where `<file>` is a
    /// plain name from `[A-Za-z0-9_.-]` not starting with `.`. Anything else (absolute paths, `..`, extra
    /// components, backslashes, NULs, unicode look-alikes) is rejected rather than rewritten.
    public static func sanitizeFramePath(_ raw: String) -> String? {
        guard raw.utf8.count <= 160, !raw.hasPrefix("/") else { return nil }
        let parts = raw.split(separator: "/", omittingEmptySubsequences: false).map(String.init)
        guard parts.count == 2, frameDirectories.contains(parts[0]) else { return nil }
        let f = parts[1]
        guard !f.isEmpty, f.count <= 128, !f.hasPrefix("."), f.allSatisfy({ allowed.contains($0) }) else { return nil }
        return parts[0] + "/" + f
    }
}

// MARK: - training log parsing (scripts/train_4dgs.py)

public struct TrainProgress: Sendable, Equatable {
    public var iteration: Int
    public var loss: Double?
    public var psnr: Double?
    public var gaussians: Int?

    /// Parses `[train] it  4100  loss 0.0123  psnr 45.70  gaussians 123456  12 ms/it`.
    public static func parse(_ line: String) -> TrainProgress? {
        guard let r = line.range(of: "[train] it") else { return nil }
        let toks = line[r.upperBound...].split(whereSeparator: { $0 == " " || $0 == "\t" })
        guard let first = toks.first, let it = Int(first) else { return nil }
        var p = TrainProgress(iteration: it)
        var i = 1
        while i + 1 < toks.count {
            switch toks[i] {
            case "loss": p.loss = Double(toks[i + 1])
            case "psnr": p.psnr = Double(toks[i + 1])
            case "gaussians": p.gaussians = Int(toks[i + 1])
            default: i -= 1
            }
            i += 2
        }
        return p
    }
}

// MARK: - byte helpers

extension Data {
    mutating func appendBigEndian(_ v: UInt32) {
        append(contentsOf: [UInt8(v >> 24 & 0xff), UInt8(v >> 16 & 0xff), UInt8(v >> 8 & 0xff), UInt8(v & 0xff)])
    }
    func readBigEndianUInt32(at i: Index) -> UInt32 {
        UInt32(self[i]) << 24 | UInt32(self[i + 1]) << 16 | UInt32(self[i + 2]) << 8 | UInt32(self[i + 3])
    }
}
