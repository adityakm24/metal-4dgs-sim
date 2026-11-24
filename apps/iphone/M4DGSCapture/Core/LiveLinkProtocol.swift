import Foundation

// Pure wire format of the phone ⇄ Mac live link (docs/formats.md §6). Like the rest of Core/, this file must not
// import Network/ARKit/UIKit: it is compiled and round-trip checked on macOS by apps/iphone/Tests/run_core_checks.sh,
// and the bytes it produces are re-parsed by an independent Python decoder (Tests/live_link_decode_check.py).
//
// Every message:
//   offset 0   4 bytes  magic "M4DG"
//   offset 4   1 byte   type
//   offset 5   3 bytes  reserved (0)
//   offset 8   uint32   header_len  (big-endian)
//   offset 12  uint32   payload_len (big-endian)
//   offset 16  header_len bytes UTF-8 JSON header, then payload_len bytes binary payload

enum LiveLinkProtocol {
    static let version = 1
    static let serviceType = "_m4dgs-live._tcp"
    static let magic: [UInt8] = Array("M4DG".utf8)
    static let prefixLength = 16
}

enum LiveMessageType: UInt8, Sendable, CaseIterable {
    case hello = 1      // phone → Mac
    case frame = 2      // phone → Mac
    case points = 3     // phone → Mac
    case end = 4        // phone → Mac
    case status = 5     // Mac → phone
    case cancel = 6     // phone → Mac
    case saved = 7      // Mac → phone
}

enum LiveLinkError: Error, Equatable, CustomStringConvertible {
    case badMagic([UInt8])
    case reservedNotZero([UInt8])
    case headerTooLarge(Int)
    case payloadTooLarge(Int)
    case messageTooLarge(Int)
    case invalidFrame(String)

    var description: String {
        switch self {
        case .badMagic(let b): return "bad magic \(b) (expected \"M4DG\")"
        case .reservedNotZero(let b): return "reserved bytes not zero: \(b)"
        case .headerTooLarge(let n): return "header_len \(n) exceeds the limit"
        case .payloadTooLarge(let n): return "payload_len \(n) exceeds the limit"
        case .messageTooLarge(let n): return "message of \(n) bytes does not fit the 32-bit length fields"
        case .invalidFrame(let s): return "invalid frame: \(s)"
        }
    }
}

/// One decoded message. `rawType` is kept so unknown (future) types can be skipped instead of failing.
struct LiveMessage: Equatable, Sendable {
    var rawType: UInt8
    var header: Data
    var payload: Data

    var type: LiveMessageType? { LiveMessageType(rawValue: rawType) }

    init(type: LiveMessageType, header: Data, payload: Data = Data()) {
        self.rawType = type.rawValue
        self.header = header
        self.payload = payload
    }

    init(rawType: UInt8, header: Data, payload: Data) {
        self.rawType = rawType
        self.header = header
        self.payload = payload
    }

    func decodeHeader<T: Decodable>(_ type: T.Type) throws -> T {
        try JSONDecoder().decode(T.self, from: header)
    }

    /// prefix ‖ header ‖ payload.
    func encoded() throws -> Data {
        try LiveLinkCodec.message(rawType: rawType, header: header, payloadParts: [payload])
    }
}

// MARK: - Headers

/// type 1 `hello`.
struct LiveHello: Codable, Equatable, Sendable {
    var protocolVersion: Int = LiveLinkProtocol.version
    var captureName: String
    var deviceModel: String
    var systemVersion: String
    var imageWidth: Int
    var imageHeight: Int
    var depthWidth: Int?
    var depthHeight: Int?
    var hasLidar: Bool

    enum CodingKeys: String, CodingKey {
        case protocolVersion = "protocol"
        case captureName = "capture_name"
        case deviceModel = "device_model"
        case systemVersion = "system_version"
        case imageWidth = "image_width"
        case imageHeight = "image_height"
        case depthWidth = "depth_width"
        case depthHeight = "depth_height"
        case hasLidar = "has_lidar"
    }

    init(captureName: String, deviceModel: String, systemVersion: String, imageWidth: Int, imageHeight: Int,
         depthWidth: Int?, depthHeight: Int?, hasLidar: Bool) {
        self.captureName = captureName
        self.deviceModel = deviceModel
        self.systemVersion = systemVersion
        self.imageWidth = imageWidth
        self.imageHeight = imageHeight
        self.depthWidth = depthWidth
        self.depthHeight = depthHeight
        self.hasLidar = hasLidar
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(protocolVersion, forKey: .protocolVersion)
        try c.encode(captureName, forKey: .captureName)
        try c.encode(deviceModel, forKey: .deviceModel)
        try c.encode(systemVersion, forKey: .systemVersion)
        try c.encode(imageWidth, forKey: .imageWidth)
        try c.encode(imageHeight, forKey: .imageHeight)
        try c.encode(depthWidth, forKey: .depthWidth)     // Optional → null (same as capture.json)
        try c.encode(depthHeight, forKey: .depthHeight)
        try c.encode(hasLidar, forKey: .hasLidar)
    }
}

/// type 2 `frame`: the §2 frame object plus the byte counts of the three payload parts.
struct LiveFrameHeader: Codable, Equatable, Sendable {
    var frame: FrameRecord
    var imageBytes: Int
    var depthBytes: Int
    var confidenceBytes: Int

    private enum ByteKeys: String, CodingKey {
        case imageBytes = "image_bytes"
        case depthBytes = "depth_bytes"
        case confidenceBytes = "confidence_bytes"
    }

    init(frame: FrameRecord, imageBytes: Int, depthBytes: Int, confidenceBytes: Int) {
        self.frame = frame
        self.imageBytes = imageBytes
        self.depthBytes = depthBytes
        self.confidenceBytes = confidenceBytes
    }

    init(from decoder: Decoder) throws {
        frame = try FrameRecord(from: decoder)
        let c = try decoder.container(keyedBy: ByteKeys.self)
        imageBytes = try c.decode(Int.self, forKey: .imageBytes)
        depthBytes = try c.decodeIfPresent(Int.self, forKey: .depthBytes) ?? 0
        confidenceBytes = try c.decodeIfPresent(Int.self, forKey: .confidenceBytes) ?? 0
    }

    /// One flat JSON object: FrameRecord's keys (identical to capture.json) + the three `*_bytes` keys.
    func encode(to encoder: Encoder) throws {
        try frame.encode(to: encoder)
        var c = encoder.container(keyedBy: ByteKeys.self)
        try c.encode(imageBytes, forKey: .imageBytes)
        try c.encode(depthBytes, forKey: .depthBytes)
        try c.encode(confidenceBytes, forKey: .confidenceBytes)
    }
}

/// type 3 `points`.
struct LivePointsHeader: Codable, Equatable, Sendable {
    var count: Int
}

/// type 5 `status` (Mac → phone, after every frame).
struct LiveStatus: Codable, Equatable, Sendable {
    var framesReceived: Int
    var bytesReceived: Int

    enum CodingKeys: String, CodingKey {
        case framesReceived = "frames_received"
        case bytesReceived = "bytes_received"
    }
}

/// type 7 `saved` (Mac → phone, after `end` was written).
struct LiveSaved: Codable, Equatable, Sendable {
    var path: String
    var frames: Int
}

/// The binary parts of one keyframe, byte-identical to the files the writer saved locally.
struct LiveFrameBlobs: Equatable, Sendable {
    var image: Data
    var depth: Data?
    var confidence: Data?

    var byteCount: Int { image.count + (depth?.count ?? 0) + (confidence?.count ?? 0) }
}

/// `points.ply` bytes plus the vertex count for the `points` header.
struct LivePoints: Equatable, Sendable {
    var count: Int
    var ply: Data
}

// MARK: - Encoding

enum LiveLinkCodec {
    static func jsonHeader<T: Encodable>(_ value: T) throws -> Data {
        let enc = JSONEncoder()
        enc.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return try enc.encode(value)
    }

    static func prefix(rawType: UInt8, headerLength: Int, payloadLength: Int) throws -> Data {
        guard headerLength >= 0, headerLength <= Int(UInt32.max) else { throw LiveLinkError.messageTooLarge(headerLength) }
        guard payloadLength >= 0, payloadLength <= Int(UInt32.max) else { throw LiveLinkError.messageTooLarge(payloadLength) }
        var d = Data(capacity: LiveLinkProtocol.prefixLength)
        d.append(contentsOf: LiveLinkProtocol.magic)
        d.append(contentsOf: [rawType, 0, 0, 0])
        appendBigEndian(UInt32(headerLength), to: &d)
        appendBigEndian(UInt32(payloadLength), to: &d)
        return d
    }

    /// prefix ‖ header ‖ payloadParts[0] ‖ payloadParts[1] ‖ … in one buffer.
    static func message(rawType: UInt8, header: Data, payloadParts: [Data]) throws -> Data {
        let payloadLength = payloadParts.reduce(0) { $0 + $1.count }
        var d = Data(capacity: LiveLinkProtocol.prefixLength + header.count + payloadLength)
        d.append(try prefix(rawType: rawType, headerLength: header.count, payloadLength: payloadLength))
        d.append(header)
        for p in payloadParts { d.append(p) }
        return d
    }

    static func message(_ type: LiveMessageType, header: Data, payloadParts: [Data] = []) throws -> Data {
        try message(rawType: type.rawValue, header: header, payloadParts: payloadParts)
    }

    static func hello(_ hello: LiveHello) throws -> Data {
        try message(.hello, header: jsonHeader(hello))
    }

    /// `frame`: header = §2 frame object + byte counts; payload = JPEG ‖ depth PNG ‖ confidence PNG.
    static func frame(_ record: FrameRecord, blobs: LiveFrameBlobs) throws -> Data {
        guard !blobs.image.isEmpty else { throw LiveLinkError.invalidFrame("frame \(record.index) has no image bytes") }
        guard (record.depth != nil) == (blobs.depth != nil) else {
            throw LiveLinkError.invalidFrame("frame \(record.index): depth path and depth bytes disagree")
        }
        guard (record.confidence != nil) == (blobs.confidence != nil) else {
            throw LiveLinkError.invalidFrame("frame \(record.index): confidence path and confidence bytes disagree")
        }
        let header = LiveFrameHeader(frame: record, imageBytes: blobs.image.count,
                                     depthBytes: blobs.depth?.count ?? 0,
                                     confidenceBytes: blobs.confidence?.count ?? 0)
        var parts = [blobs.image]
        if let d = blobs.depth { parts.append(d) }
        if let c = blobs.confidence { parts.append(c) }
        return try message(.frame, header: jsonHeader(header), payloadParts: parts)
    }

    static func points(_ points: LivePoints) throws -> Data {
        try message(.points, header: jsonHeader(LivePointsHeader(count: points.count)), payloadParts: [points.ply])
    }

    /// `end`: the header IS the capture.json document, byte for byte as written locally.
    static func end(manifest: Data) throws -> Data {
        try message(.end, header: manifest)
    }

    static func cancel() throws -> Data {
        try message(.cancel, header: Data("{}".utf8))
    }

    static func status(_ s: LiveStatus) throws -> Data {
        try message(.status, header: jsonHeader(s))
    }

    static func saved(_ s: LiveSaved) throws -> Data {
        try message(.saved, header: jsonHeader(s))
    }

    /// Splits a `frame` message payload according to its header (receiver side / tests).
    static func splitFrame(_ msg: LiveMessage) throws -> (LiveFrameHeader, LiveFrameBlobs) {
        let h = try msg.decodeHeader(LiveFrameHeader.self)
        guard h.imageBytes > 0, h.depthBytes >= 0, h.confidenceBytes >= 0,
              h.imageBytes + h.depthBytes + h.confidenceBytes == msg.payload.count else {
            throw LiveLinkError.invalidFrame("byte counts \(h.imageBytes)+\(h.depthBytes)+\(h.confidenceBytes) != payload \(msg.payload.count)")
        }
        let p = msg.payload
        let s = p.startIndex
        let image = p.subdata(in: s ..< s + h.imageBytes)
        let depth = h.depthBytes > 0 ? p.subdata(in: s + h.imageBytes ..< s + h.imageBytes + h.depthBytes) : nil
        let confStart = s + h.imageBytes + h.depthBytes
        let conf = h.confidenceBytes > 0 ? p.subdata(in: confStart ..< confStart + h.confidenceBytes) : nil
        return (h, LiveFrameBlobs(image: image, depth: depth, confidence: conf))
    }

    private static func appendBigEndian(_ v: UInt32, to d: inout Data) {
        d.append(contentsOf: [UInt8(truncatingIfNeeded: v >> 24), UInt8(truncatingIfNeeded: v >> 16),
                              UInt8(truncatingIfNeeded: v >> 8), UInt8(truncatingIfNeeded: v)])
    }
}

// MARK: - Streaming decoder

/// Incremental decoder for a TCP byte stream: `append` whatever arrived, then call `next()` until it returns nil.
struct LiveMessageDecoder: Sendable {
    var maxHeaderLength: Int
    var maxPayloadLength: Int
    private var buffer = Data()
    private var offset = 0

    /// Defaults are sized for the phone side, which only receives the tiny `status` / `saved` messages.
    init(maxHeaderLength: Int = 1 << 20, maxPayloadLength: Int = 1 << 20) {
        self.maxHeaderLength = maxHeaderLength
        self.maxPayloadLength = maxPayloadLength
    }

    var bufferedByteCount: Int { buffer.count - offset }

    mutating func append(_ data: Data) {
        if offset > 0 && (offset == buffer.count || offset >= 1 << 16) {
            buffer = Data(buffer[(buffer.startIndex + offset)...])
            offset = 0
        }
        buffer.append(data)
    }

    mutating func reset() {
        buffer = Data()
        offset = 0
    }

    /// Next complete message, nil if more bytes are needed. Throws on a corrupt stream (the connection should be closed).
    mutating func next() throws -> LiveMessage? {
        let available = buffer.count - offset
        guard available >= LiveLinkProtocol.prefixLength else { return nil }
        let base = buffer.startIndex + offset
        let magic = [UInt8](buffer[base ..< base + 4])
        guard magic == LiveLinkProtocol.magic else { throw LiveLinkError.badMagic(magic) }
        let reserved = [UInt8](buffer[base + 5 ..< base + 8])
        guard reserved == [0, 0, 0] else { throw LiveLinkError.reservedNotZero(reserved) }
        let rawType = buffer[base + 4]
        let headerLength = Int(readBigEndian(at: base + 8))
        let payloadLength = Int(readBigEndian(at: base + 12))
        guard headerLength <= maxHeaderLength else { throw LiveLinkError.headerTooLarge(headerLength) }
        guard payloadLength <= maxPayloadLength else { throw LiveLinkError.payloadTooLarge(payloadLength) }
        let total = LiveLinkProtocol.prefixLength + headerLength + payloadLength
        guard available >= total else { return nil }
        let h0 = base + LiveLinkProtocol.prefixLength
        let header = buffer.subdata(in: h0 ..< h0 + headerLength)
        let payload = buffer.subdata(in: h0 + headerLength ..< h0 + headerLength + payloadLength)
        offset += total
        return LiveMessage(rawType: rawType, header: header, payload: payload)
    }

    private func readBigEndian(at i: Int) -> UInt32 {
        (UInt32(buffer[i]) << 24) | (UInt32(buffer[i + 1]) << 16) | (UInt32(buffer[i + 2]) << 8) | UInt32(buffer[i + 3])
    }
}
