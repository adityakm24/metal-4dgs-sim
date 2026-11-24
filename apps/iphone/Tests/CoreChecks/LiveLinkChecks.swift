// Host-side checks for the live-link wire format (docs/formats.md §6) and the send-queue policy in
// M4DGSCapture/Core/LiveLinkProtocol.swift + LiveOutbox.swift.
//
// Also writes <outDir>/live/stream.bin (one message of every type, as the phone would put them on the wire) plus the
// files they must be byte-identical to, so ../live_link_decode_check.py can re-parse the stream independently.

import Foundation

@MainActor
func runLiveLinkChecks(outDir: URL) throws {
    let live = outDir.appendingPathComponent("live", isDirectory: true)
    try FileManager.default.createDirectory(at: live, withIntermediateDirectories: true)

    // Deterministic pseudo-binary blobs (contain 0x00, "M4DG", and every byte value).
    func blob(_ n: Int, seed: UInt8) -> Data {
        var d = Data((0..<n).map { UInt8(truncatingIfNeeded: $0 &* 31 &+ Int(seed)) })
        if n >= 8 { d.replaceSubrange(2..<6, with: Array("M4DG".utf8)) }
        return d
    }
    let jpeg0 = blob(70_001, seed: 7), depth0 = blob(1_234, seed: 3), conf0 = blob(321, seed: 9)
    let jpeg1 = blob(65_536, seed: 1)

    let pose: [[Double]] = [[1, 0, 0, 0.25], [0, 1, 0, -0.5], [0, 0, 1, 1.5], [0, 0, 0, 1]]
    let k: [[Double]] = [[1450, 0, 960], [0, 1450, 720], [0, 0, 1]]
    let f0 = FrameRecord(index: 0, timestamp: 0, image: "images/000000.jpg", depth: "depth/000000.png",
                         confidence: "confidence/000000.png", cameraToWorld: pose, intrinsics: k,
                         trackingState: "normal", exposureDuration: 0.008, motionBlurScore: 0.1)
    let f1 = FrameRecord(index: 1, timestamp: 0.123456789, image: "images/000001.jpg", depth: nil, confidence: nil,
                         cameraToWorld: pose, intrinsics: k, trackingState: "normal", exposureDuration: 0.01,
                         motionBlurScore: nil)
    let hello = LiveHello(captureName: "capture_20261124_101500", deviceModel: "iPhone17,3", systemVersion: "18.2",
                          imageWidth: 1920, imageHeight: 1440, depthWidth: nil, depthHeight: nil, hasLidar: false)
    let manifest = try CaptureManifest(deviceModel: "iPhone17,3", systemVersion: "18.2", createdAt: "2026-11-24T10:15:00Z",
                                       imageWidth: 1920, imageHeight: 1440, depthWidth: 256, depthHeight: 192,
                                       hasLidar: true, frames: [f0, f1]).jsonData()
    let ply = Data(PointCloudPLY.ascii([SIMD3<Float>(1, 2, 3), SIMD3<Float>(-0.5, 0.25, 1e-6)]).utf8)
    let points = LivePoints(count: 2, ply: ply)
    let status = LiveStatus(framesReceived: 2, bytesReceived: 5_000_000_123)
    let saved = LiveSaved(path: "/Users/me/M4DGS Captures/capture_20261124_101500", frames: 2)

    // MARK: prefix layout
    do {
        let m = try LiveLinkCodec.hello(hello)
        check(Array(m.prefix(4)) == Array("M4DG".utf8), "magic")
        check(m[4] == 1 && m[5] == 0 && m[6] == 0 && m[7] == 0, "type/reserved bytes: \(Array(m[4..<8]))")
        let hl = (Int(m[8]) << 24) | (Int(m[9]) << 16) | (Int(m[10]) << 8) | Int(m[11])
        let pl = (Int(m[12]) << 24) | (Int(m[13]) << 16) | (Int(m[14]) << 8) | Int(m[15])
        check(hl == m.count - 16 && pl == 0, "hello lengths hl=\(hl) pl=\(pl) total=\(m.count)")
        let p = try LiveLinkCodec.prefix(rawType: 2, headerLength: 0x01020304, payloadLength: 0xA0B0C0D0)
        check(Array(p) == [0x4D, 0x34, 0x44, 0x47, 2, 0, 0, 0, 1, 2, 3, 4, 0xA0, 0xB0, 0xC0, 0xD0], "big-endian prefix: \(Array(p))")
    }

    // MARK: header JSON keys
    do {
        let h = try JSONSerialization.jsonObject(with: LiveLinkCodec.jsonHeader(hello)) as! [String: Any]
        check(Set(h.keys) == ["protocol", "capture_name", "device_model", "system_version", "image_width", "image_height",
                               "depth_width", "depth_height", "has_lidar"], "hello keys: \(h.keys.sorted())")
        check(h["protocol"] as? Int == 1 && h["depth_width"] is NSNull, "hello values")

        let fh = LiveFrameHeader(frame: f0, imageBytes: jpeg0.count, depthBytes: depth0.count, confidenceBytes: conf0.count)
        let fo = try JSONSerialization.jsonObject(with: LiveLinkCodec.jsonHeader(fh)) as! [String: Any]
        let frameKeys: Set<String> = ["index", "timestamp", "image", "depth", "confidence", "camera_to_world", "intrinsics",
                                      "tracking_state", "exposure_duration", "motion_blur_score"]
        check(Set(fo.keys) == frameKeys.union(["image_bytes", "depth_bytes", "confidence_bytes"]), "frame keys: \(fo.keys.sorted())")
        check(fo["image_bytes"] as? Int == jpeg0.count && fo["depth_bytes"] as? Int == depth0.count, "frame byte counts")
        // The frame object must match the one in capture.json exactly.
        let mo = try JSONSerialization.jsonObject(with: manifest) as! [String: Any]
        let m0 = (mo["frames"] as! [[String: Any]])[0]
        var stripped = fo
        for key in ["image_bytes", "depth_bytes", "confidence_bytes"] { stripped.removeValue(forKey: key) }
        check(NSDictionary(dictionary: stripped).isEqual(to: m0), "frame header ≠ capture.json frame object")

        let f1h = try JSONSerialization.jsonObject(with: LiveLinkCodec.jsonHeader(
            LiveFrameHeader(frame: f1, imageBytes: 1, depthBytes: 0, confidenceBytes: 0))) as! [String: Any]
        check(f1h["depth"] is NSNull && f1h["confidence"] is NSNull && f1h["depth_bytes"] as? Int == 0, "absent depth → null + 0 bytes")
        let fhText = String(decoding: try LiveLinkCodec.jsonHeader(fh), as: UTF8.self)
        check(!fhText.contains("\\/"), "slashes escaped in frame header")
    }

    // MARK: encoder misuse is rejected
    do {
        var threw = false
        do { _ = try LiveLinkCodec.frame(f0, blobs: LiveFrameBlobs(image: jpeg0, depth: nil, confidence: conf0)) } catch { threw = true }
        check(threw, "frame with depth path but no depth bytes should throw")
        threw = false
        do { _ = try LiveLinkCodec.frame(f1, blobs: LiveFrameBlobs(image: Data(), depth: nil, confidence: nil)) } catch { threw = true }
        check(threw, "frame without image bytes should throw")
    }

    // MARK: round trip of every message type through the streaming decoder, fed in awkward chunk sizes
    let wire: [Data] = [
        try LiveLinkCodec.hello(hello),
        try LiveLinkCodec.frame(f0, blobs: LiveFrameBlobs(image: jpeg0, depth: depth0, confidence: conf0)),
        try LiveLinkCodec.frame(f1, blobs: LiveFrameBlobs(image: jpeg1, depth: nil, confidence: nil)),
        try LiveLinkCodec.points(points),
        try LiveLinkCodec.end(manifest: manifest),
        try LiveLinkCodec.status(status),
        try LiveLinkCodec.saved(saved),
        try LiveLinkCodec.cancel(),
    ]
    let stream = wire.reduce(Data(), +)
    try stream.write(to: live.appendingPathComponent("stream.bin"))
    try jpeg0.write(to: live.appendingPathComponent("frame0.jpg"))
    try depth0.write(to: live.appendingPathComponent("frame0_depth.png"))
    try conf0.write(to: live.appendingPathComponent("frame0_confidence.png"))
    try jpeg1.write(to: live.appendingPathComponent("frame1.jpg"))
    try ply.write(to: live.appendingPathComponent("points.ply"))
    try manifest.write(to: live.appendingPathComponent("capture.json"))

    for chunk in [1, 7, 16, 17, 4096, stream.count] {
        var dec = LiveMessageDecoder(maxHeaderLength: 1 << 24, maxPayloadLength: 1 << 24)
        var msgs: [LiveMessage] = []
        var i = 0
        while i < stream.count {
            dec.append(stream.subdata(in: i ..< min(stream.count, i + chunk)))
            i += chunk
            while let m = try dec.next() { msgs.append(m) }
        }
        check(dec.bufferedByteCount == 0, "chunk \(chunk): \(dec.bufferedByteCount) bytes left over")
        check(msgs.map(\.type) == [.hello, .frame, .frame, .points, .end, .status, .saved, .cancel],
              "chunk \(chunk): types \(msgs.map(\.rawType))")
        guard msgs.count == 8 else { continue }
        check((try? msgs[0].decodeHeader(LiveHello.self)) == hello, "hello round trip")
        if let (h, b) = try? LiveLinkCodec.splitFrame(msgs[1]) {
            check(h.frame == f0 && b == LiveFrameBlobs(image: jpeg0, depth: depth0, confidence: conf0), "frame 0 round trip")
        } else { check(false, "frame 0 split failed") }
        if let (h, b) = try? LiveLinkCodec.splitFrame(msgs[2]) {
            check(h.frame == f1 && b == LiveFrameBlobs(image: jpeg1, depth: nil, confidence: nil) && h.depthBytes == 0,
                  "frame 1 round trip")
        } else { check(false, "frame 1 split failed") }
        check((try? msgs[3].decodeHeader(LivePointsHeader.self))?.count == 2 && msgs[3].payload == ply, "points round trip")
        check(msgs[4].header == manifest && msgs[4].payload.isEmpty, "end header must be capture.json verbatim")
        check((try? JSONDecoder().decode(CaptureManifest.self, from: msgs[4].header))?.frames == [f0, f1], "end decodes as capture.json")
        check((try? msgs[5].decodeHeader(LiveStatus.self)) == status, "status round trip")
        check((try? msgs[6].decodeHeader(LiveSaved.self)) == saved, "saved round trip")
        check(msgs[7].header == Data("{}".utf8) && msgs[7].payload.isEmpty, "cancel round trip")
        for (m, w) in zip(msgs, wire) { check((try? m.encoded()) == w, "re-encode \(m.rawType) differs") }
    }

    // MARK: decoder errors / limits / unknown types
    do {
        var dec = LiveMessageDecoder()
        dec.append(Data("M4DX".utf8) + Data(repeating: 0, count: 12))
        var err: LiveLinkError?
        do { _ = try dec.next() } catch let e as LiveLinkError { err = e }
        check(err == .badMagic(Array("M4DX".utf8)), "bad magic: \(String(describing: err))")

        var small = LiveMessageDecoder(maxHeaderLength: 10, maxPayloadLength: 10)
        small.append(try LiveLinkCodec.status(status))
        err = nil
        do { _ = try small.next() } catch let e as LiveLinkError { err = e }
        if case .headerTooLarge = err {} else { check(false, "header limit not enforced: \(String(describing: err))") }

        var partial = LiveMessageDecoder()
        let m = try LiveLinkCodec.saved(saved)
        partial.append(m.prefix(15))
        check((try? partial.next()) == .some(nil), "15 bytes should not yield a message")
        partial.append(m.dropFirst(15).prefix(m.count - 16))
        check((try? partial.next()) == .some(nil), "all but one byte should not yield a message")
        partial.append(m.suffix(1))
        check((try? partial.next())??.type == .saved, "last byte completes the message")

        var unknown = LiveMessageDecoder()
        unknown.append(try LiveLinkCodec.message(rawType: 42, header: Data("{}".utf8), payloadParts: [Data([1, 2, 3])]))
        let u = try unknown.next()
        check(u?.rawType == 42 && u?.type == nil && u?.payload == Data([1, 2, 3]), "unknown type is surfaced, not fatal")
    }

    // MARK: outbox ordering, memory bound, spill-to-disk, rewind, cancel
    do {
        let b = { (n: Int) in LiveFrameBlobs(image: Data(count: n), depth: nil, confidence: nil) }
        func rec(_ i: Int) -> FrameRecord {
            FrameRecord(index: i, timestamp: Double(i), image: CaptureFormat.imagePath(i), depth: nil, confidence: nil,
                        cameraToWorld: pose, intrinsics: k, trackingState: "normal", exposureDuration: 0, motionBlurScore: nil)
        }
        var o = LiveOutbox(memoryLimit: 250)
        check(o.takeNext() == .hello, "hello first")
        check(o.addFrame(rec(0), blobs: b(100)), "frame 0 cached")
        check(o.addFrame(rec(1), blobs: b(100)), "frame 1 cached")
        check(!o.addFrame(rec(2), blobs: b(100)), "frame 2 must spill (limit 250)")
        check(o.cacheBytes == 200 && o.spilledFrames == 1 && o.framesPending == 3, "cache accounting \(o.cacheBytes) \(o.spilledFrames)")
        check(o.takeNext() == .frame(rec(0), b(100)), "frame 0 from memory")
        check(o.cacheBytes == 100, "cache shrinks after take")
        check(o.addFrame(rec(3), blobs: b(100)), "frame 3 fits again")
        check(o.takeNext() == .frame(rec(1), b(100)), "frame 1 from memory")
        check(o.takeNext() == .frame(rec(2), nil), "frame 2 from disk, still in order")
        check(o.takeNext() == .frame(rec(3), b(100)), "frame 3")
        check(o.takeNext() == nil && !o.isDrained, "nothing more until stop")
        o.finish(points: points, manifest: manifest)
        check(!o.addFrame(rec(4), blobs: b(1)), "frames after finish are refused")
        check(o.takeNext() == .points(points), "points after frames")
        check(!o.isDrained, "end still owed")
        check(o.takeNext() == .end(manifest) && o.isDrained && o.endTaken, "end last")
        check(o.takeNext() == nil, "drained")
        o.rewind()
        check(!o.isDrained && o.takeNext() == .hello, "rewind replays from hello")
        var replay: [LiveOutbox.Step] = []
        while let s = o.takeNext() { replay.append(s) }
        check(replay == [.frame(rec(0), nil), .frame(rec(1), nil), .frame(rec(2), nil), .frame(rec(3), nil),
                         .points(points), .end(manifest)], "replay re-reads every frame from disk, in order")

        var noPoints = LiveOutbox(memoryLimit: 1 << 20)
        _ = noPoints.takeNext()
        noPoints.finish(points: nil, manifest: manifest)
        check(noPoints.takeNext() == .end(manifest) && noPoints.isDrained, "no points → end directly")

        var c1 = LiveOutbox(memoryLimit: 1 << 20)
        c1.addFrame(rec(0), blobs: b(10))
        c1.cancel()
        check(c1.takeNext() == nil && c1.isDrained, "cancel before hello sends nothing")
        var c2 = LiveOutbox(memoryLimit: 1 << 20)
        _ = c2.takeNext()
        c2.addFrame(rec(0), blobs: b(10))
        c2.cancel()
        check(c2.cacheBytes == 0 && c2.framesPending == 0, "cancel drops queued frames")
        check(c2.takeNext() == .cancel && c2.takeNext() == nil && c2.isDrained && c2.cancelTaken, "cancel after hello")
    }

    // MARK: HUD text
    do {
        var s = LiveStreamSnapshot(macName: "Studio Mac")
        check(s.hudText.contains("connecting to Studio Mac"), s.hudText)
        s.phase = .streaming; s.framesQueued = 40; s.framesSent = 12
        check(s.isLagging && s.hudText.contains("sending 12/40") && s.hudText.contains("lagging, 28 behind"), s.hudText)
        s.phase = .saved(path: "/x/capture_1", frames: 40); s.framesSent = 40
        check(!s.isLagging && s.hudText == "Saved on Mac at /x/capture_1 (40 frames)" && s.severity == .good, s.hudText)
        s.phase = .failed("timed out"); s.macFrames = 30
        check(s.hudText.contains("Mac has 30/40") && s.severity == .error, s.hudText)
    }
}
