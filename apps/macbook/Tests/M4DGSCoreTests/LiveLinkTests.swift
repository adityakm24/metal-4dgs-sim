import XCTest
@testable import M4DGSCore

final class LiveLinkCodecTests: XCTestCase {
    func testWireLayoutIsBigEndian() throws {
        let m = LiveMessage(.frame, header: Data("{}".utf8), payload: Data([9, 8, 7]))
        let d = m.encoded()
        XCTAssertEqual(Array(d.prefix(4)), Array("M4DG".utf8))
        XCTAssertEqual(d[4], 2)
        XCTAssertEqual(Array(d[5..<8]), [0, 0, 0])
        XCTAssertEqual(Array(d[8..<12]), [0, 0, 0, 2])      // header_len = 2, big-endian
        XCTAssertEqual(Array(d[12..<16]), [0, 0, 0, 3])     // payload_len = 3
        XCTAssertEqual(Array(d[16...]), Array("{}".utf8) + [9, 8, 7])
    }

    func testRoundTrip() throws {
        let payload = Data((0..<70_000).map { UInt8($0 % 251) })
        let msgs = [
            try LiveMessage(.hello, json: ["protocol": 1, "capture_name": "capture_20261124_101500", "has_lidar": true]),
            try LiveMessage(.frame, json: ["index": 0, "image": "images/000000.jpg", "image_bytes": payload.count,
                                           "depth_bytes": 0, "confidence_bytes": 0], payload: payload),
            try LiveMessage(.status, json: ["frames_received": 1, "bytes_received": 70_123]),
            LiveMessage(.points, header: Data(#"{"count":0}"#.utf8), payload: Data("ply\n".utf8)),
            try LiveMessage(.end, json: ["format": "m4dgs-arkit-capture", "frames": [] as [Any]]),
            try LiveMessage(.cancel, json: [String: Any]()),
            try LiveMessage(.saved, json: ["path": "/tmp/x", "frames": 1]),
        ]
        var dec = LiveMessageDecoder()
        dec.append(msgs.reduce(Data()) { $0 + $1.encoded() })
        let out = try dec.drain()
        XCTAssertEqual(out, msgs)
        XCTAssertEqual(out.map(\.type), [.hello, .frame, .status, .points, .end, .cancel, .saved])
        XCTAssertEqual(try out[0].headerObject()["capture_name"] as? String, "capture_20261124_101500")
        XCTAssertEqual(dec.bufferedBytes, 0)
    }

    func testSplitPacketReassembly() throws {
        let payload = Data((0..<5000).map { UInt8(truncatingIfNeeded: $0 &* 31) })
        let msgs = (0..<4).map { i in
            LiveMessage(.frame, header: Data(#"{"index":\#(i)}"#.utf8), payload: payload.prefix(1000 * (i + 1)))
        }
        let wire = msgs.reduce(Data()) { $0 + $1.encoded() }
        // byte at a time
        var dec = LiveMessageDecoder()
        var got: [LiveMessage] = []
        for b in wire {
            dec.append(Data([b]))
            while let m = try dec.next() { got.append(m) }
        }
        XCTAssertEqual(got, msgs)
        // random chunking, including splits inside the 16-byte prefix and coalesced messages
        var rng = SystemRandomNumberGenerator()
        for _ in 0..<50 {
            var d2 = LiveMessageDecoder()
            var got2: [LiveMessage] = []
            var i = 0
            while i < wire.count {
                let n = min(Int.random(in: 1...3000, using: &rng), wire.count - i)
                d2.append(wire.subdata(in: i..<(i + n)))
                got2 += try d2.drain()
                i += n
            }
            XCTAssertEqual(got2, msgs)
        }
        // an incomplete message yields nothing yet
        var d3 = LiveMessageDecoder()
        d3.append(msgs[0].encoded().prefix(20))
        XCTAssertNil(try d3.next())
    }

    func testRejectsBadMagicAndOversize() {
        var dec = LiveMessageDecoder()
        dec.append(Data("GET / HTTP/1.1\r\n".utf8))
        XCTAssertThrowsError(try dec.next())
        var early = LiveMessageDecoder()
        early.append(Data("M5".utf8))                       // detected before the prefix is complete
        XCTAssertThrowsError(try early.next())

        var lim = LiveMessageDecoder(limits: LiveLinkLimits(maxHeaderBytes: 100, maxPayloadBytes: 1000))
        var big = LiveMessage(.frame, header: Data(repeating: 32, count: 10)).encoded()
        big.replaceSubrange(12..<16, with: [0xff, 0xff, 0xff, 0xff])   // payload_len = 4 GiB
        lim.append(big)
        XCTAssertThrowsError(try lim.next()) { XCTAssertEqual($0 as? LiveLinkError, .payloadTooLarge(0xffff_ffff, limit: 1000)) }
        var lim2 = LiveMessageDecoder(limits: LiveLinkLimits(maxHeaderBytes: 100, maxPayloadBytes: 1000))
        lim2.append(LiveMessage(.end, header: Data(repeating: 32, count: 101)).encoded())
        XCTAssertThrowsError(try lim2.next())
    }

    func testMaliciousPathsRejected() {
        for bad in ["../images/x.jpg", "/etc/passwd", "images/../../x.jpg", "images/../x.jpg", "images/.hidden",
                    "images/a/b.jpg", "images/", "images", "capture.json", "points.ply", "other/x.jpg",
                    "images\\..\\x.jpg", "images/x\0.jpg", "images/x .jpg", "images//x.jpg", "./images/x.jpg",
                    "images/\u{2215}x.jpg", "images/" + String(repeating: "a", count: 200)] {
            XCTAssertNil(LivePaths.sanitizeFramePath(bad), bad.debugDescription)
        }
        XCTAssertEqual(LivePaths.sanitizeFramePath("images/000000.jpg"), "images/000000.jpg")
        XCTAssertEqual(LivePaths.sanitizeFramePath("depth/000123.png"), "depth/000123.png")
        XCTAssertEqual(LivePaths.sanitizeFramePath("confidence/000123.png"), "confidence/000123.png")

        for bad in ["", "..", ".", "../x", "a/b", "/abs", ".hidden", "a\\b", "x:y", "a\nb", "___", String(repeating: "a", count: 200)] {
            XCTAssertNil(LivePaths.sanitizeCaptureName(bad), bad.debugDescription)
        }
        XCTAssertEqual(LivePaths.sanitizeCaptureName("capture_20261124_101500"), "capture_20261124_101500")
        XCTAssertEqual(LivePaths.sanitizeCaptureName("my capture é"), "my_capture__")
    }

    func testTrainLogParsing() {
        let p = TrainProgress.parse("[train] it  4100  loss 0.0123  psnr 45.70  gaussians 123456  12 ms/it")
        XCTAssertEqual(p, TrainProgress(iteration: 4100, loss: 0.0123, psnr: 45.70, gaussians: 123456))
        XCTAssertEqual(TrainProgress.parse("[train] it     1  loss 0.5  psnr 9.1")?.iteration, 1)
        XCTAssertNil(TrainProgress.parse("[train] 40 train / 5 test frames"))
        XCTAssertNil(TrainProgress.parse("[train] done in 12 s"))
    }
}

final class LiveCaptureSessionTests: XCTestCase {
    var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("m4dgs-live-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }
    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: root) }

    func hello(_ name: String = "capture_20261124_101500") throws -> LiveMessage {
        try LiveMessage(.hello, json: ["protocol": 1, "capture_name": name, "device_model": "iPhone17,3", "system_version": "18.2",
                                       "image_width": 4, "image_height": 3, "depth_width": 2, "depth_height": 2, "has_lidar": true])
    }

    func frame(_ i: Int, image: String? = nil, jpeg: Data = Data([0xff, 0xd8, 1, 2, 0xff, 0xd9]), depth: Data = Data([0x89, 0x50, 7])) throws -> LiveMessage {
        let name = String(format: "%06d", i)
        let h: [String: Any] = [
            "index": i, "timestamp": Double(i) * 0.1,
            "image": image ?? "images/\(name).jpg", "depth": "depth/\(name).png", "confidence": NSNull(),
            "camera_to_world": [[1, 0, 0, Double(i)], [0, 1, 0, 0], [0, 0, 1, -Double(i)], [0, 0, 0, 1]],
            "intrinsics": [[3, 0, 2], [0, 3, 1.5], [0, 0, 1]], "tracking_state": "normal",
            "exposure_duration": 0.008, "image_bytes": jpeg.count, "depth_bytes": depth.count, "confidence_bytes": 0,
        ]
        return try LiveMessage(.frame, json: h, payload: jpeg + depth)
    }

    func testFullCapture() throws {
        let s = LiveCaptureSession(capturesFolder: root)
        _ = try s.handle(try hello())
        let dir = try XCTUnwrap(s.directory)
        XCTAssertEqual(dir.lastPathComponent, "capture_20261124_101500")
        var status: [LiveMessage] = []
        for i in 0..<3 {
            let (r, ev) = try s.handle(try frame(i))
            status += r
            guard case .frame(let info) = ev.first else { return XCTFail() }
            XCTAssertEqual(info.cameraToWorld?[0][3], Double(i))
            XCTAssertEqual(info.framesReceived, i + 1)
        }
        XCTAssertEqual(status.map(\.type), [.status, .status, .status])
        let last = try status[2].headerObject()
        XCTAssertEqual(last["frames_received"] as? Int, 3)
        XCTAssertEqual(last["bytes_received"] as? Int, s.bytesReceived)
        XCTAssertEqual(try Data(contentsOf: dir.appendingPathComponent("images/000001.jpg")), Data([0xff, 0xd8, 1, 2, 0xff, 0xd9]))
        XCTAssertEqual(try Data(contentsOf: dir.appendingPathComponent("depth/000002.png")), Data([0x89, 0x50, 7]))

        let ply = Data("ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n1 2 3\n".utf8)
        _ = try s.handle(LiveMessage(.points, header: Data(#"{"count":1}"#.utf8), payload: ply))
        XCTAssertFalse(FileManager.default.fileExists(atPath: dir.appendingPathComponent("points.ply").path))
        let capJSON = Data(#"{"format":"m4dgs-arkit-capture","version":1,"frames":[{"index":0,"image":"images/000000.jpg","depth":"depth/000000.png"}]}"#.utf8)
        let (r, ev) = try s.handle(LiveMessage(.end, header: capJSON))
        XCTAssertEqual(r.first?.type, .saved)
        XCTAssertEqual(try r[0].headerObject()["path"] as? String, dir.path)
        XCTAssertEqual(try r[0].headerObject()["frames"] as? Int, 3)
        guard case .saved = ev.first else { return XCTFail() }
        XCTAssertEqual(try Data(contentsOf: dir.appendingPathComponent("capture.json")), capJSON)   // verbatim
        XCTAssertEqual(try Data(contentsOf: dir.appendingPathComponent("points.ply")), ply)
        XCTAssertFalse(FileManager.default.fileExists(atPath: dir.appendingPathComponent(LiveCaptureSession.partialFileName).path))
        XCTAssertEqual(s.state, .saved)
        XCTAssertNil(s.connectionClosed())
    }

    func testDroppedConnectionLeavesPartial() throws {
        let s = LiveCaptureSession(capturesFolder: root)
        _ = try s.handle(try hello())
        _ = try s.handle(try frame(0))
        _ = try s.handle(try frame(1))
        let dir = try XCTUnwrap(s.connectionClosed())
        let doc = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: dir.appendingPathComponent("capture.partial.json"))) as? [String: Any])
        XCTAssertEqual((doc["frames"] as? [Any])?.count, 2)
        XCTAssertEqual((doc["hello"] as? [String: Any])?["capture_name"] as? String, "capture_20261124_101500")
        XCTAssertNil(((doc["frames"] as? [[String: Any]])?.first)?["image_bytes"])
        XCTAssertFalse(FileManager.default.fileExists(atPath: dir.appendingPathComponent("capture.json").path))
    }

    func testCancelDeletesDirectory() throws {
        let s = LiveCaptureSession(capturesFolder: root)
        _ = try s.handle(try hello())
        _ = try s.handle(try frame(0))
        let dir = try XCTUnwrap(s.directory)
        XCTAssertTrue(FileManager.default.fileExists(atPath: dir.path))
        _ = try s.handle(try LiveMessage(.cancel, json: [String: Any]()))
        XCTAssertFalse(FileManager.default.fileExists(atPath: dir.path))
        XCTAssertEqual(s.state, .cancelled)
        XCTAssertNil(s.connectionClosed())
    }

    func testRejectsMaliciousInput() throws {
        // traversal in capture_name
        XCTAssertThrowsError(try LiveCaptureSession(capturesFolder: root).handle(try hello("../../evil")))
        // traversal in a frame path: nothing written outside, partial left behind
        let s = LiveCaptureSession(capturesFolder: root)
        _ = try s.handle(try hello())
        XCTAssertThrowsError(try s.handle(try frame(0, image: "../../../evil.jpg")))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.deletingLastPathComponent().appendingPathComponent("evil.jpg").path))
        XCTAssertEqual(s.state, .failed)
        XCTAssertNotNil(s.connectionClosed())
        // byte counts that do not match the payload
        let s2 = LiveCaptureSession(capturesFolder: root)
        _ = try s2.handle(try hello("cap2"))
        var m = try frame(0)
        m.payload.append(0)
        XCTAssertThrowsError(try s2.handle(m))
        // frame before hello
        XCTAssertThrowsError(try LiveCaptureSession(capturesFolder: root).handle(try frame(0)))
        // capture.json pointing outside the capture
        let s3 = LiveCaptureSession(capturesFolder: root)
        _ = try s3.handle(try hello("cap3"))
        XCTAssertThrowsError(try s3.handle(LiveMessage(.end, header: Data(#"{"format":"m4dgs-arkit-capture","frames":[{"image":"../x.jpg"}]}"#.utf8))))
        // unsupported protocol
        XCTAssertThrowsError(try LiveCaptureSession(capturesFolder: root).handle(
            LiveMessage(.hello, json: ["protocol": 2, "capture_name": "x"])))
    }

    func testDuplicateNameGetsSuffix() throws {
        let a = LiveCaptureSession(capturesFolder: root), b = LiveCaptureSession(capturesFolder: root)
        _ = try a.handle(try hello())
        _ = try b.handle(try hello())
        XCTAssertEqual(b.directory?.lastPathComponent, "capture_20261124_101500_1")
    }
}
