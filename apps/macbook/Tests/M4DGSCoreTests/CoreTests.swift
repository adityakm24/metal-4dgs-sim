import XCTest
import simd
@testable import M4DGSCore

final class SafeTensorsTests: XCTestCase {
    func testRoundTrip() throws {
        let a = SafeTensor(name: "background.means", shape: [2, 3], values: [1, 2, 3, 4, 5, -6.5])
        let b = SafeTensor(name: "background.opacity_logits", shape: [2, 1], values: [0.25, -3])
        let sg = #"{"version":1,"sh_degree":0,"nodes":[{"name":"background","kind":"background"}]}"#
        let data = try SafeTensorsWriter.encode([a, b], metadata: ["format": "m4dgs-scene", "version": "1", "scene_graph": sg])
        // header length is little-endian and the header is 8-byte aligned
        let n = data.prefix(8).enumerated().reduce(UInt64(0)) { $0 | (UInt64($1.element) << (8 * UInt64($1.offset))) }
        XCTAssertEqual(n % 8, 0)
        let f = try SafeTensorsFile(data: data)
        XCTAssertEqual(f.info("background.means")?.shape, [2, 3])
        XCTAssertEqual(try f.floats("background.means"), a.values)
        XCTAssertEqual(try f.floats("background.opacity_logits"), b.values)
        XCTAssertEqual(f.metadata["format"], "m4dgs-scene")
        XCTAssertEqual(f.metadata["scene_graph"], sg)
        XCTAssertThrowsError(try f.floats("nope"))
    }

    func testRejectsTruncated() {
        XCTAssertThrowsError(try SafeTensorsFile(data: Data([1, 2, 3])))
        var d = Data(count: 8); d[0] = 200   // header length beyond EOF
        XCTAssertThrowsError(try SafeTensorsFile(data: d))
    }
}

final class MathTests: XCTestCase {
    func assertQuat(_ a: Quat, _ b: Quat, accuracy: Double = 1e-9, file: StaticString = #filePath, line: UInt = #line) {
        // q and −q are the same rotation
        XCTAssertEqual(abs(a.normalized.dot(b.normalized)), 1, accuracy: accuracy, file: file, line: line)
    }

    func testSlerpEndpointsAndMidpoint() {
        let a = Quat.identity, b = Quat.yaw(.pi / 2)
        assertQuat(Quat.slerp(a, b, 0), a)
        assertQuat(Quat.slerp(a, b, 1), b)
        assertQuat(Quat.slerp(a, b, 0.5), Quat.yaw(.pi / 4))
        XCTAssertEqual(Quat.slerp(a, b, 0.25).yawAngle, .pi / 8, accuracy: 1e-9)
    }

    func testSlerpShortestPathAndUnnormalizedInputs() {
        let a = Quat.yaw(0.1)
        let b = -Quat.yaw(0.3)                       // same rotation as yaw(0.3), opposite hemisphere
        XCTAssertEqual(Quat.slerp(a, b, 0.5).yawAngle, 0.2, accuracy: 1e-9)
        let s = Quat(w: 2, x: 0, y: 0, z: 0)         // not normalized on disk
        let t = Quat(w: 0, x: 0, y: 0, z: 5)         // 180° yaw, scaled
        let m = Quat.slerp(s, t, 0.5)
        XCTAssertEqual(m.norm, 1, accuracy: 1e-12)
        XCTAssertEqual(abs(m.yawAngle), .pi / 2, accuracy: 1e-9)
    }

    func testQuatMatrixRoundTrip() {
        let q = (Quat(axis: [1, 2, 3], angle: 0.7) * Quat.yaw(1.1)).normalized
        assertQuat(Quat(matrix: q.matrix), q)
        let v = SIMD3<Double>(0.3, -1, 2)
        XCTAssertLessThan(simd_length(q.rotate(v) - (q * Quat(w: 0, x: v.x, y: v.y, z: v.z) * q.conjugate).vector), 1e-12)
    }

    func testTrackInterpolationAndClamping() {
        let keys = [TrackKey(t: 1, pos: [0, 0, 0], quat: .identity),
                    TrackKey(t: 3, pos: [10, 2, 0], quat: .yaw(.pi / 2))]
        // clamped (constant) outside the track's time range
        XCTAssertEqual(Track.pose(keys, at: -5).pos, [0, 0, 0])
        XCTAssertEqual(Track.pose(keys, at: 1).pos, [0, 0, 0])
        XCTAssertEqual(Track.pose(keys, at: 99).pos, [10, 2, 0])
        XCTAssertEqual(Track.pose(keys, at: 99).quat.yawAngle, .pi / 2, accuracy: 1e-12)
        // linear pos, slerp quat
        let mid = Track.pose(keys, at: 2)
        XCTAssertLessThan(simd_length(mid.pos - [5, 1, 0]), 1e-12)
        XCTAssertEqual(mid.quat.yawAngle, .pi / 4, accuracy: 1e-9)
        XCTAssertEqual(Track.pose([], at: 1), .identity)
        XCTAssertEqual(Track.pose([keys[1]], at: 0).pos, [10, 2, 0])
    }

    func testSphericalHarmonics() {
        let sh0: [SIMD3<Float>] = [[1, 0, -3]]
        let c = SH.color(degree: 0, sh: sh0, dir: [0, 0, 1])
        XCTAssertEqual(c.x, 0.5 + SH.C0, accuracy: 1e-6)
        XCTAssertEqual(c.y, 0.5, accuracy: 1e-6)
        XCTAssertEqual(c.z, 0, accuracy: 1e-6)            // clamped ≥ 0
        // degree 1: −C1·y·sh1 + C1·z·sh2 − C1·x·sh3
        let sh1: [SIMD3<Float>] = [.zero, [1, 1, 1], [2, 2, 2], [3, 3, 3]]
        let r = SH.evaluate(degree: 1, sh: sh1, dir: [1, 0, 0])
        XCTAssertEqual(r.x, -3 * SH.C1, accuracy: 1e-6)
        let r2 = SH.evaluate(degree: 1, sh: sh1, dir: [0, 1, 0])
        XCTAssertEqual(r2.x, -1 * SH.C1, accuracy: 1e-6)
        // higher-degree terms vanish for the DC-only case
        var sh3 = [SIMD3<Float>](repeating: .zero, count: 16); sh3[0] = [0.4, 0.4, 0.4]
        XCTAssertEqual(SH.evaluate(degree: 3, sh: sh3, dir: simd_normalize([1, 2, 3])).x, 0.4 * SH.C0, accuracy: 1e-6)
    }

    func testGaussian4DEvaluation() {
        let g = Gaussian4D(mean: [1, 0, 0], logScale: [0, 0, 0], quat: .identity, opacityLogit: 0,
                           velocity: [2, 0, 0], tCenter: 1, tLogScale: log(0.5))
        let node = Pose(pos: [10, 0, 0], quat: .yaw(.pi / 2))
        let e = g.evaluate(at: 2, nodePose: node)
        // μ_local = (1,0,0) + (2,0,0)·1 = (3,0,0); rotated 90° about z → (0,3,0); + (10,0,0)
        XCTAssertLessThan(simd_length(e.meanWorld - [10, 3, 0]), 1e-12)
        // α = sigmoid(0) · exp(−½ (1/0.5)²) = 0.5·e^−2
        XCTAssertEqual(e.alpha, 0.5 * exp(-2), accuracy: 1e-12)
        XCTAssertEqual(e.quatWorld.yawAngle, .pi / 2, accuracy: 1e-12)
    }

    func testEgoCameraOrientation() {
        // OpenCV camera mounted 1.5 m up looking along ego +x (cam z→+x, cam x→−y, cam y→−z)
        let c2e = RigidTransform(rowMajor: [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 1.5], [0, 0, 0, 1]])
        let k = Intrinsics(fx: 300, fy: 300, cx: 256, cy: 144, width: 512, height: 288)
        let cam = CameraView.ego(egoPose: Pose(pos: [5, 0, 0]), cameraToEgo: c2e, intrinsics: k)
        let ahead = cam.project([15, 0, 1.5])
        XCTAssertEqual(ahead.x, 256, accuracy: 1e-9); XCTAssertEqual(ahead.y, 144, accuracy: 1e-9); XCTAssertEqual(ahead.z, 10, accuracy: 1e-9)
        XCTAssertLessThan(cam.project([15, 2, 1.5]).x, 256)       // +y (left) → left of image
        XCTAssertLessThan(cam.project([15, 0, 3]).y, 144)         // up → top of image
        XCTAssertGreaterThan(cam.project([15, 0, 0]).y, 144)      // ground → below horizon
        XCTAssertLessThan(cam.project([0, 0, 1.5]).z, 0)          // behind
        // the same rotation via lookAt
        let la = CameraView.lookAt(eye: [0, 0, 0], target: [1, 0, 0])
        XCTAssertLessThan(simd_length(la.R.columns.0 - [0, -1, 0]), 1e-12)
        XCTAssertLessThan(simd_length(la.R.columns.1 - [0, 0, -1]), 1e-12)
    }
}

extension Quat { var vector: SIMD3<Double> { [x, y, z] } }

final class SceneGraphAndEditTests: XCTestCase {
    let json = """
    {"version":1,"sh_degree":1,"time_range":[0,4],"background_color":[0.1,0.2,0.3],"custom_field":{"keep":true},
     "nodes":[{"name":"background","kind":"background"},
              {"name":"actor_2","kind":"actor","category":"vehicle.car","size":[4.5,1.9,1.6],"visible_range":[1,3],
               "track":[{"t":4,"pos":[40,0,0.8],"quat":[1,0,0,0]},{"t":0,"pos":[0,0,0.8],"quat":[1,0,0,0]}]}],
     "ego":{"size":[4.5,1.9,1.6],"camera_to_ego":[[0,0,1,0],[-1,0,0,0],[0,-1,0,1.5],[0,0,0,1]],
            "camera":{"width":512,"height":288,"fx":300,"fy":300,"cx":256,"cy":144},
            "track":[{"t":0,"pos":[0,0,0],"quat":[1,0,0,0]}]}}
    """

    func testParseGraph() throws {
        let d = try SceneGraphDocument(jsonString: json)
        let g = d.graph
        XCTAssertEqual(g.shDegree, 1)
        XCTAssertEqual(g.nodes.map(\.name), ["background", "actor_2"])
        XCTAssertEqual(g.nodes[1].track.map(\.t), [0, 4])               // sorted by time
        XCTAssertEqual(g.backgroundColor, [0.1, 0.2, 0.3])
        XCTAssertEqual(g.nodes[1].visibleRange, 1...3)
        XCTAssertFalse(g.nodes[1].isVisible(at: 0.5)); XCTAssertTrue(g.nodes[1].isVisible(at: 2))
        XCTAssertTrue(g.nodes[0].isVisible(at: 100))
        XCTAssertEqual(g.ego?.intrinsics.fx, 300)
        XCTAssertEqual(g.ego?.cameraToEgo.t, [0, 0, 1.5])
        let f = SceneFrame.evaluate(graph: g, time: 0.5)
        XCTAssertEqual(f.nodeVisible, [true, false])
    }

    func testEditsPreserveUnknownAndOptionalFields() throws {
        let d = try SceneGraphDocument(jsonString: json)
        var e = TrackEdit(); e.lateral = 1
        let edited = TrackEditor.apply(e, to: d.graph.nodes[1].track, timeRange: d.graph.timeRange)
        let out = try d.jsonData(replacingTracks: ["actor_2": edited])
        let back = try SceneGraphDocument(jsonData: out)
        XCTAssertEqual(back.graph.nodes[1].track.count, edited.count)
        XCTAssertEqual(back.graph.nodes[1].track.first!.pos.y, 1, accuracy: 1e-12)
        XCTAssertEqual(back.graph.backgroundColor, [0.1, 0.2, 0.3])
        XCTAssertEqual(back.graph.nodes[1].visibleRange, 1...3)
        XCTAssertEqual((back.raw["custom_field"] as? [String: Bool])?["keep"], true)
        XCTAssertEqual(back.graph.nodes[1].category, "vehicle.car")
        XCTAssertNotNil(back.graph.ego)
    }

    func testIdentityEditIsNoOp() {
        let keys = [TrackKey(t: 0, pos: [0, 0, 0]), TrackKey(t: 4, pos: [40, 0, 0])]
        XCTAssertEqual(TrackEditor.apply(TrackEdit(), to: keys, timeRange: 0...4), keys)
    }

    func testLateralAndLongitudinalUseActorAxes() {
        // actor heading +y (yaw 90°): its left axis is −x, forward is +y
        let q = Quat.yaw(.pi / 2)
        let keys = [TrackKey(t: 0, pos: [0, 0, 0], quat: q), TrackKey(t: 2, pos: [0, 20, 0], quat: q)]
        var e = TrackEdit(); e.lateral = 1.5; e.longitudinal = 2
        let out = TrackEditor.apply(e, to: keys, timeRange: 0...2)
        let p = Track.pose(out, at: 1).pos
        XCTAssertLessThan(simd_length(p - [-1.5, 12, 0]), 1e-9)
    }

    func testTimeShiftAndSpeedScale() {
        let keys = [TrackKey(t: 0, pos: [0, 0, 0]), TrackKey(t: 10, pos: [100, 0, 0])]
        var e = TrackEdit(); e.timeShift = 1
        XCTAssertEqual(Track.pose(TrackEditor.apply(e, to: keys, timeRange: 0...10), at: 3).pos.x, 20, accuracy: 1e-9)
        e = TrackEdit(); e.speedScale = 2
        XCTAssertEqual(Track.pose(TrackEditor.apply(e, to: keys, timeRange: 0...10), at: 3).pos.x, 60, accuracy: 1e-9)
        XCTAssertEqual(Track.pose(TrackEditor.apply(e, to: keys, timeRange: 0...10), at: 8).pos.x, 100, accuracy: 1e-9) // clamped
    }

    func testCutInBlendsSmoothly() {
        let keys = [TrackKey(t: 0, pos: [0, 0, 0]), TrackKey(t: 10, pos: [100, 0, 0])]
        var e = TrackEdit(); e.cutInEnabled = true; e.cutInOffset = -2; e.cutInStart = 2; e.cutInDuration = 2
        let out = TrackEditor.apply(e, to: keys, timeRange: 0...10)
        XCTAssertEqual(Track.pose(out, at: 1).pos.y, 0, accuracy: 1e-9)
        XCTAssertEqual(Track.pose(out, at: 3).pos.y, -1, accuracy: 1e-9)    // halfway (smootherstep(0.5) = 0.5)
        XCTAssertEqual(Track.pose(out, at: 5).pos.y, -2, accuracy: 1e-9)
        // heading turns right (negative yaw) while moving toward −y, straight again afterwards
        XCTAssertLessThan(Track.pose(out, at: 3).quat.yawAngle, -0.01)
        XCTAssertEqual(Track.pose(out, at: 5).quat.yawAngle, 0, accuracy: 1e-6)
    }

    func testPerturbationParsing() {
        let (name, e) = parsePerturbation("actor_2:lateral_offset=-1.8m,t=2.5s")!
        XCTAssertEqual(name, "actor_2")
        XCTAssertTrue(e.cutInEnabled)
        XCTAssertEqual(e.cutInOffset, -1.8); XCTAssertEqual(e.cutInStart, 2.5)
    }

    func testRolloutParsing() throws {
        let r = try Rollout(data: Data("""
        {"format":"m4dgs-rollout","version":1,"scene":"x.safetensors","policy":"pilotnet","dt":0.1,
         "actor_tracks":{"actor_2":[{"t":0,"pos":[1,2,3],"quat":[1,0,0,0]}]},
         "steps":[{"t":0.1,"ego":{"pos":[1,0,0],"quat":[1,0,0,0]},"speed":8,"action":{"steer":0.1,"accel":-1},
                   "latency_ms":{"policy":1.2,"raster":3.1,"refine":18},"events":["collision:actor_2"]},
                  {"t":0.0,"ego":{"pos":[0,0,0],"quat":[1,0,0,0]},"speed":8,"action":{"steer":0,"accel":0},"events":[]}],
         "metrics":{"collision":true,"min_ttc":3.2}}
        """.utf8))
        XCTAssertEqual(r.steps.map(\.t), [0, 0.1])
        XCTAssertEqual(r.actorTracks["actor_2"]?.first?.pos, [1, 2, 3])     // dictionary keys untouched
        XCTAssertTrue(r.steps[1].hasCollision)
        XCTAssertEqual(r.steps[1].latencyMs.map(\.0), ["raster", "refine", "policy"])
        XCTAssertEqual(r.steps[1].totalLatencyMs, 22.3, accuracy: 1e-9)
        XCTAssertEqual(r.stepIndex(at: 0.05), 0); XCTAssertEqual(r.stepIndex(at: 5), 1)
        XCTAssertEqual(r.metrics.first { $0.0 == "collision" }?.1.description, "true")
    }
}
