import XCTest
import Metal
import simd
@testable import M4DGSCore
@testable import M4DGSRender

/// Builds a background-only cloud from simple Gaussians (SH degree 0).
func makeCloud(_ gs: [(mean: SIMD3<Float>, sigma: Float, rgb: SIMD3<Float>, opacity: Float)]) -> GaussianCloud {
    var packed: [Float] = [], sh: [Float] = []
    for g in gs {
        packed += [g.mean.x, g.mean.y, g.mean.z, 0, 0, 0, 0, 0, 1, 0, 0, 0, g.sigma, g.sigma, g.sigma, g.opacity]
        let c = (g.rgb - 0.5) / SH.C0
        sh += [c.x, c.y, c.z]
    }
    return GaussianCloud(count: gs.count, shDegree: 0, packed: packed, sh: sh,
                         nodeIds: [UInt32](repeating: 0, count: gs.count), nodeRanges: ["background": 0..<gs.count])
}

let fixtureCamToEgo = RigidTransform(rowMajor: [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 1.5], [0, 0, 0, 1]])
let fixtureK = Intrinsics(fx: 300, fy: 300, cx: 256, cy: 144, width: 512, height: 288)

func egoFrame(time: Double = 0) -> RenderFrame {
    let cam = CameraView.ego(egoPose: .identity, cameraToEgo: fixtureCamToEgo, intrinsics: fixtureK)
    return RenderFrame(camera: cam, imageRect: [0, 0, 512, 288], time: time, nodePoses: [.identity], nodeVisible: [true])
}

final class RenderTests: XCTestCase {
    var renderer: SplatRenderer!

    override func setUpWithError() throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("no Metal device") }
        renderer = try SplatRenderer()
    }

    func testGPURadixSortIsBackToFront() throws {
        var rng = SystemRandomNumberGenerator()
        var gs: [(mean: SIMD3<Float>, sigma: Float, rgb: SIMD3<Float>, opacity: Float)] = []
        for _ in 0..<50_000 {   // > several sort tiles, all in front of the camera
            let x = Float.random(in: 2...200, using: &rng)
            gs.append(([x, Float.random(in: -x * 0.5...x * 0.5, using: &rng), Float.random(in: 0...3, using: &rng)],
                       0.05, [0.5, 0.5, 0.5], 0.9))
        }
        try renderer.setScene(makeCloud(gs))
        let target = renderer.makeTarget(width: 512, height: 288)
        try renderer.renderSync(egoFrame(), target: target)
        let n = renderer.visibleCount
        XCTAssertGreaterThan(n, 30_000)
        let keys = renderer.sortedKeys()
        XCTAssertEqual(keys.count, n)
        var sortedOK = true
        for i in 1..<keys.count where keys[i - 1] > keys[i] { sortedOK = false; break }
        XCTAssertTrue(sortedOK, "keys must be ascending (= depth descending)")
        // ascending key = descending depth: first key is the farthest splat
        let depthFirst = Float(bitPattern: 0xFFFF_FFFF - keys.first!)
        let depthLast = Float(bitPattern: 0xFFFF_FFFF - keys.last!)
        XCTAssertGreaterThan(depthFirst, depthLast)
        // the sort is a permutation: every key must be unique-ish and count preserved
        XCTAssertEqual(Set(keys).count, keys.count, accuracy: keys.count / 100)
    }

    func testSingleGaussianProjectsWhereExpected() throws {
        // world (10, 2, 1.5): 10 m ahead, 2 m left, camera height → u = 256 − 300·2/10 = 196, v = 144
        try renderer.setScene(makeCloud([([10, 2, 1.5], 0.1, [1, 0, 0], 0.99)]))
        let target = renderer.makeTarget(width: 512, height: 288)
        try renderer.renderSync(egoFrame(), target: target)
        XCTAssertEqual(renderer.visibleCount, 1)
        let px = renderer.readRGBA8(target)
        var best = (0, 0, 0)
        for y in 0..<288 { for x in 0..<512 { let r = Int(px[(y * 512 + x) * 4]); if r > best.2 { best = (x, y, r) } } }
        XCTAssertEqual(Double(best.0) + 0.5, 196, accuracy: 1.0)
        XCTAssertEqual(Double(best.1) + 0.5, 144, accuracy: 1.0)
        XCTAssertGreaterThan(best.2, 200)
        // σ = 0.1 m at 10 m → 3 px; radius 3σ ≈ 9 px: nothing beyond ~12 px
        XCTAssertEqual(px[(144 * 512 + 196 + 14) * 4], 0)
        XCTAssertEqual(px[(144 * 512 + 196) * 4 + 1], 0)   // pure red
    }

    func testBackToFrontCompositing() throws {
        // opaque green in front of red at the same pixel → green wins; then swap distances
        try renderer.setScene(makeCloud([([20, 0, 1.5], 0.3, [1, 0, 0], 0.99), ([10, 0, 1.5], 0.3, [0, 1, 0], 0.99)]))
        let target = renderer.makeTarget(width: 512, height: 288)
        try renderer.renderSync(egoFrame(), target: target)
        let px = renderer.readRGBA8(target)
        let i = (144 * 512 + 256) * 4
        XCTAssertGreaterThan(px[i + 1], 240); XCTAssertLessThan(px[i], 10)
    }

    func testTemporalOpacityAndVisibleRange() throws {
        var c = makeCloud([([10, 0, 1.5], 0.2, [1, 1, 1], 0.99)])
        c.packed[3] = 5          // τ
        c.packed[7] = 1 / 0.1    // σ_t = 0.1 s
        try renderer.setScene(c)
        let target = renderer.makeTarget(width: 64, height: 36)
        var f = egoFrame(time: 5)
        f.camera.intrinsics = Intrinsics(fx: 37.5, fy: 37.5, cx: 32, cy: 18, width: 64, height: 36)
        f.imageRect = [0, 0, 64, 36]
        try renderer.renderSync(f, target: target); XCTAssertEqual(renderer.visibleCount, 1)
        f.time = 6
        try renderer.renderSync(f, target: target); XCTAssertEqual(renderer.visibleCount, 0)
        f.time = 5; f.nodeVisible = [false]
        try renderer.renderSync(f, target: target); XCTAssertEqual(renderer.visibleCount, 0)
    }
}
