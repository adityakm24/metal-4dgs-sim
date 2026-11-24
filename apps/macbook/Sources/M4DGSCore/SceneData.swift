import Foundation
import simd

/// GPU-ready packed Gaussians for all scene-graph nodes (docs/formats.md §3.1).
///
/// `packed` holds 16 floats per Gaussian:
///   [0..2] mean (node-local)      [3]  t_center τ
///   [4..6] velocity (node-local)  [7]  1/σ_t
///   [8..11] quat (w,x,y,z), normalized
///   [12..14] scale = exp(log_scales) [15] opacity = sigmoid(logit)
public struct GaussianCloud: Sendable {
    public static let floatsPerGaussian = 16
    public var count: Int
    public var shDegree: Int
    public var shCoeffs: Int            // K = (d+1)^2
    public var packed: [Float]
    public var sh: [Float]              // count × K × 3
    public var nodeIds: [UInt32]
    public var nodeRanges: [String: Range<Int>]

    public var shStride: Int { shCoeffs * 3 }

    public init(count: Int, shDegree: Int, packed: [Float], sh: [Float], nodeIds: [UInt32], nodeRanges: [String: Range<Int>]) {
        self.count = count; self.shDegree = shDegree; self.shCoeffs = SH.coeffCount(degree: shDegree)
        self.packed = packed; self.sh = sh; self.nodeIds = nodeIds; self.nodeRanges = nodeRanges
    }

    /// CPU reference Gaussian i (for tests / picking).
    public func gaussian(_ i: Int) -> Gaussian4D {
        let b = i * Self.floatsPerGaussian
        let p = packed
        let op = Double(p[b + 15])
        return Gaussian4D(mean: SIMD3(Double(p[b]), Double(p[b + 1]), Double(p[b + 2])),
                          logScale: SIMD3(log(Double(p[b + 12])), log(Double(p[b + 13])), log(Double(p[b + 14]))),
                          quat: Quat(w: Double(p[b + 8]), x: Double(p[b + 9]), y: Double(p[b + 10]), z: Double(p[b + 11])),
                          opacityLogit: log(op / max(1 - op, 1e-12)),
                          velocity: SIMD3(Double(p[b + 4]), Double(p[b + 5]), Double(p[b + 6])),
                          tCenter: Double(p[b + 3]),
                          tLogScale: -log(max(Double(p[b + 7]), 1e-30)))
    }

    public func shCoefficients(_ i: Int) -> [SIMD3<Float>] {
        (0..<shCoeffs).map { k in
            let b = i * shStride + k * 3
            return SIMD3(sh[b], sh[b + 1], sh[b + 2])
        }
    }

    /// Builds the cloud from a safetensors checkpoint, in scene-graph node order.
    public static func build(from st: SafeTensorsFile, graph: SceneGraph) throws -> GaussianCloud {
        let K = SH.coeffCount(degree: graph.shDegree)
        var packed: [Float] = []
        var sh: [Float] = []
        var ids: [UInt32] = []
        var ranges: [String: Range<Int>] = [:]
        var total = 0
        for (ni, node) in graph.nodes.enumerated() {
            let n = node.name
            guard let mi = st.info("\(n).means"), mi.shape.count == 2, mi.shape[1] == 3 else {
                ranges[n] = total..<total
                continue
            }
            let N = mi.shape[0]
            let means = try st.floats("\(n).means")
            let ls = try st.floats("\(n).log_scales")
            let qs = try st.floats("\(n).quats")
            let ol = try st.floats("\(n).opacity_logits")
            let shRaw = try st.floats("\(n).sh")
            let vel = st.info("\(n).velocity") != nil ? try st.floats("\(n).velocity") : [Float](repeating: 0, count: N * 3)
            let tc = st.info("\(n).t_center") != nil ? try st.floats("\(n).t_center") : [Float](repeating: 0, count: N)
            let tls = st.info("\(n).t_log_scale") != nil ? try st.floats("\(n).t_log_scale") : [Float](repeating: 30, count: N)
            guard ls.count == N * 3, qs.count == N * 4, ol.count == N, vel.count == N * 3, tc.count == N, tls.count == N else {
                throw SafeTensorsError.badHeader("inconsistent tensor shapes for node '\(n)'")
            }
            let shShape = st.info("\(n).sh")!.shape
            let Kn = shShape.count == 3 ? shShape[1] : max(1, shRaw.count / max(N * 3, 1))
            guard shRaw.count == N * Kn * 3 else { throw SafeTensorsError.badHeader("bad sh shape for '\(n)'") }

            packed.reserveCapacity(packed.count + N * floatsPerGaussian)
            sh.reserveCapacity(sh.count + N * K * 3)
            for i in 0..<N {
                var q = SIMD4<Float>(qs[i * 4], qs[i * 4 + 1], qs[i * 4 + 2], qs[i * 4 + 3])
                let qn = simd_length(q)
                q = qn > 1e-12 ? q / qn : SIMD4(1, 0, 0, 0)
                let invSigmaT = exp(-tls[i])
                packed += [means[i * 3], means[i * 3 + 1], means[i * 3 + 2], tc[i],
                           vel[i * 3], vel[i * 3 + 1], vel[i * 3 + 2], invSigmaT.isFinite ? invSigmaT : 0,
                           q.x, q.y, q.z, q.w,
                           exp(ls[i * 3]), exp(ls[i * 3 + 1]), exp(ls[i * 3 + 2]), 1 / (1 + exp(-ol[i]))]
                let base = i * Kn * 3
                for k in 0..<K {
                    if k < Kn {
                        sh += [shRaw[base + k * 3], shRaw[base + k * 3 + 1], shRaw[base + k * 3 + 2]]
                    } else {
                        sh += [0, 0, 0]
                    }
                }
            }
            ids += [UInt32](repeating: UInt32(ni), count: N)
            ranges[n] = total..<(total + N)
            total += N
        }
        return GaussianCloud(count: total, shDegree: graph.shDegree, packed: packed, sh: sh, nodeIds: ids, nodeRanges: ranges)
    }
}

/// A scene checkpoint loaded from disk, with its optional `.edits.json` sidecar (§3.3).
public final class LoadedScene: @unchecked Sendable {
    public let url: URL
    public let original: SceneGraphDocument   // scene graph stored in the checkpoint
    public let loaded: SceneGraphDocument     // after applying the sidecar (== original if none)
    public let editsApplied: URL?
    public let cloud: GaussianCloud
    public let metadata: [String: String]

    public var graph: SceneGraph { loaded.graph }

    /// `<dir>/<stem>.edits.json` (e.g. `scene_01.safetensors` → `scene_01.edits.json`).
    public static func editsURL(for scene: URL) -> URL {
        scene.deletingPathExtension().appendingPathExtension("edits.json")
    }
    /// Sidecar candidates in priority order (also accept `scene_01.safetensors.edits.json`).
    public static func editsCandidates(for scene: URL) -> [URL] {
        [editsURL(for: scene), scene.appendingPathExtension("edits.json")]
    }

    public init(url: URL, applyEdits: Bool = true, editsOverride: URL? = nil) throws {
        self.url = url
        let st = try SafeTensorsFile(url: url)
        metadata = st.metadata
        guard let sg = st.metadata["scene_graph"] else { throw SceneGraphError.missing("__metadata__.scene_graph") }
        original = try SceneGraphDocument(jsonString: sg)
        var loadedDoc = original
        var applied: URL?
        if applyEdits {
            let cands = editsOverride.map { [$0] } ?? LoadedScene.editsCandidates(for: url)
            for c in cands where FileManager.default.fileExists(atPath: c.path) {
                loadedDoc = try SceneGraphDocument(jsonData: Data(contentsOf: c))
                applied = c
                break
            }
        }
        loaded = loadedDoc
        editsApplied = applied
        // Tensors are grouped by node name; build in the (edited) graph's node order.
        cloud = try GaussianCloud.build(from: st, graph: loadedDoc.graph)
    }
}

/// Everything the renderer needs about the dynamic scene at time t.
public struct SceneFrame: Sendable {
    public var time: Double
    public var egoPose: Pose
    public var nodePoses: [Pose]        // scene-graph order
    public var nodeVisible: [Bool]

    /// Evaluates node poses: actors (any node with a track) interpolate their (possibly overridden) track;
    /// nodes without a track use identity. `visible_range` is honoured.
    public static func evaluate(graph: SceneGraph, time t: Double,
                                trackOverrides: [String: [TrackKey]] = [:],
                                egoTrack: [TrackKey]? = nil) -> SceneFrame {
        var poses: [Pose] = []
        var vis: [Bool] = []
        for n in graph.nodes {
            let tr = trackOverrides[n.name] ?? n.track
            poses.append(tr.isEmpty ? .identity : Track.pose(tr, at: t))
            vis.append(n.isVisible(at: t))
        }
        let ego = Track.pose(egoTrack ?? graph.ego?.track ?? [], at: t)
        return SceneFrame(time: t, egoPose: ego, nodePoses: poses, nodeVisible: vis)
    }
}
