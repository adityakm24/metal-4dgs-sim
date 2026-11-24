import Foundation
import simd

/// Scene graph (docs/formats.md §3.2). Parsed leniently from JSON; the raw JSON object is kept so that
/// writing `<scene>.edits.json` preserves every field we do not interpret (forward compatibility).
public struct SceneNode: Sendable, Equatable {
    public var name: String
    public var kind: String
    public var category: String?
    public var size: SIMD3<Double>?
    public var track: [TrackKey]
    /// Optional `[t0, t1]`: the node exists only for t0 ≤ t ≤ t1.
    public var visibleRange: ClosedRange<Double>?

    public var isActor: Bool { kind == "actor" }
    public var isBackground: Bool { kind == "background" }
    public func isVisible(at t: Double) -> Bool { visibleRange.map { $0.contains(t) } ?? true }
}

public struct EgoSpec: Sendable {
    public var size: SIMD3<Double>
    public var cameraToEgo: RigidTransform
    public var intrinsics: Intrinsics
    public var track: [TrackKey]
}

public struct SceneGraph: Sendable {
    public var shDegree: Int
    public var timeRange: ClosedRange<Double>
    public var nodes: [SceneNode]
    public var ego: EgoSpec?
    public var backgroundColor: SIMD3<Double>?

    public func node(named n: String) -> SceneNode? { nodes.first { $0.name == n } }
    public func index(of n: String) -> Int? { nodes.firstIndex { $0.name == n } }
    public var actors: [SceneNode] { nodes.filter { $0.isActor } }
}

public enum SceneGraphError: Error, CustomStringConvertible {
    case notJSON, missing(String)
    public var description: String {
        switch self {
        case .notJSON: return "scene_graph is not a JSON object"
        case .missing(let k): return "scene_graph is missing '\(k)'"
        }
    }
}

// MARK: lenient JSON helpers

func num(_ v: Any?) -> Double? { (v as? NSNumber)?.doubleValue }
func nums(_ v: Any?) -> [Double]? {
    guard let a = v as? [Any] else { return nil }
    let r = a.compactMap { num($0) }
    return r.count == a.count ? r : nil
}
func vec3(_ v: Any?) -> SIMD3<Double>? {
    guard let a = nums(v), a.count >= 3 else { return nil }
    return SIMD3(a[0], a[1], a[2])
}
func parseTrack(_ v: Any?) -> [TrackKey] {
    guard let arr = v as? [[String: Any]] else { return [] }
    let keys: [TrackKey] = arr.compactMap { d in
        guard let t = num(d["t"]), let p = vec3(d["pos"]) else { return nil }
        return TrackKey(t: t, pos: p, quat: Quat(nums(d["quat"]) ?? [1, 0, 0, 0]))
    }
    return Track.sorted(keys)
}
func parseRange(_ v: Any?) -> ClosedRange<Double>? {
    guard let a = nums(v), a.count >= 2 else { return nil }
    return min(a[0], a[1])...max(a[0], a[1])
}

/// Scene graph plus its raw JSON object.
public struct SceneGraphDocument: @unchecked Sendable {
    public var graph: SceneGraph
    public var raw: [String: Any]

    public init(jsonString: String) throws {
        guard let data = jsonString.data(using: .utf8) else { throw SceneGraphError.notJSON }
        try self.init(jsonData: data)
    }

    public init(jsonData: Data) throws {
        guard let obj = try JSONSerialization.jsonObject(with: jsonData) as? [String: Any] else { throw SceneGraphError.notJSON }
        try self.init(raw: obj)
    }

    public init(raw: [String: Any]) throws {
        self.raw = raw
        guard let nodesAny = raw["nodes"] as? [[String: Any]] else { throw SceneGraphError.missing("nodes") }
        let nodes: [SceneNode] = nodesAny.enumerated().map { i, d in
            SceneNode(name: (d["name"] as? String) ?? "node_\(i)",
                      kind: (d["kind"] as? String) ?? (d["track"] != nil ? "actor" : "background"),
                      category: d["category"] as? String,
                      size: vec3(d["size"]),
                      track: parseTrack(d["track"]),
                      visibleRange: parseRange(d["visible_range"]))
        }
        var ego: EgoSpec?
        if let e = raw["ego"] as? [String: Any] {
            let cam = (e["camera"] as? [String: Any]) ?? [:]
            let w = Int(num(cam["width"]) ?? 512), h = Int(num(cam["height"]) ?? 288)
            let k = Intrinsics(fx: num(cam["fx"]) ?? 300, fy: num(cam["fy"]) ?? num(cam["fx"]) ?? 300,
                               cx: num(cam["cx"]) ?? Double(w) / 2, cy: num(cam["cy"]) ?? Double(h) / 2,
                               width: w, height: h)
            let c2e = (e["camera_to_ego"] as? [[Any]]).map { $0.map { nums($0) ?? [] } }
            ego = EgoSpec(size: vec3(e["size"]) ?? [4.5, 1.9, 1.6],
                          cameraToEgo: c2e.map { RigidTransform(rowMajor: $0) } ?? .identity,
                          intrinsics: k,
                          track: parseTrack(e["track"]))
        }
        // time range: explicit, else union of tracks
        var tr = parseRange(raw["time_range"])
        if tr == nil {
            let all = nodes.flatMap { $0.track.map(\.t) } + (ego?.track.map(\.t) ?? [])
            if let lo = all.min(), let hi = all.max() { tr = lo...hi }
        }
        graph = SceneGraph(shDegree: Int(num(raw["sh_degree"]) ?? 0),
                           timeRange: tr ?? 0...1,
                           nodes: nodes,
                           ego: ego,
                           backgroundColor: vec3(raw["background_color"]))
    }

    /// Raw JSON object with the tracks of the given nodes replaced. All other fields
    /// (including optional ones like `background_color` / `visible_range`) are preserved verbatim.
    public func replacingTracks(_ tracks: [String: [TrackKey]]) -> [String: Any] {
        var out = raw
        if var nodes = out["nodes"] as? [[String: Any]] {
            for i in nodes.indices {
                if let name = nodes[i]["name"] as? String, let tr = tracks[name] {
                    nodes[i]["track"] = tr.map(\.jsonObject)
                }
            }
            out["nodes"] = nodes
        }
        return out
    }

    public func jsonData(replacingTracks tracks: [String: [TrackKey]]) throws -> Data {
        try JSONSerialization.data(withJSONObject: replacingTracks(tracks), options: [.prettyPrinted, .sortedKeys])
    }
}
