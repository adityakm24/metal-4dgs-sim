import Foundation

/// Closed-loop rollout log (docs/formats.md §4). Parsed leniently.
public struct RolloutStep: Sendable {
    public var t: Double
    public var ego: Pose
    public var speed: Double?
    public var steer: Double?
    public var accel: Double?
    public var latencyMs: [(String, Double)]   // stage order preserved from the file where possible
    public var events: [String]

    public var hasCollision: Bool { events.contains { $0.hasPrefix("collision") } }
    public var totalLatencyMs: Double { latencyMs.reduce(0) { $0 + $1.1 } }
}

public enum MetricValue: Sendable, CustomStringConvertible {
    case number(Double), bool(Bool), string(String), other(String)
    public var description: String {
        switch self {
        case .number(let d): return d == d.rounded() && abs(d) < 1e9 ? String(format: "%.0f", d) : String(format: "%.3f", d)
        case .bool(let b): return b ? "true" : "false"
        case .string(let s): return s
        case .other(let s): return s
        }
    }
}

public struct Rollout: Sendable {
    public var scene: String?
    public var policy: String?
    public var dt: Double?
    public var perturbations: [String]
    public var actorTracks: [String: [TrackKey]]
    public var steps: [RolloutStep]
    public var metrics: [(String, MetricValue)]

    public var timeRange: ClosedRange<Double> {
        guard let a = steps.first?.t, let b = steps.last?.t else { return 0...0 }
        return a...max(a, b)
    }
    public var egoTrack: [TrackKey] { steps.map { TrackKey(t: $0.t, pos: $0.ego.pos, quat: $0.ego.quat) } }

    /// Index of the last step with step.t ≤ t (clamped).
    public func stepIndex(at t: Double) -> Int? {
        guard !steps.isEmpty else { return nil }
        var lo = 0, hi = steps.count - 1
        if t <= steps[0].t { return 0 }
        if t >= steps[hi].t { return hi }
        while hi - lo > 1 {
            let m = (lo + hi) / 2
            if steps[m].t <= t { lo = m } else { hi = m }
        }
        return lo
    }

    public init(url: URL) throws { try self.init(data: Data(contentsOf: url)) }

    public init(data: Data) throws {
        guard let d = try JSONSerialization.jsonObject(with: data) as? [String: Any] else { throw SceneGraphError.notJSON }
        scene = d["scene"] as? String
        policy = d["policy"] as? String
        dt = num(d["dt"])
        perturbations = (d["perturbations"] as? [Any])?.map { ($0 as? String) ?? "\($0)" } ?? []
        var at: [String: [TrackKey]] = [:]
        if let a = d["actor_tracks"] as? [String: Any] {
            for (k, v) in a { at[k] = parseTrack(v) }
        }
        actorTracks = at
        let preferred = ["raster", "refine", "policy"]
        steps = ((d["steps"] as? [[String: Any]]) ?? []).compactMap { s in
            guard let t = num(s["t"]) else { return nil }
            let e = (s["ego"] as? [String: Any]) ?? [:]
            let act = (s["action"] as? [String: Any]) ?? [:]
            var lat: [(String, Double)] = []
            if let l = s["latency_ms"] as? [String: Any] {
                let keys = l.keys.sorted { a, b in
                    let ia = preferred.firstIndex(of: a) ?? 99, ib = preferred.firstIndex(of: b) ?? 99
                    return ia != ib ? ia < ib : a < b
                }
                lat = keys.compactMap { k in num(l[k]).map { (k, $0) } }
            }
            return RolloutStep(t: t,
                               ego: Pose(pos: vec3(e["pos"]) ?? .zero, quat: Quat(nums(e["quat"]) ?? [1, 0, 0, 0])),
                               speed: num(s["speed"]), steer: num(act["steer"]), accel: num(act["accel"]),
                               latencyMs: lat,
                               events: (s["events"] as? [Any])?.map { ($0 as? String) ?? "\($0)" } ?? [])
        }.sorted { $0.t < $1.t }
        var m: [(String, MetricValue)] = []
        if let md = d["metrics"] as? [String: Any] {
            for k in md.keys.sorted() {
                let v = md[k]!
                if let n = v as? NSNumber {
                    if CFGetTypeID(n) == CFBooleanGetTypeID() { m.append((k, .bool(n.boolValue))) } else { m.append((k, .number(n.doubleValue))) }
                } else if let s = v as? String { m.append((k, .string(s))) } else { m.append((k, .other("\(v)"))) }
            }
        }
        metrics = m
    }

    /// Resolves the `scene` path: absolute, relative to the rollout file (and its parents), or to the cwd.
    public func resolveScene(relativeTo rolloutURL: URL) -> URL? {
        guard let s = scene, !s.isEmpty else { return nil }
        let fm = FileManager.default
        if s.hasPrefix("/") { return fm.fileExists(atPath: s) ? URL(fileURLWithPath: s) : nil }
        var dir = rolloutURL.deletingLastPathComponent()
        for _ in 0..<6 {
            let c = dir.appendingPathComponent(s)
            if fm.fileExists(atPath: c.path) { return c.standardizedFileURL }
            let byName = dir.appendingPathComponent((s as NSString).lastPathComponent)
            if fm.fileExists(atPath: byName.path) { return byName.standardizedFileURL }
            dir = dir.deletingLastPathComponent()
        }
        let cwd = URL(fileURLWithPath: fm.currentDirectoryPath).appendingPathComponent(s)
        return fm.fileExists(atPath: cwd.path) ? cwd : nil
    }
}
