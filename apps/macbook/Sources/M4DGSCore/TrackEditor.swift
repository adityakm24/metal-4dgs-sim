import Foundation
import simd

/// Programmatic trajectory edit for one actor. Offsets are in the actor's own frame
/// (x forward, y left, z up) at each instant.
public struct TrackEdit: Codable, Equatable, Sendable {
    public var lateral: Double = 0          // m, along the actor's left (+y) axis
    public var longitudinal: Double = 0     // m, along the actor's forward (+x) axis
    public var timeShift: Double = 0        // s, positive = actor arrives later
    public var speedScale: Double = 1       // >1 = faster along the same path
    public var cutInEnabled: Bool = false
    public var cutInOffset: Double = -1.8   // m, extra lateral offset reached at the end of the window
    public var cutInStart: Double = 2.5     // s (scene time)
    public var cutInDuration: Double = 1.5  // s
    public var alignHeading: Bool = true    // yaw the actor along the cut-in path

    public init() {}

    public var isIdentity: Bool {
        lateral == 0 && longitudinal == 0 && timeShift == 0 && speedScale == 1 && !(cutInEnabled && cutInOffset != 0)
    }

    /// Total lateral offset at scene time t (static + cut-in blend).
    public func lateralOffset(at t: Double) -> Double {
        var l = lateral
        if cutInEnabled {
            let d = max(cutInDuration, 1e-3)
            l += cutInOffset * smootherstep((t - cutInStart) / d)
        }
        return l
    }

    /// Track time sampled for scene time t (time shift + speed scale about the track start t0).
    public func sourceTime(_ t: Double, trackStart t0: Double) -> Double {
        t0 + max(speedScale, 0) * (t - t0 - timeShift)
    }
}

public enum TrackEditor {
    /// Applies an edit to a track. Identity edits return the track unchanged; otherwise the track is
    /// resampled every `dt` seconds over `timeRange` (endpoints included) so that the time remap and the
    /// smooth cut-in are represented faithfully by the lerp/slerp interpolation readers use.
    public static func apply(_ e: TrackEdit, to keys: [TrackKey], timeRange: ClosedRange<Double>, dt: Double = 0.1) -> [TrackKey] {
        guard !e.isIdentity, !keys.isEmpty else { return keys }
        let t0 = keys.first!.t
        let lo = min(timeRange.lowerBound, t0), hi = max(timeRange.upperBound, lo)
        let n = max(1, Int(((hi - lo) / dt).rounded(.up)))
        var out: [TrackKey] = []
        out.reserveCapacity(n + 1)
        for i in 0...n {
            let t = i == n ? hi : lo + Double(i) * dt
            out.append(sample(e, keys: keys, t0: t0, at: t))
        }
        return out
    }

    static func sample(_ e: TrackEdit, keys: [TrackKey], t0: Double, at t: Double) -> TrackKey {
        let base = Track.pose(keys, at: e.sourceTime(t, trackStart: t0))
        let lat = e.lateralOffset(at: t)
        let pos = base.pos + base.forward * e.longitudinal + base.left * lat
        var q = base.quat
        if e.alignHeading && e.cutInEnabled && e.cutInOffset != 0 {
            let h = 0.02
            let dLat = (e.lateralOffset(at: t + h) - e.lateralOffset(at: t - h)) / (2 * h)
            let pa = Track.pose(keys, at: e.sourceTime(t - h, trackStart: t0)).pos
            let pb = Track.pose(keys, at: e.sourceTime(t + h, trackStart: t0)).pos
            let v = simd_dot((pb - pa) / (2 * h), base.forward)
            if abs(v) > 0.5 {
                q = (base.quat * Quat.yaw(atan(dLat / v))).normalized
            }
        }
        return TrackKey(t: t, pos: pos, quat: q)
    }
}

/// Parses a perturbation string like `actor_2:lateral_offset=-1.8m,t=2.5s` (as used by eval/closed_loop)
/// into an edit; unknown keys are ignored. Returns (actor, edit).
public func parsePerturbation(_ s: String) -> (String, TrackEdit)? {
    let parts = s.split(separator: ":", maxSplits: 1).map(String.init)
    guard parts.count == 2 else { return nil }
    var e = TrackEdit()
    var kv: [String: Double] = [:]
    for item in parts[1].split(separator: ",") {
        let p = item.split(separator: "=", maxSplits: 1).map { $0.trimmingCharacters(in: .whitespaces) }
        guard p.count == 2 else { continue }
        let v = p[1].trimmingCharacters(in: CharacterSet(charactersIn: "ms "))
        if let d = Double(v) { kv[p[0]] = d }
    }
    if let l = kv["lateral_offset"] {
        if let t = kv["t"] {
            e.cutInEnabled = true; e.cutInOffset = l; e.cutInStart = t; e.cutInDuration = kv["duration"] ?? 1.5
        } else { e.lateral = l }
    }
    if let x = kv["longitudinal_offset"] { e.longitudinal = x }
    if let x = kv["time_shift"] { e.timeShift = x }
    if let x = kv["speed_scale"] { e.speedScale = x }
    return (parts[0], e)
}
