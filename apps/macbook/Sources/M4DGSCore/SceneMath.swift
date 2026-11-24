import Foundation
import simd

// MARK: - Quaternion (w, x, y, z), Hamilton convention (docs/formats.md §1)

public struct Quat: Equatable, Sendable, CustomStringConvertible {
    public var w: Double, x: Double, y: Double, z: Double

    public init(w: Double, x: Double, y: Double, z: Double) { self.w = w; self.x = x; self.y = y; self.z = z }
    /// From a `[w, x, y, z]` array (identity if malformed).
    public init(_ a: [Double]) {
        if a.count == 4 { self.init(w: a[0], x: a[1], y: a[2], z: a[3]) } else { self = .identity }
    }
    public static let identity = Quat(w: 1, x: 0, y: 0, z: 0)

    public init(axis: SIMD3<Double>, angle: Double) {
        let a = simd_normalize(axis)
        let s = sin(angle / 2)
        self.init(w: cos(angle / 2), x: a.x * s, y: a.y * s, z: a.z * s)
    }
    /// Rotation about +z (yaw) — the world is z-up.
    public static func yaw(_ angle: Double) -> Quat { Quat(axis: [0, 0, 1], angle: angle) }

    public var array: [Double] { [w, x, y, z] }
    public var description: String { String(format: "(%.4f, %.4f, %.4f, %.4f)", w, x, y, z) }
    public var norm: Double { (w * w + x * x + y * y + z * z).squareRoot() }
    public var normalized: Quat {
        let n = norm
        return n > 1e-12 ? Quat(w: w / n, x: x / n, y: y / n, z: z / n) : .identity
    }
    public var conjugate: Quat { Quat(w: w, x: -x, y: -y, z: -z) }
    public static prefix func - (q: Quat) -> Quat { Quat(w: -q.w, x: -q.x, y: -q.y, z: -q.z) }

    public static func * (a: Quat, b: Quat) -> Quat {
        Quat(w: a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
             x: a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
             y: a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
             z: a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w)
    }

    public func dot(_ o: Quat) -> Double { w * o.w + x * o.x + y * o.y + z * o.z }

    /// Rotation matrix of the normalized quaternion.
    public var matrix: simd_double3x3 {
        let q = normalized
        let (w, x, y, z) = (q.w, q.x, q.y, q.z)
        let c0 = SIMD3<Double>(1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y))
        let c1 = SIMD3<Double>(2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x))
        let c2 = SIMD3<Double>(2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y))
        return simd_double3x3(columns: (c0, c1, c2))
    }

    public func rotate(_ v: SIMD3<Double>) -> SIMD3<Double> { matrix * v }

    /// Quaternion from a (near-)orthonormal rotation matrix (Shepperd's method).
    public init(matrix m: simd_double3x3) {
        let r00 = m[0][0], r11 = m[1][1], r22 = m[2][2]
        let r01 = m[1][0], r10 = m[0][1], r02 = m[2][0], r20 = m[0][2], r12 = m[2][1], r21 = m[1][2]
        let tr = r00 + r11 + r22
        var q: Quat
        if tr > 0 {
            let s = (tr + 1).squareRoot() * 2
            q = Quat(w: 0.25 * s, x: (r21 - r12) / s, y: (r02 - r20) / s, z: (r10 - r01) / s)
        } else if r00 > r11 && r00 > r22 {
            let s = (1 + r00 - r11 - r22).squareRoot() * 2
            q = Quat(w: (r21 - r12) / s, x: 0.25 * s, y: (r01 + r10) / s, z: (r02 + r20) / s)
        } else if r11 > r22 {
            let s = (1 + r11 - r00 - r22).squareRoot() * 2
            q = Quat(w: (r02 - r20) / s, x: (r01 + r10) / s, y: 0.25 * s, z: (r12 + r21) / s)
        } else {
            let s = (1 + r22 - r00 - r11).squareRoot() * 2
            q = Quat(w: (r10 - r01) / s, x: (r02 + r20) / s, y: (r12 + r21) / s, z: 0.25 * s)
        }
        self = q.normalized
    }

    /// Yaw (heading about +z) of the rotated +x axis.
    public var yawAngle: Double {
        let f = rotate([1, 0, 0])
        return atan2(f.y, f.x)
    }

    /// Spherical linear interpolation along the shortest arc. Inputs need not be normalized.
    public static func slerp(_ a0: Quat, _ b0: Quat, _ t: Double) -> Quat {
        let a = a0.normalized
        var b = b0.normalized
        var d = a.dot(b)
        if d < 0 { b = -b; d = -d }
        if d > 0.9995 {
            return Quat(w: a.w + (b.w - a.w) * t, x: a.x + (b.x - a.x) * t,
                        y: a.y + (b.y - a.y) * t, z: a.z + (b.z - a.z) * t).normalized
        }
        let th0 = acos(min(d, 1))
        let th = th0 * t
        let s0 = cos(th) - d * sin(th) / sin(th0)
        let s1 = sin(th) / sin(th0)
        return Quat(w: a.w * s0 + b.w * s1, x: a.x * s0 + b.x * s1,
                    y: a.y * s0 + b.y * s1, z: a.z * s0 + b.z * s1).normalized
    }
}

// MARK: - Poses

/// Rigid transform `p_out = R · p_in + t` (rotation stored as a matrix; used for cameras).
public struct RigidTransform: Sendable {
    public var R: simd_double3x3
    public var t: SIMD3<Double>
    public init(R: simd_double3x3, t: SIMD3<Double>) { self.R = R; self.t = t }
    public static let identity = RigidTransform(R: matrix_identity_double3x3, t: .zero)

    /// From a 4x4 row-major nested list (docs/formats.md §1).
    public init(rowMajor m: [[Double]]) {
        guard m.count >= 3, m.allSatisfy({ $0.count >= 4 }) else { self = .identity; return }
        let c0 = SIMD3<Double>(m[0][0], m[1][0], m[2][0])
        let c1 = SIMD3<Double>(m[0][1], m[1][1], m[2][1])
        let c2 = SIMD3<Double>(m[0][2], m[1][2], m[2][2])
        R = simd_double3x3(columns: (c0, c1, c2))
        t = SIMD3<Double>(m[0][3], m[1][3], m[2][3])
    }

    public var rowMajor: [[Double]] {
        [[R[0][0], R[1][0], R[2][0], t.x],
         [R[0][1], R[1][1], R[2][1], t.y],
         [R[0][2], R[1][2], R[2][2], t.z],
         [0, 0, 0, 1]]
    }

    public func apply(_ p: SIMD3<Double>) -> SIMD3<Double> { R * p + t }
    public static func * (a: RigidTransform, b: RigidTransform) -> RigidTransform {
        RigidTransform(R: a.R * b.R, t: a.R * b.t + a.t)
    }
    /// Inverse assuming R is a rotation.
    public var inverse: RigidTransform {
        let rt = R.transpose
        return RigidTransform(R: rt, t: -(rt * t))
    }
    public var matrix4x4: simd_double4x4 {
        simd_double4x4(columns: (SIMD4(R.columns.0, 0), SIMD4(R.columns.1, 0), SIMD4(R.columns.2, 0), SIMD4(t, 1)))
    }
}

public struct Pose: Sendable, Equatable {
    public var pos: SIMD3<Double>
    public var quat: Quat
    public init(pos: SIMD3<Double> = .zero, quat: Quat = .identity) { self.pos = pos; self.quat = quat }
    public static let identity = Pose()
    public var rigid: RigidTransform { RigidTransform(R: quat.matrix, t: pos) }
    public func apply(_ p: SIMD3<Double>) -> SIMD3<Double> { quat.rotate(p) + pos }
    public var forward: SIMD3<Double> { quat.rotate([1, 0, 0]) }
    public var left: SIMD3<Double> { quat.rotate([0, 1, 0]) }
    public var up: SIMD3<Double> { quat.rotate([0, 0, 1]) }
}

// MARK: - Tracks

public struct TrackKey: Codable, Equatable, Sendable {
    public var t: Double
    public var pos: SIMD3<Double>
    public var quat: Quat

    public init(t: Double, pos: SIMD3<Double>, quat: Quat = .identity) { self.t = t; self.pos = pos; self.quat = quat }
    public var pose: Pose { Pose(pos: pos, quat: quat) }

    enum CodingKeys: String, CodingKey { case t, pos, quat }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        t = try c.decode(Double.self, forKey: .t)
        let p = try c.decode([Double].self, forKey: .pos)
        pos = p.count >= 3 ? SIMD3(p[0], p[1], p[2]) : .zero
        quat = Quat((try c.decodeIfPresent([Double].self, forKey: .quat)) ?? [1, 0, 0, 0])
    }
    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(t, forKey: .t)
        try c.encode([pos.x, pos.y, pos.z], forKey: .pos)
        try c.encode(quat.array, forKey: .quat)
    }
    /// JSON-serializable dictionary (for raw scene-graph rewriting).
    public var jsonObject: [String: Any] {
        ["t": t, "pos": [pos.x, pos.y, pos.z], "quat": quat.array]
    }
}

public enum Track {
    /// Pose on a track at time `t`: linear in pos, slerp in quat, clamped outside the track's time range (§3.2).
    /// `keys` must be sorted by `t`. Empty track → identity.
    public static func pose(_ keys: [TrackKey], at t: Double) -> Pose {
        guard let first = keys.first, let last = keys.last else { return .identity }
        if keys.count == 1 || t <= first.t { return Pose(pos: first.pos, quat: first.quat.normalized) }
        if t >= last.t { return Pose(pos: last.pos, quat: last.quat.normalized) }
        // binary search for the segment [i, i+1] containing t
        var lo = 0, hi = keys.count - 1
        while hi - lo > 1 {
            let mid = (lo + hi) / 2
            if keys[mid].t <= t { lo = mid } else { hi = mid }
        }
        let a = keys[lo], b = keys[hi]
        let span = b.t - a.t
        let u = span > 1e-12 ? (t - a.t) / span : 0
        return Pose(pos: a.pos + (b.pos - a.pos) * u, quat: Quat.slerp(a.quat, b.quat, u))
    }

    public static func sorted(_ keys: [TrackKey]) -> [TrackKey] { keys.sorted { $0.t < $1.t } }
    public static func timeRange(_ keys: [TrackKey]) -> ClosedRange<Double>? {
        guard let f = keys.first, let l = keys.last else { return nil }
        return f.t...max(f.t, l.t)
    }
}

// MARK: - Spherical harmonics (3DGS conventions)

public enum SH {
    public static let C0: Float = 0.28209479177387814
    public static let C1: Float = 0.4886025119029199
    public static let C2: [Float] = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
                                     -1.0925484305920792, 0.5462742152960396]
    public static let C3: [Float] = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
                                     0.3731763325901154, -0.4570457994644658, 1.445305721320277,
                                     -0.5900435899266435]

    public static func coeffCount(degree: Int) -> Int { (degree + 1) * (degree + 1) }

    /// Σ_k SH_k(dir) · sh[k] for degree 0...3. `sh` holds K RGB triples. `dir` must be unit length.
    public static func evaluate(degree: Int, sh: [SIMD3<Float>], dir d: SIMD3<Float>) -> SIMD3<Float> {
        var r = C0 * sh[0]
        if degree < 1 { return r }
        let x = d.x, y = d.y, z = d.z
        r += -C1 * y * sh[1] + C1 * z * sh[2] - C1 * x * sh[3]
        if degree < 2 { return r }
        let xx = x * x, yy = y * y, zz = z * z, xy = x * y, yz = y * z, xz = x * z
        r += C2[0] * xy * sh[4] + C2[1] * yz * sh[5] + C2[2] * (2 * zz - xx - yy) * sh[6]
        r += C2[3] * xz * sh[7] + C2[4] * (xx - yy) * sh[8]
        if degree < 3 { return r }
        r += C3[0] * y * (3 * xx - yy) * sh[9] + C3[1] * xy * z * sh[10]
        r += C3[2] * y * (4 * zz - xx - yy) * sh[11] + C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[12]
        r += C3[4] * x * (4 * zz - xx - yy) * sh[13] + C3[5] * z * (xx - yy) * sh[14]
        r += C3[6] * x * (xx - 3 * yy) * sh[15]
        return r
    }

    /// rgb = max(0.5 + Σ SH·coeffs, 0)
    public static func color(degree: Int, sh: [SIMD3<Float>], dir: SIMD3<Float>) -> SIMD3<Float> {
        simd_max(evaluate(degree: degree, sh: sh, dir: dir) + SIMD3<Float>(repeating: 0.5), .zero)
    }
}

// MARK: - 4D Gaussian evaluation (§3.1) — CPU reference of what the Metal preprocess kernel does

public struct Gaussian4D: Sendable {
    public var mean: SIMD3<Double>
    public var logScale: SIMD3<Double>
    public var quat: Quat
    public var opacityLogit: Double
    public var velocity: SIMD3<Double>
    public var tCenter: Double
    public var tLogScale: Double

    public init(mean: SIMD3<Double>, logScale: SIMD3<Double> = .zero, quat: Quat = .identity, opacityLogit: Double = 0,
                velocity: SIMD3<Double> = .zero, tCenter: Double = 0, tLogScale: Double = 10) {
        self.mean = mean; self.logScale = logScale; self.quat = quat; self.opacityLogit = opacityLogit
        self.velocity = velocity; self.tCenter = tCenter; self.tLogScale = tLogScale
    }

    public struct Evaluated: Sendable {
        public var meanWorld: SIMD3<Double>
        public var quatWorld: Quat
        public var alpha: Double
        public var scale: SIMD3<Double>
    }

    public static func sigmoid(_ x: Double) -> Double { 1 / (1 + exp(-x)) }

    public func evaluate(at t: Double, nodePose: Pose = .identity) -> Evaluated {
        let dt = t - tCenter
        let local = mean + velocity * dt
        let st = exp(tLogScale)
        let a = Gaussian4D.sigmoid(opacityLogit) * exp(-0.5 * (dt / st) * (dt / st))
        let qn = nodePose.quat.normalized
        return Evaluated(meanWorld: qn.rotate(local) + nodePose.pos,
                         quatWorld: qn * quat.normalized,
                         alpha: a,
                         scale: SIMD3(exp(logScale.x), exp(logScale.y), exp(logScale.z)))
    }

    /// World-space covariance Σ = R S Sᵀ Rᵀ.
    public static func covariance(quat: Quat, scale: SIMD3<Double>) -> simd_double3x3 {
        let R = quat.matrix
        let S = simd_double3x3(diagonal: scale)
        let M = R * S
        return M * M.transpose
    }
}

public func smootherstep(_ x: Double) -> Double {
    let t = min(max(x, 0), 1)
    return t * t * t * (t * (t * 6 - 15) + 10)
}
