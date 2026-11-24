import simd

// Pure keyframe-selection logic (no ARKit), checked on macOS by apps/iphone/Tests.

struct KeyframeSettings: Sendable, Equatable {
    /// Minimum time between kept keyframes (s).
    var minInterval: Double = 0.1
    /// Keep when the camera moved at least this far since the last keyframe (m)...
    var minTranslation: Float = 0.05
    /// ...or rotated at least this much (degrees).
    var minRotationDegrees: Float = 5
    /// Reject frames whose angular speed (from consecutive poses) exceeds this (rad/s) — motion blur.
    var maxAngularSpeed: Float = 1.0

    var minRotationRadians: Float { minRotationDegrees * .pi / 180 }
}

enum KeyframeDecision: String, Sendable, Equatable {
    case keep
    case trackingNotNormal
    case motionBlur
    case tooSoon
    case tooClose
}

struct KeyframeSelector: Sendable {
    var settings: KeyframeSettings
    /// Angular speed (rad/s) between the two most recent evaluated frames.
    private(set) var angularSpeed: Float = 0

    private var previousTimestamp: Double?
    private var previousPose: simd_float4x4?
    private var lastKeptTimestamp: Double?
    private var lastKeptPose: simd_float4x4?

    init(settings: KeyframeSettings = KeyframeSettings()) {
        self.settings = settings
    }

    /// Call for every ARFrame (also when not recording, so angular speed stays warm).
    /// Does not mark the frame as kept; call `markKept` once it was actually handed to the writer.
    mutating func evaluate(timestamp: Double, pose: simd_float4x4, trackingNormal: Bool) -> KeyframeDecision {
        let finite = MatrixSerialization.isFinite(pose)
        if finite, let pt = previousTimestamp, let pp = previousPose {
            let dt = timestamp - pt
            if dt > 1e-4 {
                angularSpeed = Self.rotationAngle(pp, pose) / Float(dt)
            }
        } else {
            angularSpeed = 0
        }
        if finite {
            previousTimestamp = timestamp
            previousPose = pose
        }

        guard trackingNormal, finite else { return .trackingNotNormal }
        if angularSpeed > settings.maxAngularSpeed { return .motionBlur }
        guard let kt = lastKeptTimestamp, let kp = lastKeptPose else { return .keep }
        if timestamp - kt < settings.minInterval { return .tooSoon }
        let translation = simd_distance(Self.translation(kp), Self.translation(pose))
        let rotation = Self.rotationAngle(kp, pose)
        if translation >= settings.minTranslation || rotation >= settings.minRotationRadians {
            return .keep
        }
        return .tooClose
    }

    mutating func markKept(timestamp: Double, pose: simd_float4x4) {
        lastKeptTimestamp = timestamp
        lastKeptPose = pose
    }

    /// Forget the last keyframe (start of a new recording); angular-speed history is kept.
    mutating func resetKeyframes() {
        lastKeptTimestamp = nil
        lastKeptPose = nil
    }

    static func translation(_ m: simd_float4x4) -> SIMD3<Float> {
        SIMD3<Float>(m.columns.3.x, m.columns.3.y, m.columns.3.z)
    }

    /// Angle (rad, in [0, π]) of the relative rotation between the rotation parts of `a` and `b`.
    /// Uses atan2(|axis|, trace − 1), which is well-conditioned for small angles.
    static func rotationAngle(_ a: simd_float4x4, _ b: simd_float4x4) -> Float {
        func r3(_ m: simd_float4x4) -> simd_double3x3 {
            simd_double3x3(columns: (
                SIMD3<Double>(Double(m[0][0]), Double(m[0][1]), Double(m[0][2])),
                SIMD3<Double>(Double(m[1][0]), Double(m[1][1]), Double(m[1][2])),
                SIMD3<Double>(Double(m[2][0]), Double(m[2][1]), Double(m[2][2]))))
        }
        let r = r3(a).transpose * r3(b)   // r[column][row]
        let trace = r[0][0] + r[1][1] + r[2][2]
        let vx = r[1][2] - r[2][1]
        let vy = r[2][0] - r[0][2]
        let vz = r[0][1] - r[1][0]
        let s = (vx * vx + vy * vy + vz * vz).squareRoot()
        return Float(atan2(s, trace - 1))
    }
}
