import Foundation
import simd

/// Pinhole intrinsics in pixels (pixel centers at integer + 0.5, OpenCV camera: +x right, +y down, +z forward).
public struct Intrinsics: Sendable, Equatable {
    public var fx: Double, fy: Double, cx: Double, cy: Double
    public var width: Int, height: Int
    public init(fx: Double, fy: Double, cx: Double, cy: Double, width: Int, height: Int) {
        self.fx = fx; self.fy = fy; self.cx = cx; self.cy = cy; self.width = width; self.height = height
    }

    /// Symmetric pinhole from a vertical field of view.
    public static func fromFov(verticalFov: Double, width: Int, height: Int) -> Intrinsics {
        let fy = Double(height) * 0.5 / tan(verticalFov * 0.5)
        return Intrinsics(fx: fy, fy: fy, cx: Double(width) * 0.5, cy: Double(height) * 0.5, width: width, height: height)
    }

    /// Fits the image into a target of `targetW × targetH` pixels preserving aspect (letterbox).
    /// Returns the scaled intrinsics expressed in target pixels plus the image rectangle (x0, y0, x1, y1).
    public func fitted(targetWidth tw: Int, targetHeight th: Int) -> (Intrinsics, SIMD4<Double>) {
        let s = min(Double(tw) / Double(width), Double(th) / Double(height))
        let ox = (Double(tw) - Double(width) * s) * 0.5
        let oy = (Double(th) - Double(height) * s) * 0.5
        let k = Intrinsics(fx: fx * s, fy: fy * s, cx: cx * s + ox, cy: cy * s + oy, width: tw, height: th)
        return (k, SIMD4(ox, oy, ox + Double(width) * s, oy + Double(height) * s))
    }
}

/// A posed OpenCV camera.
public struct CameraView: Sendable {
    public var camToWorld: RigidTransform
    public var intrinsics: Intrinsics
    public init(camToWorld: RigidTransform, intrinsics: Intrinsics) {
        self.camToWorld = camToWorld; self.intrinsics = intrinsics
    }
    public var worldToCam: RigidTransform { camToWorld.inverse }
    public var center: SIMD3<Double> { camToWorld.t }

    /// Projects a world point → (u, v, depth). Depth ≤ 0 means behind the camera.
    public func project(_ pw: SIMD3<Double>) -> SIMD3<Double> {
        let pc = worldToCam.apply(pw)
        guard pc.z > 1e-9 else { return SIMD3(.nan, .nan, pc.z) }
        return SIMD3(intrinsics.fx * pc.x / pc.z + intrinsics.cx, intrinsics.fy * pc.y / pc.z + intrinsics.cy, pc.z)
    }

    /// Ego camera: camera_to_world = ego_to_world(t) ∘ camera_to_ego.
    public static func ego(egoPose: Pose, cameraToEgo: RigidTransform, intrinsics: Intrinsics) -> CameraView {
        CameraView(camToWorld: egoPose.rigid * cameraToEgo, intrinsics: intrinsics)
    }

    /// OpenCV look-at in a z-up world.
    public static func lookAt(eye: SIMD3<Double>, target: SIMD3<Double>, up: SIMD3<Double> = [0, 0, 1]) -> RigidTransform {
        var z = target - eye
        if simd_length(z) < 1e-9 { z = [1, 0, 0] }
        z = simd_normalize(z)
        var x = simd_cross(z, up)
        if simd_length(x) < 1e-6 { x = simd_cross(z, [0, 1, 0]) }
        x = simd_normalize(x)
        let y = simd_cross(z, x) // points "down" in the image
        return RigidTransform(R: simd_double3x3(columns: (x, y, z)), t: eye)
    }
}

/// Orbit / fly camera around a target point in a z-up world.
public struct OrbitCamera: Sendable, Equatable {
    public var target: SIMD3<Double>
    public var yaw: Double      // azimuth of the eye around the target (rad, from +x toward +y)
    public var pitch: Double    // elevation of the eye above the target (rad)
    public var distance: Double
    public var verticalFov: Double = 55 * .pi / 180

    public init(target: SIMD3<Double> = [10, 0, 0], yaw: Double = .pi, pitch: Double = 0.35, distance: Double = 25) {
        self.target = target; self.yaw = yaw; self.pitch = pitch; self.distance = distance
    }

    public var eye: SIMD3<Double> {
        target + distance * SIMD3(cos(pitch) * cos(yaw), cos(pitch) * sin(yaw), sin(pitch))
    }
    public var camToWorld: RigidTransform { CameraView.lookAt(eye: eye, target: target) }

    /// Horizontal forward / left of the viewing direction (for WASD).
    public var groundForward: SIMD3<Double> { SIMD3(-cos(yaw), -sin(yaw), 0) }
    public var groundLeft: SIMD3<Double> { SIMD3(sin(yaw), -cos(yaw), 0) } // = ẑ × groundForward

    public mutating func orbit(dYaw: Double, dPitch: Double) {
        yaw += dYaw
        pitch = min(max(pitch + dPitch, -1.45), 1.52)
    }
    public mutating func zoom(factor: Double) { distance = min(max(distance * factor, 0.5), 2000) }
    /// Pan in the camera's image plane by a fraction of the distance.
    public mutating func pan(right: Double, up: Double) {
        let c = camToWorld
        let r = c.R.columns.0, d = c.R.columns.1
        target += (r * right - d * up) * distance
    }

    /// Places the orbit camera so that its eye and look direction match an existing camera.
    public static func matching(_ cam: RigidTransform, distance: Double = 12) -> OrbitCamera {
        let f = c2(cam)
        let target = cam.t + f * distance
        let back = -f
        let pitch = asin(min(max(back.z, -1), 1))
        let yaw = atan2(back.y, back.x)
        return OrbitCamera(target: target, yaw: yaw, pitch: pitch, distance: distance)
    }
    private static func c2(_ c: RigidTransform) -> SIMD3<Double> { simd_normalize(c.R.columns.2) }
}
