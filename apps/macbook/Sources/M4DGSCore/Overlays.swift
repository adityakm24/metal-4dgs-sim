import Foundation
import simd

/// A world-space line segment drawn as a screen-space quad of `width` pixels.
public struct LineSegment: Sendable {
    public var a: SIMD3<Float>
    public var b: SIMD3<Float>
    public var color: SIMD4<Float>
    public var width: Float
    public init(_ a: SIMD3<Double>, _ b: SIMD3<Double>, color: SIMD4<Float>, width: Float = 2) {
        self.a = SIMD3<Float>(a); self.b = SIMD3<Float>(b); self.color = color; self.width = width
    }
}

public enum Overlay {
    /// 12 edges of an oriented box (size l,w,h centred at the node origin) + a cross on the front face.
    public static func box(pose: Pose, size: SIMD3<Double>, color: SIMD4<Float>, width: Float = 2) -> [LineSegment] {
        let h = size * 0.5
        var c: [SIMD3<Double>] = []
        for i in 0..<8 {
            let s = SIMD3<Double>(i & 1 == 0 ? -1 : 1, i & 2 == 0 ? -1 : 1, i & 4 == 0 ? -1 : 1)
            c.append(pose.apply(s * h))
        }
        let edges = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
        var out = edges.map { LineSegment(c[$0.0], c[$0.1], color: color, width: width) }
        // front face is +x: corners 1,3,5,7
        out.append(LineSegment(c[1], c[7], color: color, width: width))
        out.append(LineSegment(c[3], c[5], color: color, width: width))
        return out
    }

    public static func polyline(_ pts: [SIMD3<Double>], color: SIMD4<Float>, width: Float = 2) -> [LineSegment] {
        guard pts.count > 1 else { return [] }
        return (1..<pts.count).map { LineSegment(pts[$0 - 1], pts[$0], color: color, width: width) }
    }

    /// Track drawn on the ground under the node (z lowered by `drop`).
    public static func track(_ keys: [TrackKey], drop: Double, color: SIMD4<Float>, width: Float = 2) -> [LineSegment] {
        polyline(keys.map { $0.pos - SIMD3(0, 0, drop) }, color: color, width: width)
    }

    /// Camera frustum: centre to the four image corners at `depth` metres.
    public static func frustum(_ cam: CameraView, depth: Double, color: SIMD4<Float>, width: Float = 1.5) -> [LineSegment] {
        let k = cam.intrinsics
        let corners = [(0.0, 0.0), (Double(k.width), 0.0), (Double(k.width), Double(k.height)), (0.0, Double(k.height))].map { (u, v) in
            cam.camToWorld.apply(SIMD3((u - k.cx) / k.fx * depth, (v - k.cy) / k.fy * depth, depth))
        }
        var out = corners.map { LineSegment(cam.center, $0, color: color, width: width) }
        for i in 0..<4 { out.append(LineSegment(corners[i], corners[(i + 1) % 4], color: color, width: width)) }
        return out
    }
}
