import Foundation
import Metal
import simd
import M4DGSCore

// MARK: - GPU structs (must match Shaders.swift)

struct GPUFrameUniforms {
    var worldToCam: simd_float4x4
    var camCenter: SIMD4<Float>
    var intr: SIMD4<Float>
    var clipRect: SIMD4<Float>
    var viewport: SIMD2<Float>
    var time: Float
    var nearZ: Float
    var numGaussians: UInt32
    var shDegree: UInt32
    var shStride: UInt32
    var lowPass: Float
    var fill: SIMD4<Float>
}

struct GPUNodePose {
    var c0: SIMD4<Float>, c1: SIMD4<Float>, c2: SIMD4<Float>
    var pos: SIMD4<Float>
    var quat: SIMD4<Float>
}

struct GPULineSeg {
    var a: SIMD4<Float>, b: SIMD4<Float>, color: SIMD4<Float>
}

public enum RendererError: Error, CustomStringConvertible {
    case noDevice, shaderCompile(String), pipeline(String), unsupported(String)
    public var description: String {
        switch self {
        case .noDevice: return "no Metal device"
        case .shaderCompile(let s): return "shader compile failed: \(s)"
        case .pipeline(let s): return "pipeline creation failed: \(s)"
        case .unsupported(let s): return "unsupported GPU: \(s)"
        }
    }
}

/// Everything needed to render one frame.
public struct RenderFrame {
    /// Camera with intrinsics expressed in *target* pixels.
    public var camera: CameraView
    /// Target-pixel rectangle (x0, y0, x1, y1) where the scene is drawn (letterbox for the ego camera).
    public var imageRect: SIMD4<Double>
    public var time: Double
    public var nodePoses: [Pose]
    public var nodeVisible: [Bool]
    public var lines: [LineSegment] = []
    public var background: SIMD3<Double> = .zero
    public var letterbox: SIMD3<Double> = [0.09, 0.09, 0.1]
    public var nearZ: Double = 0.1
    public var lowPass: Double = 0.3

    public init(camera: CameraView, imageRect: SIMD4<Double>, time: Double, nodePoses: [Pose], nodeVisible: [Bool]) {
        self.camera = camera; self.imageRect = imageRect; self.time = time
        self.nodePoses = nodePoses; self.nodeVisible = nodeVisible
    }
}

public struct GPUTimings {
    public var totalMs: [Double] = []
    public var preprocessMs: [Double] = []
    public var sortMs: [Double] = []
    public var rasterMs: [Double] = []
}

public func percentile(_ xs: [Double], _ p: Double) -> Double {
    guard !xs.isEmpty else { return .nan }
    let s = xs.sorted()
    let idx = min(s.count - 1, max(0, Int((p / 100 * Double(s.count - 1)).rounded())))
    return s[idx]
}

/// Metal 4DGS splat renderer: compute preprocess → GPU radix sort → instanced quad raster.
public final class SplatRenderer: @unchecked Sendable {
    public let device: MTLDevice
    public let queue: MTLCommandQueue
    let library: MTLLibrary
    let preprocessPSO: MTLComputePipelineState
    let histPSO: MTLComputePipelineState
    let scanPSO: MTLComputePipelineState
    let scatterPSO: MTLComputePipelineState
    let argsPSO: MTLComputePipelineState
    private var splatPSOs: [MTLPixelFormat: MTLRenderPipelineState] = [:]
    private var linePSOs: [MTLPixelFormat: MTLRenderPipelineState] = [:]
    private var fillPSOs: [MTLPixelFormat: MTLRenderPipelineState] = [:]
    let scanThreads: Int

    // scene
    public private(set) var gaussianCount = 0
    public private(set) var shDegree = 0
    private var shStride = 3
    private var gaussBuf: MTLBuffer!
    private var shBuf: MTLBuffer!
    private var nodeIdBuf: MTLBuffer!
    // per-frame GPU scratch
    private var splatBuf: MTLBuffer!
    private var keysA: MTLBuffer!, valsA: MTLBuffer!, keysB: MTLBuffer!, valsB: MTLBuffer!
    private var histBuf: MTLBuffer!
    private var quadIndexBuf: MTLBuffer!
    private let counterBuf: MTLBuffer
    private let argsBuf: MTLBuffer
    private var maxTiles = 1

    public init(device: MTLDevice? = nil) throws {
        guard let dev = device ?? MTLCreateSystemDefaultDevice() else { throw RendererError.noDevice }
        self.device = dev
        guard let q = dev.makeCommandQueue() else { throw RendererError.noDevice }
        queue = q
        let lib: MTLLibrary
        do {
            let opts = MTLCompileOptions()
            opts.mathMode = .fast
            lib = try dev.makeLibrary(source: ShaderSource.source, options: opts)
        } catch {
            throw RendererError.shaderCompile("\(error)")
        }
        library = lib
        func cps(_ name: String) throws -> MTLComputePipelineState {
            guard let f = lib.makeFunction(name: name) else { throw RendererError.pipeline("missing \(name)") }
            return try dev.makeComputePipelineState(function: f)
        }
        preprocessPSO = try cps("preprocess")
        histPSO = try cps("radix_hist")
        scanPSO = try cps("radix_scan")
        scatterPSO = try cps("radix_scatter")
        argsPSO = try cps("make_draw_args")
        guard scatterPSO.threadExecutionWidth == 32 else {
            throw RendererError.unsupported("radix sort assumes 32-wide SIMD groups (got \(scatterPSO.threadExecutionWidth))")
        }
        guard scatterPSO.maxTotalThreadsPerThreadgroup >= ShaderSource.sortThreadgroup,
              histPSO.maxTotalThreadsPerThreadgroup >= ShaderSource.sortThreadgroup else {
            throw RendererError.unsupported("sort kernels need \(ShaderSource.sortThreadgroup) threads per threadgroup")
        }
        scanThreads = min(1024, scanPSO.maxTotalThreadsPerThreadgroup) / 32 * 32
        counterBuf = dev.makeBuffer(length: 16, options: .storageModeShared)!
        argsBuf = dev.makeBuffer(length: 32, options: .storageModeShared)!
        precondition(MemoryLayout<GPUFrameUniforms>.size == 160, "GPUFrameUniforms layout mismatch")
        precondition(MemoryLayout<GPUNodePose>.stride == 80 && MemoryLayout<GPULineSeg>.stride == 48)
        try setScene(GaussianCloud(count: 0, shDegree: 0, packed: [], sh: [], nodeIds: [], nodeRanges: [:]))
    }

    private func buffer<T>(_ a: [T], minLength: Int = 16) -> MTLBuffer {
        let len = max(a.count * MemoryLayout<T>.stride, minLength)
        if a.isEmpty { return device.makeBuffer(length: len, options: .storageModeShared)! }
        return a.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! }
    }

    /// Uploads a Gaussian cloud and (re)allocates the per-frame scratch buffers.
    public func setScene(_ cloud: GaussianCloud) throws {
        gaussianCount = cloud.count
        shDegree = cloud.shDegree
        shStride = cloud.shStride
        gaussBuf = buffer(cloud.packed)
        shBuf = buffer(cloud.sh)
        nodeIdBuf = buffer(cloud.nodeIds)
        let n = max(cloud.count, 1)
        splatBuf = device.makeBuffer(length: n * 48, options: .storageModePrivate)
        keysA = device.makeBuffer(length: n * 4, options: .storageModeShared)
        valsA = device.makeBuffer(length: n * 4, options: .storageModeShared)
        keysB = device.makeBuffer(length: n * 4, options: .storageModePrivate)
        valsB = device.makeBuffer(length: n * 4, options: .storageModePrivate)
        maxTiles = max(1, (n + ShaderSource.sortTile - 1) / ShaderSource.sortTile)
        histBuf = device.makeBuffer(length: 256 * maxTiles * 4, options: .storageModePrivate)
        // two triangles per splat: (0,1,2) (2,1,3) offset by 4k
        var idx = [UInt32](repeating: 0, count: n * 6)
        for k in 0..<n {
            let b = UInt32(4 * k)
            idx[6 * k] = b; idx[6 * k + 1] = b + 1; idx[6 * k + 2] = b + 2
            idx[6 * k + 3] = b + 2; idx[6 * k + 4] = b + 1; idx[6 * k + 5] = b + 3
        }
        quadIndexBuf = buffer(idx)
        guard splatBuf != nil, keysA != nil, valsA != nil, keysB != nil, valsB != nil, histBuf != nil else {
            throw RendererError.pipeline("buffer allocation failed for \(n) Gaussians")
        }
    }

    /// Visible (post-cull) splat count of the most recently *completed* frame.
    public var visibleCount: Int { Int(counterBuf.contents().load(as: UInt32.self)) }

    /// Sorted keys of the last completed frame (debug / tests).
    public func sortedKeys() -> [UInt32] {
        let n = visibleCount
        let p = keysA.contents().bindMemory(to: UInt32.self, capacity: n)
        return Array(UnsafeBufferPointer(start: p, count: n))
    }

    private func uniforms(_ f: RenderFrame, viewport: SIMD2<Float>) -> GPUFrameUniforms {
        let k = f.camera.intrinsics
        let w2c = f.camera.worldToCam.matrix4x4
        let m = simd_float4x4(columns: (SIMD4<Float>(w2c.columns.0), SIMD4<Float>(w2c.columns.1),
                                        SIMD4<Float>(w2c.columns.2), SIMD4<Float>(w2c.columns.3)))
        return GPUFrameUniforms(worldToCam: m,
                                camCenter: SIMD4<Float>(SIMD3<Float>(f.camera.center), 1),
                                intr: SIMD4<Float>(Float(k.fx), Float(k.fy), Float(k.cx), Float(k.cy)),
                                clipRect: SIMD4<Float>(f.imageRect),
                                viewport: viewport,
                                time: Float(f.time),
                                nearZ: Float(f.nearZ),
                                numGaussians: UInt32(gaussianCount),
                                shDegree: UInt32(shDegree),
                                shStride: UInt32(shStride),
                                lowPass: Float(f.lowPass),
                                fill: SIMD4<Float>(SIMD3<Float>(f.background), 1))
    }

    private func nodeBuffer(_ f: RenderFrame) -> MTLBuffer {
        var nodes: [GPUNodePose] = f.nodePoses.enumerated().map { i, p in
            let R = p.quat.normalized.matrix
            let q = p.quat.normalized
            let vis: Float = (i < f.nodeVisible.count ? f.nodeVisible[i] : true) ? 1 : 0
            return GPUNodePose(c0: SIMD4<Float>(SIMD3<Float>(R.columns.0), 0),
                               c1: SIMD4<Float>(SIMD3<Float>(R.columns.1), 0),
                               c2: SIMD4<Float>(SIMD3<Float>(R.columns.2), 0),
                               pos: SIMD4<Float>(SIMD3<Float>(p.pos), vis),
                               quat: SIMD4<Float>(Float(q.w), Float(q.x), Float(q.y), Float(q.z)))
        }
        if nodes.isEmpty {
            nodes = [GPUNodePose(c0: [1, 0, 0, 0], c1: [0, 1, 0, 0], c2: [0, 0, 1, 0], pos: [0, 0, 0, 1], quat: [1, 0, 0, 0])]
        }
        return buffer(nodes)
    }

    // MARK: encoding

    /// Stage 1: preprocess (also resets the visible counter).
    public func encodePreprocess(_ f: RenderFrame, viewport: SIMD2<Float>, into cb: MTLCommandBuffer) {
        if let blit = cb.makeBlitCommandEncoder() {
            blit.label = "reset counter"
            blit.fill(buffer: counterBuf, range: 0..<16, value: 0)
            blit.endEncoding()
        }
        guard gaussianCount > 0, let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = "preprocess"
        var u = uniforms(f, viewport: viewport)
        enc.setComputePipelineState(preprocessPSO)
        enc.setBuffer(gaussBuf, offset: 0, index: 0)
        enc.setBuffer(shBuf, offset: 0, index: 1)
        enc.setBuffer(nodeIdBuf, offset: 0, index: 2)
        enc.setBuffer(nodeBuffer(f), offset: 0, index: 3)
        enc.setBytes(&u, length: MemoryLayout<GPUFrameUniforms>.stride, index: 4)
        enc.setBuffer(splatBuf, offset: 0, index: 5)
        enc.setBuffer(keysA, offset: 0, index: 6)
        enc.setBuffer(valsA, offset: 0, index: 7)
        enc.setBuffer(counterBuf, offset: 0, index: 8)
        let tg = 256
        enc.dispatchThreadgroups(MTLSize(width: (gaussianCount + tg - 1) / tg, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Stage 2: 4-pass radix sort of (key, index) pairs in keysA/valsA, plus indirect draw args.
    public func encodeSort(into cb: MTLCommandBuffer) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = "radix sort"
        let tgs = MTLSize(width: maxTiles, height: 1, depth: 1)
        let tpg = MTLSize(width: ShaderSource.sortThreadgroup, height: 1, depth: 1)
        if gaussianCount > 0 {
            var src = (keysA!, valsA!), dst = (keysB!, valsB!)
            for pass in 0..<4 {
                var shift = UInt32(pass * 8)
                enc.setComputePipelineState(histPSO)
                enc.setBuffer(src.0, offset: 0, index: 0)
                enc.setBuffer(histBuf, offset: 0, index: 1)
                enc.setBuffer(counterBuf, offset: 0, index: 2)
                enc.setBytes(&shift, length: 4, index: 3)
                enc.dispatchThreadgroups(tgs, threadsPerThreadgroup: tpg)
                enc.memoryBarrier(scope: .buffers)

                enc.setComputePipelineState(scanPSO)
                enc.setBuffer(histBuf, offset: 0, index: 0)
                enc.setBuffer(counterBuf, offset: 0, index: 1)
                enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: scanThreads, height: 1, depth: 1))
                enc.memoryBarrier(scope: .buffers)

                enc.setComputePipelineState(scatterPSO)
                enc.setBuffer(src.0, offset: 0, index: 0)
                enc.setBuffer(src.1, offset: 0, index: 1)
                enc.setBuffer(dst.0, offset: 0, index: 2)
                enc.setBuffer(dst.1, offset: 0, index: 3)
                enc.setBuffer(histBuf, offset: 0, index: 4)
                enc.setBuffer(counterBuf, offset: 0, index: 5)
                enc.setBytes(&shift, length: 4, index: 6)
                enc.dispatchThreadgroups(tgs, threadsPerThreadgroup: tpg)
                enc.memoryBarrier(scope: .buffers)
                swap(&src, &dst)
            }
        }
        enc.setComputePipelineState(argsPSO)
        enc.setBuffer(counterBuf, offset: 0, index: 0)
        enc.setBuffer(argsBuf, offset: 0, index: 1)
        enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        enc.endEncoding()
    }

    private func renderPSO(_ cache: inout [MTLPixelFormat: MTLRenderPipelineState], _ vs: String, _ fs: String,
                           _ format: MTLPixelFormat) throws -> MTLRenderPipelineState {
        if let p = cache[format] { return p }
        let d = MTLRenderPipelineDescriptor()
        d.vertexFunction = library.makeFunction(name: vs)
        d.fragmentFunction = library.makeFunction(name: fs)
        let ca = d.colorAttachments[0]!
        ca.pixelFormat = format
        ca.isBlendingEnabled = true
        ca.rgbBlendOperation = .add; ca.alphaBlendOperation = .add
        ca.sourceRGBBlendFactor = .one; ca.sourceAlphaBlendFactor = .one
        ca.destinationRGBBlendFactor = .oneMinusSourceAlpha; ca.destinationAlphaBlendFactor = .oneMinusSourceAlpha
        let p = try device.makeRenderPipelineState(descriptor: d)
        cache[format] = p
        return p
    }

    /// Stage 3: raster splats (back-to-front, premultiplied) and overlay lines into `target`.
    public func encodeRaster(_ f: RenderFrame, target: MTLTexture, into cb: MTLCommandBuffer) throws {
        let fmt = target.pixelFormat
        let splatPSO = try renderPSO(&splatPSOs, "splat_vs", "splat_fs", fmt)
        let linePSO = try renderPSO(&linePSOs, "line_vs", "line_fs", fmt)
        let fillPSO = try renderPSO(&fillPSOs, "fill_vs", "fill_fs", fmt)
        let W = target.width, H = target.height
        let r = f.imageRect
        let fullImage = r.x <= 0.5 && r.y <= 0.5 && r.z >= Double(W) - 0.5 && r.w >= Double(H) - 0.5
        let rpd = MTLRenderPassDescriptor()
        rpd.colorAttachments[0].texture = target
        rpd.colorAttachments[0].loadAction = .clear
        rpd.colorAttachments[0].storeAction = .store
        let cc = fullImage ? f.background : f.letterbox
        rpd.colorAttachments[0].clearColor = MTLClearColor(red: cc.x, green: cc.y, blue: cc.z, alpha: 1)
        guard let enc = cb.makeRenderCommandEncoder(descriptor: rpd) else { return }
        enc.label = "splat raster"
        var u = uniforms(f, viewport: SIMD2(Float(W), Float(H)))
        let x0 = max(0, min(W - 1, Int(r.x.rounded(.down)))), y0 = max(0, min(H - 1, Int(r.y.rounded(.down))))
        let x1 = max(x0 + 1, min(W, Int(r.z.rounded(.up)))), y1 = max(y0 + 1, min(H, Int(r.w.rounded(.up))))
        enc.setScissorRect(MTLScissorRect(x: x0, y: y0, width: x1 - x0, height: y1 - y0))
        if !fullImage {
            enc.setRenderPipelineState(fillPSO)
            enc.setFragmentBytes(&u, length: MemoryLayout<GPUFrameUniforms>.stride, index: 0)
            enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        }
        if gaussianCount > 0 {
            enc.setRenderPipelineState(splatPSO)
            enc.setVertexBuffer(splatBuf, offset: 0, index: 0)
            enc.setVertexBuffer(valsA, offset: 0, index: 1)
            enc.setVertexBytes(&u, length: MemoryLayout<GPUFrameUniforms>.stride, index: 2)
            enc.drawIndexedPrimitives(type: .triangle, indexType: .uint32, indexBuffer: quadIndexBuf, indexBufferOffset: 0,
                                      indirectBuffer: argsBuf, indirectBufferOffset: 0)
        }
        if !f.lines.isEmpty {
            let segs = f.lines.map {
                GPULineSeg(a: SIMD4($0.a, $0.width), b: SIMD4($0.b, 0), color: $0.color)
            }
            enc.setRenderPipelineState(linePSO)
            enc.setVertexBuffer(buffer(segs), offset: 0, index: 0)
            enc.setVertexBytes(&u, length: MemoryLayout<GPUFrameUniforms>.stride, index: 1)
            enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: segs.count)
        }
        enc.endEncoding()
    }

    /// Encodes a full frame into one command buffer.
    public func encodeFrame(_ f: RenderFrame, target: MTLTexture, into cb: MTLCommandBuffer) throws {
        encodePreprocess(f, viewport: SIMD2(Float(target.width), Float(target.height)), into: cb)
        encodeSort(into: cb)
        try encodeRaster(f, target: target, into: cb)
    }

    // MARK: offscreen helpers

    public func makeTarget(width: Int, height: Int, format: MTLPixelFormat = .rgba16Float) -> MTLTexture {
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: width, height: height, mipmapped: false)
        d.usage = [.renderTarget, .shaderRead]
        d.storageMode = .private
        return device.makeTexture(descriptor: d)!
    }

    /// Renders one frame synchronously; returns GPU time in ms.
    @discardableResult
    public func renderSync(_ f: RenderFrame, target: MTLTexture) throws -> Double {
        let cb = queue.makeCommandBuffer()!
        try encodeFrame(f, target: target, into: cb)
        cb.commit()
        cb.waitUntilCompleted()
        if let e = cb.error { throw e }
        return (cb.gpuEndTime - cb.gpuStartTime) * 1000
    }

    /// Benchmarks `iterations` frames: whole-frame GPU time from one command buffer per frame, plus a
    /// per-stage breakdown from frames split into three command buffers.
    public func benchmark(_ f: RenderFrame, target: MTLTexture, iterations: Int, warmup: Int = 10) throws -> GPUTimings {
        var t = GPUTimings()
        for _ in 0..<warmup { try renderSync(f, target: target) }
        for _ in 0..<iterations { t.totalMs.append(try renderSync(f, target: target)) }
        let vp = SIMD2(Float(target.width), Float(target.height))
        for _ in 0..<max(1, iterations / 2) {
            let c1 = queue.makeCommandBuffer()!, c2 = queue.makeCommandBuffer()!, c3 = queue.makeCommandBuffer()!
            encodePreprocess(f, viewport: vp, into: c1)
            encodeSort(into: c2)
            try encodeRaster(f, target: target, into: c3)
            c1.commit(); c2.commit(); c3.commit()
            c3.waitUntilCompleted()
            t.preprocessMs.append((c1.gpuEndTime - c1.gpuStartTime) * 1000)
            t.sortMs.append((c2.gpuEndTime - c2.gpuStartTime) * 1000)
            t.rasterMs.append((c3.gpuEndTime - c3.gpuStartTime) * 1000)
        }
        return t
    }

    /// Sustained throughput: `iterations` frames submitted back-to-back (≤ `inFlight` queued, as in the
    /// interactive viewer). Returns mean GPU ms per frame = (last gpuEndTime − first gpuStartTime) / N.
    /// (Per-command-buffer spans overlap when queued, so they are not used here.)
    public func benchmarkSustained(_ f: RenderFrame, target: MTLTexture, iterations: Int, inFlight: Int = 3) throws -> Double {
        let sem = DispatchSemaphore(value: inFlight)
        var cbs: [MTLCommandBuffer] = []
        for _ in 0..<10 { try renderSync(f, target: target) }
        for _ in 0..<iterations {
            sem.wait()
            let cb = queue.makeCommandBuffer()!
            try encodeFrame(f, target: target, into: cb)
            cb.addCompletedHandler { _ in sem.signal() }
            cb.commit()
            cbs.append(cb)
        }
        cbs.last?.waitUntilCompleted()
        guard let first = cbs.first, let last = cbs.last else { return .nan }
        return (last.gpuEndTime - first.gpuStartTime) * 1000 / Double(iterations)
    }

    /// Reads back an rgba16Float / rgba8 / bgra8 texture as 8-bit RGBA rows (top to bottom).
    public func readRGBA8(_ tex: MTLTexture) -> [UInt8] {
        let W = tex.width, H = tex.height
        let bpp = tex.pixelFormat == .rgba16Float ? 8 : 4
        let buf = device.makeBuffer(length: W * H * bpp, options: .storageModeShared)!
        let cb = queue.makeCommandBuffer()!
        let blit = cb.makeBlitCommandEncoder()!
        blit.copy(from: tex, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                  sourceSize: MTLSize(width: W, height: H, depth: 1), to: buf, destinationOffset: 0,
                  destinationBytesPerRow: W * bpp, destinationBytesPerImage: W * H * bpp)
        blit.endEncoding()
        cb.commit(); cb.waitUntilCompleted()
        var out = [UInt8](repeating: 255, count: W * H * 4)
        if bpp == 8 {
            let p = buf.contents().bindMemory(to: Float16.self, capacity: W * H * 4)
            for i in 0..<(W * H * 4) {
                out[i] = UInt8(max(0, min(255, (Float(p[i]) * 255).rounded())))
            }
        } else {
            let p = buf.contents().bindMemory(to: UInt8.self, capacity: W * H * 4)
            let bgra = tex.pixelFormat == .bgra8Unorm || tex.pixelFormat == .bgra8Unorm_srgb
            for i in 0..<(W * H) {
                out[i * 4 + 0] = p[i * 4 + (bgra ? 2 : 0)]
                out[i * 4 + 1] = p[i * 4 + 1]
                out[i * 4 + 2] = p[i * 4 + (bgra ? 0 : 2)]
                out[i * 4 + 3] = p[i * 4 + 3]
            }
        }
        return out
    }
}
