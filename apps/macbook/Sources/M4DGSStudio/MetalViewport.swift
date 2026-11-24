import SwiftUI
import MetalKit
import QuartzCore
import M4DGSCore
import M4DGSRender

/// SwiftUI wrapper around an MTKView that renders the scene every display frame.
struct MetalViewport: NSViewRepresentable {
    let model: AppModel

    func makeCoordinator() -> Coordinator { Coordinator(model: model) }

    func makeNSView(context: Context) -> InteractiveMTKView {
        let v = InteractiveMTKView(frame: .zero, device: model.renderer?.device)
        v.colorPixelFormat = .bgra8Unorm
        v.colorspace = CGColorSpace(name: CGColorSpace.sRGB)
        v.clearColor = MTLClearColor(red: 0.09, green: 0.09, blue: 0.1, alpha: 1)
        v.preferredFramesPerSecond = 120
        v.enableSetNeedsDisplay = false
        v.isPaused = false
        v.framebufferOnly = true
        v.delegate = context.coordinator
        v.coordinator = context.coordinator
        context.coordinator.view = v
        return v
    }

    func updateNSView(_ nsView: InteractiveMTKView, context: Context) {}

    @MainActor
    final class Coordinator: NSObject, MTKViewDelegate {
        let model: AppModel
        weak var view: InteractiveMTKView?
        var lastCamera: CameraView?
        private var frameTimes: [CFTimeInterval] = []
        private var gpuAccum: [Double] = []
        private var lastStatsUpdate: CFTimeInterval = 0
        private let inFlight = DispatchSemaphore(value: 3)

        init(model: AppModel) { self.model = model }

        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

        func draw(in view: MTKView) {
            let now = CACurrentMediaTime()
            model.tick(now: now)
            if let v = view as? InteractiveMTKView { applyKeys(v, now: now) }
            frameTimes.append(now)
            if frameTimes.count > 60 { frameTimes.removeFirst(frameTimes.count - 60) }

            guard let renderer = model.renderer else { return }
            let W = Int(view.drawableSize.width), H = Int(view.drawableSize.height)
            guard W > 0, H > 0 else { return }
            guard inFlight.wait(timeout: .now()) == .success else { return }   // GPU busy: skip this frame
            guard let drawable = view.currentDrawable else { inFlight.signal(); return }
            let cb = renderer.queue.makeCommandBuffer()!
            if var fb = model.frameBuilder() {
                if model.cameraMode == .free && model.followEgo {
                    let ego = fb.sceneFrame(at: model.time).egoPose
                    model.orbit.target = ego.pos + SIMD3(0, 0, 1)
                }
                fb.overlays.selected = model.selectedNode
                let cam: CameraChoice = model.cameraMode == .ego ? .ego : .free(model.orbit)
                var frame = fb.build(time: model.time, camera: cam, targetWidth: W, targetHeight: H)
                let scale = Float(view.window?.backingScaleFactor ?? 2)
                for i in frame.lines.indices { frame.lines[i].width *= scale }
                lastCamera = frame.camera
                do { try renderer.encodeFrame(frame, target: drawable.texture, into: cb) } catch {
                    model.errorMessage = "render error: \(error)"
                }
            } else {
                let rpd = MTLRenderPassDescriptor()
                rpd.colorAttachments[0].texture = drawable.texture
                rpd.colorAttachments[0].loadAction = .clear
                rpd.colorAttachments[0].clearColor = view.clearColor
                rpd.colorAttachments[0].storeAction = .store
                cb.makeRenderCommandEncoder(descriptor: rpd)?.endEncoding()
            }
            cb.present(drawable)
            let sem = inFlight
            cb.addCompletedHandler { [weak self] c in
                let ms = (c.gpuEndTime - c.gpuStartTime) * 1000
                let visible = renderer.visibleCount
                sem.signal()
                Task { @MainActor in self?.recordGPU(ms, visible: visible) }
            }
            cb.commit()
        }

        private func recordGPU(_ ms: Double, visible: Int) {
            gpuAccum.append(ms)
            let now = CACurrentMediaTime()
            if now - lastStatsUpdate > 0.25 {
                lastStatsUpdate = now
                model.gpuMs = percentile(gpuAccum, 50)
                model.visibleSplats = visible
                if frameTimes.count > 1 {
                    model.fps = Double(frameTimes.count - 1) / (frameTimes.last! - frameTimes.first!)
                }
                gpuAccum.removeAll(keepingCapacity: true)
            }
        }

        // MARK: input

        private var lastKeyTime: CFTimeInterval?
        func applyKeys(_ v: InteractiveMTKView, now: CFTimeInterval) {
            defer { lastKeyTime = now }
            guard let last = lastKeyTime, !v.keysDown.isEmpty else { return }
            let dt = min(now - last, 0.1)
            let fast = v.shiftDown ? 3.0 : 1.0
            let speed = max(4, model.orbit.distance * 0.8) * fast * dt
            var o = model.orbit
            var move = SIMD3<Double>(repeating: 0)
            if v.keysDown.contains("w") { move += o.groundForward }
            if v.keysDown.contains("s") { move -= o.groundForward }
            if v.keysDown.contains("a") { move += o.groundLeft }
            if v.keysDown.contains("d") { move -= o.groundLeft }
            if v.keysDown.contains("e") { move.z += 1 }
            if v.keysDown.contains("q") { move.z -= 1 }
            guard move != .zero else { return }
            if model.cameraMode == .ego { model.enterFreeCamera(); o = model.orbit }
            model.followEgo = false
            o.target += move * speed
            model.orbit = o
        }

        func drag(dx: Double, dy: Double, pan: Bool) {
            if model.cameraMode == .ego { model.enterFreeCamera() }
            if pan {
                model.followEgo = false
                model.orbit.pan(right: -dx * 0.0015, up: dy * 0.0015)
            } else {
                model.orbit.orbit(dYaw: -dx * 0.006, dPitch: -dy * 0.006)
            }
        }

        func zoom(_ factor: Double) {
            if model.cameraMode == .ego { model.enterFreeCamera() }
            model.orbit.zoom(factor: factor)
        }

        func click(pixel: SIMD2<Double>) {
            if let cam = lastCamera { model.pick(pixel: pixel, camera: cam) }
        }

        func key(_ chars: String, keyCode: UInt16) -> Bool {
            switch keyCode {
            case 49: model.playing.toggle(); return true        // space
            case 123: model.step(-1); return true               // ←
            case 124: model.step(1); return true                // →
            default: break
            }
            switch chars {
            case "1": model.cameraMode = .ego; return true
            case "2": model.enterFreeCamera(); return true
            case "f": model.followEgo.toggle(); if model.followEgo { model.cameraMode = .free }; return true
            default: return false
            }
        }
    }
}

/// MTKView that forwards mouse / keyboard input (orbit, pan, zoom, WASD fly, picking).
final class InteractiveMTKView: MTKView {
    weak var coordinator: MetalViewport.Coordinator?
    var keysDown: Set<String> = []
    var shiftDown = false
    private var dragged = false

    override var acceptsFirstResponder: Bool { true }
    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }

    override func mouseDown(with event: NSEvent) { window?.makeFirstResponder(self); dragged = false }
    override func mouseDragged(with event: NSEvent) {
        dragged = true
        let pan = event.modifierFlags.contains(.option) || event.modifierFlags.contains(.shift)
        coordinator?.drag(dx: event.deltaX, dy: event.deltaY, pan: pan)
    }
    override func mouseUp(with event: NSEvent) {
        guard !dragged else { return }
        let p = convertToBacking(convert(event.locationInWindow, from: nil))
        coordinator?.click(pixel: SIMD2(Double(p.x), Double(drawableSize.height) - Double(p.y)))
    }
    override func rightMouseDown(with event: NSEvent) { window?.makeFirstResponder(self) }
    override func rightMouseDragged(with event: NSEvent) { coordinator?.drag(dx: event.deltaX, dy: event.deltaY, pan: true) }
    override func otherMouseDragged(with event: NSEvent) { coordinator?.drag(dx: event.deltaX, dy: event.deltaY, pan: true) }
    override func scrollWheel(with event: NSEvent) {
        let d = event.hasPreciseScrollingDeltas ? event.scrollingDeltaY * 0.01 : event.scrollingDeltaY * 0.1
        coordinator?.zoom(exp(-d))
    }
    override func magnify(with event: NSEvent) { coordinator?.zoom(exp(-event.magnification)) }

    override func keyDown(with event: NSEvent) {
        shiftDown = event.modifierFlags.contains(.shift)
        let c = (event.charactersIgnoringModifiers ?? "").lowercased()
        if ["w", "a", "s", "d", "q", "e"].contains(c) { keysDown.insert(c); return }
        if event.isARepeat && event.keyCode == 49 { return }
        if coordinator?.key(c, keyCode: event.keyCode) == true { return }
        super.keyDown(with: event)
    }
    override func keyUp(with event: NSEvent) {
        keysDown.remove((event.charactersIgnoringModifiers ?? "").lowercased())
    }
    override func flagsChanged(with event: NSEvent) { shiftDown = event.modifierFlags.contains(.shift) }
    override func resignFirstResponder() -> Bool { keysDown.removeAll(); return super.resignFirstResponder() }
}
