import ARKit
import SceneKit
import SwiftUI

/// Live camera preview driven by the capture engine's ARSession (the view does not own the session).
struct ARPreview: UIViewRepresentable {
    let session: ARSession
    var showFeaturePoints: Bool

    func makeUIView(context: Context) -> ARSCNView {
        let view = ARSCNView(frame: .zero)
        view.session = session
        view.scene = SCNScene()
        view.automaticallyUpdatesLighting = false
        view.rendersCameraGrain = false
        view.debugOptions = showFeaturePoints ? [.showFeaturePoints] : []
        return view
    }

    func updateUIView(_ view: ARSCNView, context: Context) {
        if view.session !== session { view.session = session }
        view.debugOptions = showFeaturePoints ? [.showFeaturePoints] : []
    }
}
