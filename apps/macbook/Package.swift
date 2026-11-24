// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "M4DGSStudio",
    platforms: [.macOS(.v15)],
    products: [
        .executable(name: "M4DGSStudio", targets: ["M4DGSStudio"]),
        .library(name: "M4DGSCore", targets: ["M4DGSCore"]),
        .library(name: "M4DGSRender", targets: ["M4DGSRender"]),
    ],
    targets: [
        // Pure Swift: safetensors, scene graph, track math, SH / 4D evaluation, edits, rollouts.
        .target(name: "M4DGSCore"),
        // Metal renderer (shaders compiled at runtime from embedded source).
        .target(name: "M4DGSRender", dependencies: ["M4DGSCore"]),
        // SwiftUI app + headless CLI.
        .executableTarget(name: "M4DGSStudio", dependencies: ["M4DGSCore", "M4DGSRender"]),
        .testTarget(name: "M4DGSCoreTests", dependencies: ["M4DGSCore"]),
        .testTarget(name: "M4DGSRenderTests", dependencies: ["M4DGSCore", "M4DGSRender"]),
    ],
    swiftLanguageModes: [.v5]
)
