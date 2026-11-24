import SwiftUI

struct SettingsView: View {
    let hasLiDAR: Bool
    let arSupported: Bool

    @AppStorage(SettingsKey.minInterval) private var minInterval = SettingsDefault.minInterval
    @AppStorage(SettingsKey.minTranslationCM) private var minTranslationCM = SettingsDefault.minTranslationCM
    @AppStorage(SettingsKey.minRotationDeg) private var minRotationDeg = SettingsDefault.minRotationDeg
    @AppStorage(SettingsKey.maxAngularSpeed) private var maxAngularSpeed = SettingsDefault.maxAngularSpeed
    @AppStorage(SettingsKey.halfResolution) private var halfResolution = SettingsDefault.halfResolution
    @AppStorage(SettingsKey.autoFocus) private var autoFocus = SettingsDefault.autoFocus
    @AppStorage(SettingsKey.showFeaturePoints) private var showFeaturePoints = SettingsDefault.showFeaturePoints
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        Form {
            Section {
                slider("Min interval", value: $minInterval, range: 0...1, step: 0.05, text: String(format: "%.2f s", minInterval))
                slider("Min translation", value: $minTranslationCM, range: 1...30, step: 1, text: String(format: "%.0f cm", minTranslationCM))
                slider("Min rotation", value: $minRotationDeg, range: 1...30, step: 1, text: String(format: "%.0f°", minRotationDeg))
                slider("Max angular speed", value: $maxAngularSpeed, range: 0.2...4, step: 0.1, text: String(format: "%.1f rad/s", maxAngularSpeed))
            } header: {
                Text("Keyframes")
            } footer: {
                Text("A frame is kept when at least the min interval has passed since the last keyframe AND the camera moved the min translation OR turned the min rotation. Frames are rejected while tracking isn't normal or when the angular speed exceeds the max (motion blur).")
            }

            Section {
                Toggle("Half resolution", isOn: $halfResolution)
                Toggle("Autofocus", isOn: $autoFocus)
                Toggle("Show feature points", isOn: $showFeaturePoints)
            } header: {
                Text("Camera")
            } footer: {
                Text("Half resolution stores 960×720 JPEGs (intrinsics are scaled to match). Turning autofocus off keeps the focal length steadier across frames.")
            }

            Section("Device") {
                LabeledContent("ARKit world tracking", value: arSupported ? "Yes" : "No")
                LabeledContent("LiDAR depth", value: hasLiDAR ? "Yes" : "No")
                LabeledContent("Model", value: DeviceInfo.modelIdentifier)
            }

            Section {
                Button("Reset to defaults") {
                    minInterval = SettingsDefault.minInterval
                    minTranslationCM = SettingsDefault.minTranslationCM
                    minRotationDeg = SettingsDefault.minRotationDeg
                    maxAngularSpeed = SettingsDefault.maxAngularSpeed
                    halfResolution = SettingsDefault.halfResolution
                    autoFocus = SettingsDefault.autoFocus
                    showFeaturePoints = SettingsDefault.showFeaturePoints
                }
            }
        }
        .navigationTitle("Settings")
        .toolbar {
            ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } }
        }
    }

    private func slider(_ title: String, value: Binding<Double>, range: ClosedRange<Double>, step: Double, text: String) -> some View {
        VStack(alignment: .leading) {
            LabeledContent(title, value: text)
            Slider(value: value, in: range, step: step)
        }
    }
}
