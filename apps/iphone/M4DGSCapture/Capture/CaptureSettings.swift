import Foundation

/// UserDefaults keys shared by `SettingsView` (@AppStorage) and `CaptureSettings.load()`.
enum SettingsKey {
    static let minInterval = "keyframe.minInterval"            // s
    static let minTranslationCM = "keyframe.minTranslationCM"  // cm
    static let minRotationDeg = "keyframe.minRotationDeg"      // degrees
    static let maxAngularSpeed = "keyframe.maxAngularSpeed"    // rad/s
    static let halfResolution = "image.halfResolution"
    static let autoFocus = "camera.autoFocus"
    static let showFeaturePoints = "preview.showFeaturePoints"
}

enum SettingsDefault {
    static let minInterval = 0.1
    static let minTranslationCM = 5.0
    static let minRotationDeg = 5.0
    static let maxAngularSpeed = 1.0
    static let halfResolution = false
    static let autoFocus = true
    static let showFeaturePoints = true

    static var registration: [String: Any] {
        [SettingsKey.minInterval: minInterval,
         SettingsKey.minTranslationCM: minTranslationCM,
         SettingsKey.minRotationDeg: minRotationDeg,
         SettingsKey.maxAngularSpeed: maxAngularSpeed,
         SettingsKey.halfResolution: halfResolution,
         SettingsKey.autoFocus: autoFocus,
         SettingsKey.showFeaturePoints: showFeaturePoints]
    }
}

struct CaptureSettings: Sendable, Equatable {
    var keyframe = KeyframeSettings()
    var halfResolution = false
    var jpegQuality = 0.9
    var autoFocus = true

    static func load(from d: UserDefaults = .standard) -> CaptureSettings {
        d.register(defaults: SettingsDefault.registration)
        var s = CaptureSettings()
        s.keyframe.minInterval = max(0, d.double(forKey: SettingsKey.minInterval))
        s.keyframe.minTranslation = Float(max(0, d.double(forKey: SettingsKey.minTranslationCM)) / 100)
        s.keyframe.minRotationDegrees = Float(max(0, d.double(forKey: SettingsKey.minRotationDeg)))
        s.keyframe.maxAngularSpeed = Float(max(0.05, d.double(forKey: SettingsKey.maxAngularSpeed)))
        s.halfResolution = d.bool(forKey: SettingsKey.halfResolution)
        s.autoFocus = d.bool(forKey: SettingsKey.autoFocus)
        return s
    }
}

/// Device/session metadata recorded in capture.json.
struct CaptureMetadata: Sendable {
    var deviceModel: String
    var systemVersion: String
    var createdAt: String
}

enum DeviceInfo {
    /// Hardware model identifier, e.g. "iPhone17,3".
    static var modelIdentifier: String {
        if let sim = ProcessInfo.processInfo.environment["SIMULATOR_MODEL_IDENTIFIER"] { return sim }
        var info = utsname()
        uname(&info)
        return withUnsafeBytes(of: info.machine) { raw in
            String(decoding: raw.prefix { $0 != 0 }, as: UTF8.self)
        }
    }
}
