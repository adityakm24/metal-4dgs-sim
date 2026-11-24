import SwiftUI

@main
struct M4DGSCaptureApp: App {
    init() {
        UserDefaults.standard.register(defaults: SettingsDefault.registration)
    }

    var body: some Scene {
        WindowGroup {
            CaptureScreen()
                .preferredColorScheme(.dark)
        }
    }
}
