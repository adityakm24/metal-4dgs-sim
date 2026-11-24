import Foundation

/// Persistent settings (UserDefaults, domain `dev.m4dgs.studio` for both the .app and the bare SwiftPM binary).
/// Unset / invalid values fall back to detected defaults.
enum StudioSettings {
    static let domain = "dev.m4dgs.studio"
    static let repoRootKey = "repoRoot"
    static let capturesFolderKey = "capturesFolder"
    static let iterationsKey = "reconstructIterations"
    static let defaultIterations = 7000

    static var defaults: UserDefaults {
        if Bundle.main.bundleIdentifier == domain { return .standard }
        return UserDefaults(suiteName: domain) ?? .standard
    }

    static func isRepoRoot(_ url: URL) -> Bool {
        FileManager.default.fileExists(atPath: url.appendingPathComponent("scripts/train_4dgs.py").path)
    }

    /// Nearest ancestor of the executable, then of the current directory, containing `scripts/train_4dgs.py`.
    static func detectRepoRoot() -> URL? {
        var starts: [URL] = []
        if let exe = Bundle.main.executableURL { starts.append(exe.resolvingSymlinksInPath().deletingLastPathComponent()) }
        starts.append(URL(fileURLWithPath: FileManager.default.currentDirectoryPath))
        for s in starts {
            var u = s.standardizedFileURL
            while true {
                if isRepoRoot(u) { return u }
                let p = u.deletingLastPathComponent()
                if p.path == u.path { break }
                u = p
            }
        }
        return nil
    }

    /// The stored repo root if it is valid, else the detected one.
    static var repoRoot: URL? {
        if let s = defaults.string(forKey: repoRootKey), !s.isEmpty {
            let u = URL(fileURLWithPath: (s as NSString).expandingTildeInPath)
            if isRepoRoot(u) { return u }
        }
        return detectRepoRoot()
    }

    static func defaultCapturesFolder(repo: URL?) -> URL {
        if let r = repo { return r.appendingPathComponent("data/live", isDirectory: true) }
        return FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Documents/M4DGS Captures", isDirectory: true)
    }

    static var capturesFolder: URL {
        if let s = defaults.string(forKey: capturesFolderKey), !s.isEmpty {
            return URL(fileURLWithPath: (s as NSString).expandingTildeInPath, isDirectory: true)
        }
        return defaultCapturesFolder(repo: repoRoot)
    }

    static var iterations: Int {
        let n = defaults.integer(forKey: iterationsKey)
        return n > 0 ? n : defaultIterations
    }

    /// Stores a path, or removes the key (→ default) when empty.
    static func set(_ key: String, path: String?) {
        if let p = path, !p.isEmpty { defaults.set(p, forKey: key) } else { defaults.removeObject(forKey: key) }
    }
}
