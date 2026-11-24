import Foundation
import Observation

struct CaptureItem: Identifiable, Hashable, Sendable {
    var url: URL
    var id: URL { url }
    var name: String { url.lastPathComponent }
    /// nil when capture.json is missing (capture interrupted before Stop).
    var frameCount: Int?
    var hasLidar: Bool
    var sizeBytes: Int64
    var date: Date
    var isComplete: Bool { frameCount != nil }
}

/// Lists, deletes and zips captures stored in the app's Documents directory (visible in the Files app).
@MainActor
@Observable
final class CaptureStore {
    nonisolated static var capturesDirectory: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    private(set) var items: [CaptureItem] = []
    private(set) var isLoading = false

    func refresh() async {
        isLoading = true
        items = await Task.detached(priority: .userInitiated) { Self.scan() }.value
        isLoading = false
    }

    func delete(_ item: CaptureItem) throws {
        try FileManager.default.removeItem(at: item.url)
        items.removeAll { $0.id == item.id }
    }

    /// Zip a capture directory (top-level folder `capture_.../` inside the zip) into a temporary file.
    func exportZip(_ item: CaptureItem) async throws -> URL {
        let url = item.url
        return try await Task.detached(priority: .userInitiated) { try Self.zip(directory: url) }.value
    }

    nonisolated static func exportsDirectory() -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("Exports", isDirectory: true)
    }

    nonisolated static func removeExport(_ url: URL) {
        try? FileManager.default.removeItem(at: url)
    }

    // MARK: - Background helpers

    nonisolated static func scan() -> [CaptureItem] {
        let fm = FileManager.default
        let keys: [URLResourceKey] = [.isDirectoryKey, .creationDateKey, .contentModificationDateKey]
        guard let urls = try? fm.contentsOfDirectory(at: capturesDirectory, includingPropertiesForKeys: keys,
                                                     options: [.skipsHiddenFiles]) else { return [] }
        var out: [CaptureItem] = []
        for url in urls where url.lastPathComponent.hasPrefix(CaptureFormat.directoryPrefix) {
            let values = try? url.resourceValues(forKeys: Set(keys))
            guard values?.isDirectory == true else { continue }
            var frameCount: Int?
            var hasLidar = false
            if let data = try? Data(contentsOf: url.appendingPathComponent(CaptureFormat.manifestFileName)),
               let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                frameCount = (obj["frames"] as? [Any])?.count ?? 0
                hasLidar = obj["has_lidar"] as? Bool ?? false
            }
            out.append(CaptureItem(url: url, frameCount: frameCount, hasLidar: hasLidar,
                                   sizeBytes: directorySize(url),
                                   date: values?.creationDate ?? values?.contentModificationDate ?? .distantPast))
        }
        return out.sorted { $0.date > $1.date }
    }

    nonisolated static func directorySize(_ url: URL) -> Int64 {
        let keys: [URLResourceKey] = [.isRegularFileKey, .fileSizeKey]
        guard let e = FileManager.default.enumerator(at: url, includingPropertiesForKeys: keys) else { return 0 }
        var total: Int64 = 0
        for case let file as URL in e {
            guard let v = try? file.resourceValues(forKeys: Set(keys)), v.isRegularFile == true else { continue }
            total += Int64(v.fileSize ?? 0)
        }
        return total
    }

    /// NSFileCoordinator `.forUploading` on a directory produces a zip of it in a temporary location that is
    /// only valid inside the accessor block, so it is moved into our own exports directory.
    nonisolated static func zip(directory: URL) throws -> URL {
        let fm = FileManager.default
        let exports = exportsDirectory()
        try fm.createDirectory(at: exports, withIntermediateDirectories: true)
        let dest = exports.appendingPathComponent(directory.lastPathComponent).appendingPathExtension("zip")
        try? fm.removeItem(at: dest)

        var coordinatorError: NSError?
        var result: Result<URL, Error> = .failure(CocoaError(.fileReadUnknown))
        NSFileCoordinator().coordinate(readingItemAt: directory, options: [.forUploading], error: &coordinatorError) { zipURL in
            do {
                try fm.moveItem(at: zipURL, to: dest)
                result = .success(dest)
            } catch {
                do {
                    try fm.copyItem(at: zipURL, to: dest)
                    result = .success(dest)
                } catch {
                    result = .failure(error)
                }
            }
        }
        if let coordinatorError { throw coordinatorError }
        return try result.get()
    }
}
