import SwiftUI
import UIKit

struct CapturesListView: View {
    @State private var store = CaptureStore()
    @State private var exportingID: CaptureItem.ID?
    @State private var shareFile: ShareFile?
    @State private var pendingDelete: CaptureItem?
    @State private var errorText: String?
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        List {
            if store.items.isEmpty && !store.isLoading {
                ContentUnavailableView("No captures yet", systemImage: "camera.viewfinder",
                                       description: Text("Record a capture, then share it here as a .zip (AirDrop to your Mac)."))
            }
            ForEach(store.items) { item in
                row(item)
                    .swipeActions(edge: .trailing, allowsFullSwipe: false) {
                        Button(role: .destructive) { pendingDelete = item } label: { Label("Delete", systemImage: "trash") }
                    }
                    .swipeActions(edge: .leading) {
                        if item.isComplete {
                            Button { share(item) } label: { Label("Share", systemImage: "square.and.arrow.up") }
                                .tint(.blue)
                        }
                    }
                    .contextMenu {
                        if item.isComplete {
                            Button { share(item) } label: { Label("Share .zip", systemImage: "square.and.arrow.up") }
                        }
                        Button(role: .destructive) { pendingDelete = item } label: { Label("Delete", systemImage: "trash") }
                    }
            }
        }
        .navigationTitle("Captures")
        .toolbar {
            ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } }
        }
        .overlay { if store.isLoading && store.items.isEmpty { ProgressView() } }
        .task { await store.refresh() }
        .refreshable { await store.refresh() }
        .confirmationDialog("Delete \(pendingDelete?.name ?? "capture")?",
                            isPresented: Binding(get: { pendingDelete != nil }, set: { if !$0 { pendingDelete = nil } }),
                            titleVisibility: .visible,
                            presenting: pendingDelete) { item in
            Button("Delete", role: .destructive) {
                do { try store.delete(item) } catch { errorText = error.localizedDescription }
            }
        } message: { _ in
            Text("This permanently removes the capture from this iPhone.")
        }
        .sheet(item: $shareFile, onDismiss: {
            if let url = shareFile?.url { CaptureStore.removeExport(url) }
        }) { file in
            ActivityView(items: [file.url]) {
                CaptureStore.removeExport(file.url)
                shareFile = nil
            }
            .ignoresSafeArea()
        }
        .alert("Error", isPresented: Binding(get: { errorText != nil }, set: { if !$0 { errorText = nil } })) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(errorText ?? "")
        }
    }

    @ViewBuilder
    private func row(_ item: CaptureItem) -> some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                Text(item.name).font(.headline.monospaced()).lineLimit(1).minimumScaleFactor(0.7)
                HStack(spacing: 8) {
                    if let n = item.frameCount {
                        Text("\(n) frames")
                    } else {
                        Text("incomplete (no capture.json)").foregroundStyle(.orange)
                    }
                    Text(ByteCountFormatter.string(fromByteCount: item.sizeBytes, countStyle: .file))
                    if item.hasLidar { Text("LiDAR").foregroundStyle(.green) }
                }
                .font(.caption)
                .foregroundStyle(.secondary)
                Text(item.date, format: .dateTime.year().month().day().hour().minute())
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            if exportingID == item.id {
                ProgressView()
            } else if item.isComplete {
                Button { share(item) } label: { Image(systemName: "square.and.arrow.up") }
                    .buttonStyle(.borderless)
                    .accessibilityLabel("Share \(item.name) as zip")
            }
        }
    }

    private func share(_ item: CaptureItem) {
        guard exportingID == nil else { return }
        exportingID = item.id
        Task {
            defer { exportingID = nil }
            do {
                let url = try await store.exportZip(item)
                shareFile = ShareFile(url: url)
            } catch {
                errorText = "Could not create zip: \(error.localizedDescription)"
            }
        }
    }
}

struct ShareFile: Identifiable {
    let url: URL
    var id: URL { url }
}

/// UIActivityViewController (AirDrop, Files, ...) for the exported zip.
struct ActivityView: UIViewControllerRepresentable {
    let items: [URL]
    var onComplete: () -> Void = {}

    func makeUIViewController(context: Context) -> UIActivityViewController {
        let vc = UIActivityViewController(activityItems: items, applicationActivities: nil)
        vc.completionWithItemsHandler = { _, _, _, _ in onComplete() }
        return vc
    }

    func updateUIViewController(_ vc: UIActivityViewController, context: Context) {}
}
