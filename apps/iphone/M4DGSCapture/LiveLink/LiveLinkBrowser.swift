import Foundation
import Network
import Observation

/// Browses Bonjour for M4DGS Studio (`_m4dgs-live._tcp`, docs/formats.md §6) and lists the Macs it finds.
/// The first browse triggers the iOS local-network permission prompt.
@MainActor
@Observable
final class LiveLinkBrowser {
    enum State: Equatable {
        case idle
        case searching
        case denied            // local network access not allowed
        case failed(String)
    }

    /// Instance names (= the Mac's name), sorted.
    private(set) var macs: [String] = []
    private(set) var state: State = .idle

    @ObservationIgnored private var browser: NWBrowser?
    @ObservationIgnored private var endpoints: [String: NWEndpoint] = [:]

    var isRunning: Bool { browser != nil }

    func start() {
        guard browser == nil else { return }
        let params = NWParameters.tcp
        let b = NWBrowser(for: .bonjour(type: LiveLinkProtocol.serviceType, domain: nil), using: params)
        b.stateUpdateHandler = { [weak self] newState in
            MainActor.assumeIsolated { self?.update(newState) }
        }
        b.browseResultsChangedHandler = { [weak self] results, _ in
            MainActor.assumeIsolated { self?.update(results) }
        }
        browser = b
        state = .searching
        b.start(queue: .main)
    }

    func stop() {
        browser?.cancel()
        browser = nil
        macs = []
        endpoints = [:]
        state = .idle
    }

    /// The browsed endpoint when the Mac is currently visible, otherwise a Bonjour service endpoint by name
    /// (NWConnection resolves it itself, so a Mac that appears later still works).
    func endpoint(for name: String) -> NWEndpoint {
        endpoints[name] ?? .service(name: name, type: LiveLinkProtocol.serviceType, domain: "local.", interface: nil)
    }

    /// One line for the HUD / Settings when not recording.
    func statusLine(selected: String) -> String {
        switch state {
        case .denied:
            return "Mac: local network access denied — allow it in iOS Settings › Privacy & Security › Local Network"
        case .failed(let reason):
            return "Mac: discovery failed (\(reason))"
        case .idle, .searching:
            if selected.isEmpty {
                return macs.isEmpty ? "Mac: searching for M4DGS Studio…" : "Mac: pick a Mac in Settings (\(macs.count) found)"
            }
            return macs.contains(selected) ? "Mac: \(selected) ready" : "Mac: searching for \(selected)…"
        }
    }

    // MARK: - Private

    private func update(_ newState: NWBrowser.State) {
        switch newState {
        case .ready, .setup:
            state = .searching
        case .waiting(let error), .failed(let error):
            if Self.isPolicyDenied(error) {
                state = .denied
            } else if case .failed = newState {
                state = .failed(error.localizedDescription)
                // Restart once the failure is reported; NWBrowser cannot be restarted after .failed.
                browser?.cancel()
                browser = nil
            } else {
                state = .searching
            }
        case .cancelled:
            break
        @unknown default:
            break
        }
    }

    private func update(_ results: Set<NWBrowser.Result>) {
        var found: [String: NWEndpoint] = [:]
        for r in results {
            if case .service(let name, _, _, _) = r.endpoint { found[name] = r.endpoint }
        }
        endpoints = found
        macs = found.keys.sorted()
        if state == .denied { state = .searching }
    }

    private static func isPolicyDenied(_ error: NWError) -> Bool {
        if case .dns(let code) = error, code == -65570 { return true }   // kDNSServiceErr_PolicyDenied
        return false
    }
}
