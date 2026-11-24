import Foundation
import Network
import SystemConfiguration
import M4DGSCore

/// Live link receiver (docs/formats.md §6): an `NWListener` advertising Bonjour `_m4dgs-live._tcp` under this Mac's
/// name. Each accepted TCP connection is one capture, driven by a `LiveCaptureSession` (M4DGSCore) that writes the
/// §2 capture directory under `capturesFolder`. All work happens on `queue`; `onEvent` is called there too.
final class LiveReceiver: @unchecked Sendable {
    enum Event {
        case listening(port: UInt16, serviceName: String)
        case serviceRegistered(name: String)
        case listenerFailed(String)
        case stopped
        case connected(id: Int, endpoint: String)
        case session(id: Int, LiveSessionEvent)
        case protocolError(id: Int, String)
        /// `partial` is the capture directory left with `capture.partial.json`, if the capture did not complete.
        case disconnected(id: Int, state: LiveCaptureSession.State, partial: URL?)
    }

    let queue = DispatchQueue(label: "m4dgs.live-receiver")
    var onEvent: ((Event) -> Void)?
    /// Read when a connection is accepted (changes apply to the next capture).
    var capturesFolder: URL {
        get { queue.sync { _capturesFolder } }
        set { queue.async { self._capturesFolder = newValue } }
    }

    private var _capturesFolder: URL
    private var listener: NWListener?
    private var connections: [Int: LiveConnection] = [:]
    private var nextID = 1

    init(capturesFolder: URL) { _capturesFolder = capturesFolder }

    static var macName: String {
        if let n = SCDynamicStoreCopyComputerName(nil, nil) as String?, !n.isEmpty { return n }
        return Host.current().localizedName ?? ProcessInfo.processInfo.hostName
    }

    var isListening: Bool { queue.sync { listener != nil } }

    /// Starts listening (port 0 = any free port; Bonjour carries the port to the phone).
    func start(port: UInt16 = 0, serviceName: String = LiveReceiver.macName) throws {
        let tcp = NWProtocolTCP.Options()
        tcp.enableKeepalive = true          // notice a phone that vanished from Wi-Fi
        tcp.keepaliveIdle = 10
        tcp.keepaliveInterval = 5
        tcp.keepaliveCount = 4
        tcp.noDelay = true
        let params = NWParameters(tls: nil, tcp: tcp)
        params.allowLocalEndpointReuse = true
        params.includePeerToPeer = true
        let l: NWListener
        if port == 0 { l = try NWListener(using: params) } else {
            guard let p = NWEndpoint.Port(rawValue: port) else { throw LiveLinkError.protocolViolation("bad port \(port)") }
            l = try NWListener(using: params, on: p)
        }
        l.service = NWListener.Service(name: serviceName, type: LiveLink.serviceType)
        l.stateUpdateHandler = { [weak self, weak l] st in
            guard let self else { return }
            switch st {
            case .ready: self.onEvent?(.listening(port: l?.port?.rawValue ?? 0, serviceName: serviceName))
            case .failed(let e):
                self.onEvent?(.listenerFailed("\(e)"))
                l?.cancel()
                if self.listener === l { self.listener = nil }
            case .waiting(let e): self.onEvent?(.listenerFailed("waiting: \(e)"))
            case .cancelled: break
            default: break
            }
        }
        l.serviceRegistrationUpdateHandler = { [weak self] change in
            if case .add(let ep) = change, case .service(let name, _, _, _) = ep { self?.onEvent?(.serviceRegistered(name: name)) }
        }
        l.newConnectionHandler = { [weak self] c in self?.accept(c) }
        queue.sync { listener = l }
        l.start(queue: queue)
    }

    /// Stops listening and closes open connections (incomplete captures keep `capture.partial.json`).
    func stop() {
        queue.async { [self] in
            listener?.cancel()
            listener = nil
            for c in connections.values { c.close(reason: nil) }
            onEvent?(.stopped)
        }
    }

    private func accept(_ c: NWConnection) {
        let id = nextID
        nextID += 1
        let lc = LiveConnection(id: id, connection: c, session: LiveCaptureSession(capturesFolder: _capturesFolder), receiver: self)
        connections[id] = lc
        onEvent?(.connected(id: id, endpoint: "\(c.endpoint)"))
        lc.start(queue: queue)
    }

    fileprivate func connectionEnded(_ lc: LiveConnection) {
        connections[lc.id] = nil
        let partial = lc.session.connectionClosed()
        onEvent?(.disconnected(id: lc.id, state: lc.session.state, partial: partial))
    }

    fileprivate func emit(_ e: Event) { onEvent?(e) }
}

private final class LiveConnection {
    let id: Int
    let connection: NWConnection
    let session: LiveCaptureSession
    weak var receiver: LiveReceiver?
    var decoder = LiveMessageDecoder()
    var ended = false

    init(id: Int, connection: NWConnection, session: LiveCaptureSession, receiver: LiveReceiver) {
        self.id = id; self.connection = connection; self.session = session; self.receiver = receiver
    }

    func start(queue: DispatchQueue) {
        connection.stateUpdateHandler = { [self] st in
            switch st {
            case .failed, .cancelled: finish()
            default: break
            }
        }
        connection.start(queue: queue)
        receive()
    }

    func receive() {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 1 << 20) { [self] data, _, isComplete, error in
            if ended { return }
            if let data, !data.isEmpty {
                decoder.append(data)
                do {
                    while !session.isFinished, let m = try decoder.next() { try process(m) }
                } catch {
                    close(reason: "\(error)")
                    return
                }
            }
            if session.isFinished { return }      // closed after saved / cancel
            if isComplete || error != nil {
                close(reason: error.map { "\($0)" })
                return
            }
            receive()
        }
    }

    func process(_ m: LiveMessage) throws {
        let (replies, events) = try session.handle(m)
        for e in events { receiver?.emit(.session(id: id, e)) }
        for r in replies {
            let last = session.isFinished && r == replies.last
            connection.send(content: r.encoded(), completion: .contentProcessed { [self] err in
                if let err { close(reason: "send failed: \(err)") } else if last { close(reason: nil) }
            })
        }
        if session.state == .cancelled { close(reason: nil) }
    }

    func close(reason: String?) {
        guard !ended else { return }
        if let r = reason { receiver?.emit(.protocolError(id: id, r)) }
        connection.cancel()
        finish()
    }

    private func finish() {
        guard !ended else { return }
        ended = true
        connection.stateUpdateHandler = nil
        receiver?.connectionEnded(self)
    }
}
