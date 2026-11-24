import Foundation

/// Minimal safetensors reader / writer (docs/formats.md §3).
/// Layout: 8-byte little-endian header length N, N bytes of JSON header, then raw little-endian tensor bytes.
public enum SafeTensorsError: Error, CustomStringConvertible {
    case tooSmall
    case badHeader(String)
    case missingTensor(String)
    case unsupportedDType(name: String, dtype: String)
    case outOfBounds(String)

    public var description: String {
        switch self {
        case .tooSmall: return "file too small to be a safetensors file"
        case .badHeader(let s): return "bad safetensors header: \(s)"
        case .missingTensor(let n): return "missing tensor '\(n)'"
        case .unsupportedDType(let n, let d): return "tensor '\(n)' has unsupported dtype \(d)"
        case .outOfBounds(let n): return "tensor '\(n)' data offsets out of bounds"
        }
    }
}

public struct TensorInfo: Sendable {
    public let name: String
    public let dtype: String
    public let shape: [Int]
    /// Byte range relative to the start of the data section.
    public let byteRange: Range<Int>
    public var elementCount: Int { shape.reduce(1, *) }
}

public final class SafeTensorsFile {
    public let data: Data
    public let headerSize: Int
    public let tensors: [String: TensorInfo]
    public let metadata: [String: String]

    public convenience init(url: URL) throws {
        try self.init(data: Data(contentsOf: url, options: .alwaysMapped))
    }

    public init(data: Data) throws {
        self.data = data
        guard data.count >= 8 else { throw SafeTensorsError.tooSmall }
        let n: UInt64 = data.withUnsafeBytes { raw in
            var v: UInt64 = 0
            withUnsafeMutableBytes(of: &v) { $0.copyMemory(from: UnsafeRawBufferPointer(rebasing: raw[0..<8])) }
            return UInt64(littleEndian: v)
        }
        guard n > 0, Int(n) <= data.count - 8 else { throw SafeTensorsError.badHeader("header length \(n) exceeds file size") }
        headerSize = Int(n)
        let headerBytes = data.subdata(in: data.startIndex + 8 ..< data.startIndex + 8 + headerSize)
        guard let obj = try JSONSerialization.jsonObject(with: headerBytes) as? [String: Any] else {
            throw SafeTensorsError.badHeader("header is not a JSON object")
        }
        let dataSize = data.count - 8 - headerSize
        var ts: [String: TensorInfo] = [:]
        var meta: [String: String] = [:]
        for (key, value) in obj {
            if key == "__metadata__" {
                if let m = value as? [String: Any] {
                    for (mk, mv) in m { meta[mk] = (mv as? String) ?? "\(mv)" }
                }
                continue
            }
            guard let d = value as? [String: Any],
                  let dtype = d["dtype"] as? String,
                  let shapeAny = d["shape"] as? [Any],
                  let offs = d["data_offsets"] as? [Any], offs.count == 2 else {
                throw SafeTensorsError.badHeader("malformed entry '\(key)'")
            }
            let shape = shapeAny.compactMap { ($0 as? NSNumber)?.intValue }
            guard shape.count == shapeAny.count,
                  let b = (offs[0] as? NSNumber)?.intValue, let e = (offs[1] as? NSNumber)?.intValue,
                  b >= 0, e >= b, e <= dataSize else { throw SafeTensorsError.outOfBounds(key) }
            ts[key] = TensorInfo(name: key, dtype: dtype, shape: shape, byteRange: b..<e)
        }
        tensors = ts
        metadata = meta
    }

    public func info(_ name: String) -> TensorInfo? { tensors[name] }

    /// Returns tensor values as Float (F32 native; F16 / F64 / BF16 converted for leniency).
    public func floats(_ name: String) throws -> [Float] {
        guard let t = tensors[name] else { throw SafeTensorsError.missingTensor(name) }
        let start = 8 + headerSize + t.byteRange.lowerBound
        let byteCount = t.byteRange.count
        let n = t.elementCount
        func expect(_ bytesPer: Int) throws {
            if byteCount != n * bytesPer { throw SafeTensorsError.outOfBounds(name) }
        }
        return try data.withUnsafeBytes { raw -> [Float] in
            let src = UnsafeRawBufferPointer(rebasing: raw[start ..< start + byteCount])
            switch t.dtype {
            case "F32":
                try expect(4)
                var out = [Float](repeating: 0, count: n)
                out.withUnsafeMutableBytes { $0.copyMemory(from: src) }
                return out
            case "F16":
                try expect(2)
                var h = [Float16](repeating: 0, count: n)
                h.withUnsafeMutableBytes { $0.copyMemory(from: src) }
                return h.map { Float($0) }
            case "BF16":
                try expect(2)
                var u = [UInt16](repeating: 0, count: n)
                u.withUnsafeMutableBytes { $0.copyMemory(from: src) }
                return u.map { Float(bitPattern: UInt32($0) << 16) }
            case "F64":
                try expect(8)
                var d = [Double](repeating: 0, count: n)
                d.withUnsafeMutableBytes { $0.copyMemory(from: src) }
                return d.map { Float($0) }
            default:
                throw SafeTensorsError.unsupportedDType(name: name, dtype: t.dtype)
            }
        }
    }
}

public struct SafeTensor {
    public var name: String
    public var shape: [Int]
    public var values: [Float]
    public init(name: String, shape: [Int], values: [Float]) {
        self.name = name; self.shape = shape; self.values = values
    }
}

public enum SafeTensorsWriter {
    /// Encodes F32 tensors (sorted by name, contiguous) with an optional `__metadata__` string map.
    public static func encode(_ tensors: [SafeTensor], metadata: [String: String] = [:]) throws -> Data {
        var header: [String: Any] = [:]
        if !metadata.isEmpty { header["__metadata__"] = metadata }
        var offset = 0
        let sorted = tensors.sorted { $0.name < $1.name }
        for t in sorted {
            precondition(t.shape.reduce(1, *) == t.values.count, "shape/value mismatch for \(t.name)")
            let bytes = t.values.count * 4
            header[t.name] = ["dtype": "F32", "shape": t.shape, "data_offsets": [offset, offset + bytes]] as [String: Any]
            offset += bytes
        }
        var json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        while json.count % 8 != 0 { json.append(0x20) }
        var out = Data()
        var len = UInt64(json.count).littleEndian
        withUnsafeBytes(of: &len) { out.append(contentsOf: $0) }
        out.append(json)
        for t in sorted { t.values.withUnsafeBytes { out.append(contentsOf: $0) } }
        return out
    }
}
