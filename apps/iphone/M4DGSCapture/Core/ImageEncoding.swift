import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

// CoreGraphics/ImageIO encoders (available on iOS and macOS; checked on macOS by apps/iphone/Tests).

enum ImageEncoding {
    /// 16-bit grayscale PNG (e.g. depth in millimeters). Values are written verbatim.
    static func pngGray16(_ pixels: [UInt16], size: PixelSize) -> Data? {
        guard pixels.count == size.width * size.height, size.width > 0, size.height > 0 else { return nil }
        let data = pixels.withUnsafeBufferPointer { Data(buffer: $0) }   // native = little-endian
        let info = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue | CGBitmapInfo.byteOrder16Little.rawValue)
        return pngGray(data: data, size: size, bitsPerComponent: 16, bitmapInfo: info)
    }

    /// 8-bit grayscale PNG (e.g. ARConfidenceLevel 0/1/2). Values are written verbatim.
    static func pngGray8(_ pixels: [UInt8], size: PixelSize) -> Data? {
        guard pixels.count == size.width * size.height, size.width > 0, size.height > 0 else { return nil }
        let info = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)
        return pngGray(data: Data(pixels), size: size, bitsPerComponent: 8, bitmapInfo: info)
    }

    static func jpeg(_ image: CGImage, quality: Double) -> Data? {
        encode(image, type: UTType.jpeg,
               properties: [kCGImageDestinationLossyCompressionQuality: quality] as CFDictionary)
    }

    private static func pngGray(data: Data, size: PixelSize, bitsPerComponent: Int, bitmapInfo: CGBitmapInfo) -> Data? {
        guard let provider = CGDataProvider(data: data as CFData),
              let image = CGImage(width: size.width,
                                  height: size.height,
                                  bitsPerComponent: bitsPerComponent,
                                  bitsPerPixel: bitsPerComponent,
                                  bytesPerRow: size.width * bitsPerComponent / 8,
                                  space: CGColorSpaceCreateDeviceGray(),
                                  bitmapInfo: bitmapInfo,
                                  provider: provider,
                                  decode: nil,
                                  shouldInterpolate: false,
                                  intent: .defaultIntent)
        else { return nil }
        return encode(image, type: UTType.png, properties: nil)
    }

    private static func encode(_ image: CGImage, type: UTType, properties: CFDictionary?) -> Data? {
        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(out as CFMutableData, type.identifier as CFString, 1, nil) else {
            return nil
        }
        CGImageDestinationAddImage(dest, image, properties)
        guard CGImageDestinationFinalize(dest) else { return nil }
        return out as Data
    }
}
