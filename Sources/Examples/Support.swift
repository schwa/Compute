import AVFoundation
import Compute
import CoreGraphics
import Foundation
import Metal
import MetalSupportLite

public func getMachTimeInNanoseconds() -> UInt64 {
    var timebase = mach_timebase_info_data_t()
    mach_timebase_info(&timebase)
    let currentTime = mach_absolute_time()
    return currentTime * UInt64(timebase.numer) / UInt64(timebase.denom)
}

@discardableResult
public func timeit<R>(_ work: () throws -> R, display: (UInt64) -> Void) rethrows -> R {
    let start = getMachTimeInNanoseconds()
    let result = try work()
    let end = getMachTimeInNanoseconds()
    display(end - start)
    return result
}

@discardableResult
public func timeit<R>(_ label: String? = nil, _ work: () throws -> R) rethrows -> R {
    try timeit(work) { delta in
        let measurement = Measurement(value: Double(delta), unit: UnitDuration.nanoseconds)
        let measurementMS = measurement.converted(to: .milliseconds)
        print("\(label ?? "<unamed>"): \(measurementMS.formatted())")
    }
}

@discardableResult
public func timeit<R>(_ label: String? = nil, length: Int, _ work: () throws -> R) rethrows -> R {
    try timeit(work) { delta in
        let seconds = Double(delta) / 1_000_000_000
        let bytesPerSecond = Double(length) / seconds
        let gigabytesPerSecond = Measurement(value: bytesPerSecond, unit: UnitInformationStorage.bytes)
            .converted(to: .gigabytes)
        print("Time: \(Measurement(value: Double(seconds), unit: UnitDuration.seconds).converted(to: .milliseconds).formatted())")
        print("Speed: \(gigabytesPerSecond.formatted(.measurement(width: .abbreviated, usage: .asProvided)))/s")
    }
}

class TextureToVideoWriter {
    private var assetWriter: AVAssetWriter
    private var writerInput: AVAssetWriterInput?
    private var adaptor: AVAssetWriterInputPixelBufferAdaptor?

    let outputURL: URL
    let temporaryURL: URL
    private let size: CGSize
    private let pixelFormat = kCVPixelFormatType_32BGRA

    var endTime: CMTime?

    init(outputURL: URL, size: CGSize) throws {
        self.outputURL = outputURL
        let temporaryURL = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).mp4")
        self.temporaryURL = temporaryURL
        self.size = size

        assetWriter = try AVAssetWriter(outputURL: temporaryURL, fileType: .mov)
    }

    func start() {
        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.hevc,
            AVVideoWidthKey: size.width,
            AVVideoHeightKey: size.height
        ]

        writerInput = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        writerInput?.expectsMediaDataInRealTime = true

        adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: writerInput!,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: pixelFormat,
                kCVPixelBufferWidthKey as String: Int(size.width),
                kCVPixelBufferHeightKey as String: Int(size.height)
            ]
        )

        if assetWriter.canAdd(writerInput!) {
            assetWriter.add(writerInput!)
        }

        assetWriter.startWriting()
        assetWriter.startSession(atSourceTime: .zero)
    }

    func writeFrame(texture: MTLTexture, at time: CMTime) {
        autoreleasepool {
            guard let adaptor, let pixelBufferPool = adaptor.pixelBufferPool else {
                fatalError()
            }
            var pixelBuffer: CVPixelBuffer?
            CVPixelBufferPoolCreatePixelBuffer(nil, pixelBufferPool, &pixelBuffer)
            guard let pixelBuffer else {
                fatalError()
            }
            CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
            guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
                fatalError()
            }
            let region = MTLRegionMake2D(0, 0, texture.width, texture.height)
            texture.getBytes(baseAddress, bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer), from: region, mipmapLevel: 0)
            CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
            writeFrame(pixelBuffer: pixelBuffer, at: time)
            endTime = time
        }
    }

    func writeFrame(pixelBuffer: CVPixelBuffer, at time: CMTime) {
        autoreleasepool {
            guard let writerInput, let adaptor else {
                fatalError()
            }
            if writerInput.isReadyForMoreMediaData == false {
                // This isn't pretty but it works?
                while writerInput.isReadyForMoreMediaData == false {
                    usleep(10 * 1_000)
                }
            }
            adaptor.append(pixelBuffer, withPresentationTime: time)
        }
    }

    func finish() async throws {
        writerInput?.markAsFinished()
        assetWriter.endSession(atSourceTime: endTime!)
        await assetWriter.finishWriting()
        let fileManager = FileManager()
        if fileManager.fileExists(atPath: outputURL.path) {
            try FileManager().removeItem(at: outputURL)
        }
        try FileManager().moveItem(at: temporaryURL, to: outputURL)
    }
}

extension MTLTexture {
    func export(to url: URL) throws {
        assert(pixelFormat == .bgra8Unorm)
        assert(depth == 1)

        let bytesPerRow = width * MemoryLayout<UInt8>.size * 4
        guard let colorSpace = CGColorSpace(name: CGColorSpace.genericRGBLinear) else {
            throw ComputeError.resourceCreationFailure
        }
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue)
            .union(.byteOrder32Little)
        guard let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: bytesPerRow, space: colorSpace, bitmapInfo: bitmapInfo.rawValue) else {
            throw ComputeError.resourceCreationFailure
        }
        guard let contextData = context.data else {
            throw ComputeError.resourceCreationFailure
        }
        getBytes(contextData, bytesPerRow: bytesPerRow, from: MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0), size: MTLSize(width: width, height: height, depth: 1)), mipmapLevel: 0)

        guard let image = context.makeImage() else {
            throw ComputeError.resourceCreationFailure
        }

        guard let imageDestination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            throw ComputeError.resourceCreationFailure
        }
        CGImageDestinationAddImage(imageDestination, image, nil)
        CGImageDestinationFinalize(imageDestination)
    }
}

extension MTLBuffer {
    func withUnsafeBytes<ResultType, ContentType>(_ body: (UnsafeBufferPointer<ContentType>) throws -> ResultType) rethrows -> ResultType {
        try withUnsafeBytes { (buffer: UnsafeRawBufferPointer) in
            try buffer.withMemoryRebound(to: ContentType.self, body)
        }
    }

    func withUnsafeBytes<ResultType>(_ body: (UnsafeRawBufferPointer) throws -> ResultType) rethrows -> ResultType {
        try body(UnsafeRawBufferPointer(start: contents(), count: length))
    }
}

extension Array where Element: Equatable {

    struct Run {
        var element: Element
        var count: Int
    }

    func rle() -> [Run] {

        var lastElement: Element?
        var runLength = 0

        var runs: [Run] = []

        for element in self {
            if element == lastElement {
                runLength += 1
            }
            else {
                if let lastElement {
                    runs.append(.init(element: lastElement, count: runLength))
                }
                lastElement = element
                runLength = 1
            }
        }

        if let lastElement {
            runs.append(.init(element: lastElement, count: runLength))
        }

        return runs
    }

}

extension Collection where Element == UInt32 {
    func prefixSum() -> [UInt32] {
        var output = Array(repeating: UInt32.zero, count: count)
        for j in 1..<count {
            output[j] = self[self.index(self.startIndex, offsetBy: j-1)] + output[j-1]
        }
        return output
    }
}

// MARK: -

func log2(_ value: UInt32, ceiling: Bool = false) -> UInt32 {
    let result = UInt32(31 - value.leadingZeroBitCount)
    return ceiling && (1 << result) < value ? result + 1 : result
}

func log2(_ value: Int, ceiling: Bool = false) -> Int {
    precondition(value > 0, "log2 is only defined for positive numbers")
    let result = 63 - value.leadingZeroBitCount
    return ceiling && (1 << result) < value ? result + 1 : result
}

// MARK: -

infix operator **: MultiplicationPrecedence

func ** (base: Int, exponent: Int) -> Int {
    return Int(pow(Double(base), Double(exponent)))
}

func ** (base: UInt32, exponent: UInt32) -> UInt32 {
    return UInt32(pow(Double(base), Double(exponent)))
}

extension Array {
    init(_ buffer: MTLBuffer) {
        let pointer = buffer.contents().bindMemory(to: Element.self, capacity: buffer.length / MemoryLayout<Element>.size)
        let buffer = UnsafeBufferPointer<Element>(start: pointer, count: buffer.length / MemoryLayout<Element>.size)
        self = Array(buffer)
    }

}

func ceildiv <T>(_ x: T, _ y: T) -> T where T: BinaryInteger {
    (x + y - 1) / y
}

extension Array {
    init(_ buffer: TypedMTLBuffer<Element>) {
        self = buffer.withUnsafeMTLBuffer { buffer in
            Array(buffer!)
        }
    }
}

extension Compute.Argument {
    static func buffer<T>(_ data: TypedMTLBuffer<T>) -> Self {
        data.withUnsafeMTLBuffer { buffer in
            return .buffer(buffer!)
        }
    }
}

func nextPowerOfTwo(_ n: Int) -> Int {
    return Int(pow(2.0, Double(Int(log2(Double(n))).advanced(by: 1))) + 0.5)
}

extension MTLSize: @retroactive ExpressibleByArrayLiteral {
    public init(arrayLiteral elements: Int...) {
        switch elements.count {
        case 1:
            self = .init(elements[0], 1, 1)
        case 2:
            self = .init(elements[0], elements[1], 1)
        case 3:
            self = .init(elements[0], elements[1], elements[2])
        default:
            fatalError()
        }
    }

}

/// Generates an array of integers by repeatedly dividing the start value by the divisor and rounding up.
/// - Parameters:
///   - start: The initial value to start the sequence.
///   - divisor: The number to divide by in each iteration.
/// - Returns: An array of integers representing the sequence.
/// - Precondition: `divisor` must be greater than 0.
func stride(from start: Int, dividingBy divisor: Int) -> AnyIterator<Int> {
    precondition(divisor > 0, "Divisor must be greater than 0")
    var current = start
    return AnyIterator {
        guard current > 1 else { return nil }
        defer { current = ceildiv(current, divisor) }
        return current
    }
}

public extension MTLSize {
    init(_ size: CGSize) {
        self.init(width: Int(size.width), height: Int(size.height), depth: 1)
    }

    init(_ width: Int, _ height: Int, _ depth: Int) {
        self = MTLSize(width: width, height: height, depth: depth)
    }

    init(width: Int) {
        self = MTLSize(width: width, height: 1, depth: 1)
    }

    init(width: Int, height: Int) {
        self = MTLSize(width: width, height: height, depth: 1)
    }
}

extension MTLDevice {
    func capture <R>(enabled: Bool = true, _ block: () throws -> R) throws -> R {
        guard enabled else {
            return try block()
        }
        let captureManager = MTLCaptureManager.shared()
        let captureScope = captureManager.makeCaptureScope(device: self)
        let captureDescriptor = MTLCaptureDescriptor()
        captureDescriptor.captureObject = captureScope
        try captureManager.startCapture(with: captureDescriptor)
        captureScope.begin()
        defer {
            captureScope.end()
        }
        return try block()
    }

    func makeBuffer(bytesOf content: [some Any], options: MTLResourceOptions) throws -> MTLBuffer {
        try content.withUnsafeBytes { buffer in
            let baseAddress = buffer.baseAddress!
            guard let buffer = makeBuffer(bytes: baseAddress, length: buffer.count, options: options) else {
                fatalError()
            }
            return buffer
        }
    }

}

public extension MTLBuffer {
    func data() -> Data {
        Data(bytes: contents(), count: length)
    }

    /// Update a MTLBuffer's contents using an inout type block
    func with<T, R>(type: T.Type, _ block: (inout T) -> R) -> R {
        let value = contents().bindMemory(to: T.self, capacity: 1)
        return block(&value.pointee)
    }

    func withEx<T, R>(type: T.Type, count: Int, _ block: (UnsafeMutableBufferPointer<T>) -> R) -> R {
        let pointer = contents().bindMemory(to: T.self, capacity: count)
        let buffer = UnsafeMutableBufferPointer(start: pointer, count: count)
        return block(buffer)
    }

    func contentsBuffer() -> UnsafeMutableRawBufferPointer {
        UnsafeMutableRawBufferPointer(start: contents(), count: length)
    }

    func contentsBuffer<T>(of type: T.Type) -> UnsafeMutableBufferPointer<T> {
        contentsBuffer().bindMemory(to: type)
    }
    func labelled(_ label: String) -> MTLBuffer {
        self.label = label
        return self
    }
}
