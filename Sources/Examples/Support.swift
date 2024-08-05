import AVFoundation
import Compute
import CoreGraphics
import Foundation
import Metal

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
