import Compute
import Metal
import os

enum CountingSortDemo: Demo {

    static let logging = false
    static let capture = true

    static func main() async throws {
        let device = MTLCreateSystemDefaultDevice()!
        let compute = try Compute(device: device, logger: Logger(), logging: logging)
        let library = ShaderLibrary.bundle(.module, name: "debug")
        let count = 1_500_000
        //let elements: [UInt32] = (0..<count).map { _ in UInt32.random(in: 0..<100) }
        let elements: [UInt32] = (0..<count).map { UInt32($0) % 100 }

        let shift = 0
        let expectedResult = elements
            .reduce(into: Array(repeating: UInt32.zero, count: 256)) { result, value in
                let value = UInt8(value >> shift & 0xFF)
                result[Int(value)] += 1
            }

        let input = try device.makeTypedBuffer(data: elements)
        let output: TypedMTLBuffer<UInt32> = try device.makeTypedBuffer(count: 256)
        var histogram = try compute.makePipeline(function: library.function(name: "CountingSort::histogram"))
        histogram.arguments.input = .buffer(input)
        histogram.arguments.output = .buffer(output)
        histogram.arguments.count = .int(input.count)
        histogram.arguments.shift = .int(shift)
        try device.capture(enabled: capture) {
            try compute.run(pipeline: histogram, width: input.count)
        }

        let result = Array(output)
        print(result)
        print(expectedResult)
        print(result == expectedResult)

    }
}
