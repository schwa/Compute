import Compute
import Metal
import os

enum Histogram: Demo {
    static func main() async throws {
        let device = MTLCreateSystemDefaultDevice()!
        let values: [UInt32] = timeit("Generate random input") {
            (0..<1_000_000).map { n in UInt32.random(in: 0..<20) }
        }
        let input = try device.makeTypedBuffer(data: values)
        let bucketCount = 256
        let output: TypedMTLBuffer<UInt32> = try device.makeTypedBuffer(count: 256)
        let compute = try Compute(device: device, logger: Logger())
        let library = ShaderLibrary.bundle(.module, name: "debug")
        var histogram = try compute.makePipeline(function: .init(library: library, name: "Histogram::histogram1"))
        histogram.arguments.input = .buffer(input)
        histogram.arguments.count = .int(input.count)
        histogram.arguments.shift = .int(0)
        histogram.arguments.output = .buffer(output)
        try timeit("GPU") {
            try compute.run(pipeline: histogram, threads: [values.count], threadsPerThreadgroup: [histogram.maxTotalThreadsPerThreadgroup])
        }
        let result = Array(output)
        let expectedResult = timeit("CPU") {
             values.reduce(into: Array<UInt32>(repeating: 0, count: bucketCount)) { partialResult, value in
                let index = Int(value)
                partialResult[index] += 1
            }
        }

        print(result)
        print(expectedResult)
        print(result == expectedResult)
    }
}
