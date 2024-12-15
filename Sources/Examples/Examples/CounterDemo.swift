import Compute
import Metal
import os

enum CounterDemo: Demo {
    static let source = #"""
        #include <metal_stdlib>
        #include <metal_logging>

        using namespace metal;

        uint thread_position_in_grid [[thread_position_in_grid]];

        kernel void main(
            device float *counters [[buffer(0)]],
            constant uint &count [[buffer(1)]],
            constant uint &step [[buffer(2)]]


        ) {
            for(uint n = 0; n != count; ++n) {
                counters[thread_position_in_grid] += float(n);
            }
        }
    """#

    static func main() async throws {
        let device = MTLCreateSystemDefaultDevice()!
        let logger = Logger()
        let compute = try Compute(device: device, logger: logger)
        let library = ShaderLibrary.source(source, enableLogging: true)
        let numberOfCounters = 100240
        let counters = device.makeBuffer(length: MemoryLayout<UInt32>.size * numberOfCounters, options: [])!
        var pipeline = try compute.makePipeline(function: library.main)
        pipeline.arguments.counters = .buffer(counters)
        pipeline.arguments.count = .int(100_000_000)
        pipeline.arguments.step = .int(1)
        try timeit {
            try compute.run(pipeline: pipeline, width: numberOfCounters)
        }
//        print(Array<Float>(counters))

    }
}
