import Compute
import Metal
import MetalSupport
import os

enum Histogram: Demo {
    static let source = #"""
        #include <metal_stdlib>
        #include <metal_logging>

        using namespace metal;

        uint thread_position_in_grid [[thread_position_in_grid]];
        uint thread_position_in_threadgroup [[thread_position_in_threadgroup]];

        kernel void histogram(
            device uint *input [[buffer(0)]],
            device atomic_uint *buckets [[buffer(1)]],
            constant uint &bucketCount [[buffer(2)]],
            threadgroup atomic_uint *scratch [[threadgroup(0)]]
        ) {
            const uint value = input[thread_position_in_grid];
            atomic_fetch_add_explicit(&scratch[value], value < bucketCount ? 1 : 0, memory_order_relaxed);

            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (thread_position_in_threadgroup == 0) {
                for (uint n = 0; n != bucketCount; ++n) {
                    const uint value = atomic_load_explicit(&scratch[n], memory_order_relaxed);
                    atomic_fetch_add_explicit(&buckets[n], value, memory_order_relaxed);
                }
            }
        }
"""#

    static func main() async throws {
        let device = MTLCreateSystemDefaultDevice()!
        let values: [UInt32] = timeit("Generate random input") {
            (0..<1_000_000).map { n in UInt32.random(in: 0..<20) }
        }
        let input = try device.makeBuffer(bytesOf: values, options: [])
        let bucketCount = 32
        let buckets = device.makeBuffer(length: bucketCount * MemoryLayout<UInt32>.stride, options: [])!
        let compute = try Compute(device: device, logger: Logger())
        let library = ShaderLibrary.source(source, enableLogging: true)
        var histogram = try compute.makePipeline(function: library.histogram)
        histogram.arguments.input = .buffer(input)
        histogram.arguments.buckets = .buffer(buckets)
        histogram.arguments.bucketCount = .int(UInt32(bucketCount))
        histogram.arguments.scratch = .threadgroupMemoryLength(bucketCount * MemoryLayout<UInt>.stride) // TODO: Align 16.
        try timeit("GPU") {
            try compute.run(pipeline: histogram, threads: [values.count], threadsPerThreadgroup: [histogram.maxTotalThreadsPerThreadgroup])
        }
        let result = Array<UInt32>(buckets)
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
