import Compute
import os
import Metal
import MetalSupport

let source = #"""
    #include <metal_stdlib>
    #include <metal_logging>

    using namespace metal;

    // Thread-specific attributes
    uint thread_position_in_grid [[thread_position_in_grid]];  // Position of the current thread in the grid
    uint threads_per_simdgroup [[threads_per_simdgroup]];      // Number of threads in a SIMD group
    uint threadgroup_position_in_grid [[threadgroup_position_in_grid]];  // Position of the current threadgroup in the grid

    // Kernel function for parallel reduction sum
    kernel void parallel_reduction_sum(
        constant uint* input [[buffer(0)]],   // Input buffer
        device uint* output [[buffer(1)]]    // Output buffer
    )
    {
        // Get the value for the current thread
        uint value = input[thread_position_in_grid];

        // Perform parallel reduction within SIMD group
        for (uint offset = threads_per_simdgroup / 2; offset > 0; offset >>= 1) {
            // Add the value from the thread 'offset' positions ahead
            value += simd_shuffle_and_fill_down(value, 0u, offset);
        }

        // Only the first thread in each SIMD group writes the result
        if (simd_is_first()) {
            output[thread_position_in_grid / threads_per_simdgroup] = value;
        }
    }
"""#

struct SIMDReduce: Demo {

    static func main() async throws {
        try dualBuffer()
    }

    static func dualBuffer() throws {
        let device = MTLCreateSystemDefaultDevice()!
        // Create N values and sum them up in the CPU...
        var count = 50_000_000
        let values = timeit("Generating \(count) values") {
            (0..<count).map { _ in UInt32.random(in: 0..<100) }
        }
        let expectedResult = timeit("CPU") {
            values.reduce(0, +)
        }
        // Create a "Compute" object
        let compute = try Compute(device: device, logger: Logger())
        // Create a shader library from our source string
        let library = ShaderLibrary.source(source, enableLogging: true)
        // Create a compute pass from our function
        var pipeline = try compute.makePipeline(function: library.parallel_reduction_sum)
        // Save the threadExecutionWidth (always 32 on current Apple Silicon). The number of threads that work per simd group.
        let threadExecutionWidth = pipeline.computePipelineState.threadExecutionWidth
        // Create input and output metal buffers to work in... fill the input buffer with our sample data
        var input = try device.makeBuffer(bytesOf: values, options: [])
        var output = try device.makeBuffer(bytesOf: Array(repeating: UInt32.zero, count: ceildiv(count, threadExecutionWidth)), options: [])
        // We'll be running this until we have one count elft
        try timeit("GPU") {
            try compute.task { task in
                try task { dispatch in
                    repeat {
                        // Load our buffers into the pass
                        pipeline.arguments.input = .buffer(input)
                        pipeline.arguments.output = .buffer(output)
                        // Perform one dispatch on the pass
                        let maxTotalThreadsPerThreadgroup = pipeline.computePipelineState.maxTotalThreadsPerThreadgroup
                        try dispatch(pipeline: pipeline, threads: MTLSize(width: count), threadsPerThreadgroup: MTLSize(width: maxTotalThreadsPerThreadgroup))
                        // For the next run we'll be doing less work... The output buffer contains input.count / threadExecutionWidth results...
                        count = ceildiv(count, threadExecutionWidth)
                        // If we're not done yet swap the buffers....
                        if count != 1 {
                            swap(&input, &output)
                        }
                    }
                    while count > 1
                }
            }
        }
        let result = Array(output.contentsBuffer(of: UInt32.self))[0]
        print("Compute result:", expectedResult, result, result == expectedResult)
    }
}
