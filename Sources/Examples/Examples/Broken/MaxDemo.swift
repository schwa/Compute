import Compute
import CoreImage
import Metal
import MetalKit
import os
import UniformTypeIdentifiers

enum MaxValue: Demo {

    // Finds the maximum value in an array using a single-threaded Metal compute shader.
    // This method is very inefficient and is provided as an example of a suboptimal approach.
    static func badIdea(values: [Int32], expectedValue: Int32) throws {
        let source = #"""
            #include <metal_stdlib>

            using namespace metal;

            kernel void maxValue(
                const device uint *input [[buffer(0)]],
                constant uint &count [[buffer(1)]],
                device uint &output [[buffer(2)]]
            ) {
                uint temp = 0;
                for (uint n = 0; n != count; ++n) {
                    temp = max(temp, input[n]);
                }
                output = temp;
            }
        """#

        let device = MTLCreateSystemDefaultDevice()!
        let input = device.makeBuffer(bytes: values, length: MemoryLayout<Int32>.stride * values.count, options: [])!
        let output = device.makeBuffer(length: MemoryLayout<Int32>.size)!
        let compute = try Compute(device: device)
        let library = ShaderLibrary.source(source)
        var maxValue = try compute.makePipeline(function: library.maxValue)
        maxValue.arguments.input = .buffer(input)
        maxValue.arguments.count = .int(UInt32(values.count))
        maxValue.arguments.output = .buffer(output)
        try timeit(#function) {
            try compute.run(pipeline: maxValue, width: 1)
        }
        let result = output.contents().assumingMemoryBound(to: Int32.self)[0]
        print(result)
        assert(result == expectedValue)
    }

    // Finds the maximum value in an array using an atomic operation in a Metal compute shader.
    // Despite all threads fighting over a single lock this version is still extremely fast.
    static func simpleAtomic(values: [Int32], expectedValue: Int32) throws {
        let source = #"""
            #include <metal_stdlib>

            using namespace metal;

            uint thread_position_in_grid [[thread_position_in_grid]];

            kernel void maxValue(
                const device uint *input [[buffer(0)]],
                device atomic_uint *output [[buffer(1)]]
            ) {
                const uint value = input[thread_position_in_grid];
                atomic_fetch_max_explicit(output, value, memory_order_relaxed);
            }
        """#
        let device = MTLCreateSystemDefaultDevice()!
        let input = device.makeBuffer(bytes: values, length: MemoryLayout<Int32>.stride * values.count, options: [])!
        let output = device.makeBuffer(length: MemoryLayout<Int32>.size)!
        let compute = try Compute(device: device)
        let library = ShaderLibrary.source(source)
        var maxValue = try compute.makePipeline(function: library.maxValue)
        maxValue.arguments.input = .buffer(input)
        maxValue.arguments.output = .buffer(output)
        try timeit(#function) {
            try compute.run(pipeline: maxValue, width: values.count)
        }
        let result = output.contents().assumingMemoryBound(to: Int32.self)[0]
        print(result)
        assert(result == expectedValue)
    }

    // Finds the maximum value in an array using a multi-pass approach in a Metal compute shader.
    // This method uses SIMD group operations for efficient parallel processing.
    // Note: this method is destructive and intermediate values are written to the input buffer.
    // TODO: This method may occasionally fail for reasons that are currently unclear.
    static func multipass(values: [Int32], expectedValue: Int32) throws {
        let source = #"""
        #include <metal_stdlib>

        using namespace metal;

        // Get the global thread position in the execution grid
        uint thread_position_in_grid [[thread_position_in_grid]];

        // Get the number of threads per SIMD group
        uint threads_per_simdgroup [[threads_per_simdgroup]];

        kernel void maxValue(
            device int *input [[buffer(0)]],    // Input/output buffer
            constant uint &count [[buffer(1)]], // Total number of elements
            constant uint &stride [[buffer(2)]]   // Stride between elements processed by each thread
        ) {
            // Calculate the index for this thread
            const uint index = thread_position_in_grid * stride;

            // Get the value for this thread, or INT_MIN if out of bounds
            uint localValue = index >= count ? INT_MIN : input[index];

            // Perform a parallel reduction to find the maximum value
            for (uint offset = threads_per_simdgroup >> 1; offset > 0; offset >>= 1) {
                // Get the value from another thread in the SIMD group
                const uint remoteValue = simd_shuffle_down(localValue, offset);

                // Update the current value with the maximum of current and remote
                localValue = max(localValue, remoteValue);
            }

            // Only the first thread in each SIMD group writes the result
            if (simd_is_first()) {
                input[index] = localValue;
            }
        }
        """#
        let device = MTLCreateSystemDefaultDevice()!
        let input = device.makeBuffer(bytes: values, length: MemoryLayout<Int32>.stride * values.count, options: [])!
        let compute = try Compute(device: device)
        let library = ShaderLibrary.source(source)
        var pipeline = try compute.makePipeline(function: library.maxValue)
        pipeline.arguments.input = .buffer(input)

        // This is equivalent `threads_per_simdgroup` in MSL.
        let threadExecutionWidth = pipeline.computePipelineState.threadExecutionWidth
        assert(threadExecutionWidth == 32)
        let maxTotalThreadsPerThreadgroup = pipeline.computePipelineState.maxTotalThreadsPerThreadgroup

        try timeit(#function) {

            // Initialize the stride (between elements processed by each thread) to 1
            var stride = 1

            try compute.task { task in
                try task { dispatch in
                    // Continue looping while the stride is less than or equal to the total count of values
                    while stride <= values.count {
                        // Set the 'count' argument for the compute pipeline to the total number of values
                        pipeline.arguments.count = .int(Int32(values.count))

                        // Set the 'stride' argument for the compute pipeline
                        pipeline.arguments.stride = .int(UInt32(stride))

                        // Dispatch the compute pipeline
                        try dispatch(
                            pipeline: pipeline,
                            // Set the total number of threads to process all values
                            threads: MTLSize(width: values.count / stride, height: 1, depth: 1),
                            // Set the number of threads per threadgroup to the maximum allowed
                            threadsPerThreadgroup: MTLSize(width: maxTotalThreadsPerThreadgroup, height: 1, depth: 1)
                        )

                        // Increase the stride by multiplying it with the thread execution width
                        // This effectively reduces the number of active threads in each iteration
                        stride *= threadExecutionWidth
                    }
                }
            }
        }

        let result = input.contents().assumingMemoryBound(to: Int32.self)[0]
        print(result)
        assert(result == expectedValue)
    }

    static func main() async throws {
        //        var values = Array(Array(repeating: Int32.zero, count: 1000))
        let expectedValue: Int32 = 123456789

        #if os(macOS)
        let count: Int32 = 100_000_000
        #else
        let count: Int32 = 1_000_000
        #endif

        var values = Array(Int32.zero ..< count)
        values[Int.random(in: 0..<values.count)] = expectedValue

        timeit("Array.max()") {
            print(values.max()!)
        }
        try multipass(values: values, expectedValue: expectedValue)
        try simpleAtomic(values: values, expectedValue: expectedValue)
        try badIdea(values: values, expectedValue: expectedValue)
    }
}
