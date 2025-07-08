import Compute
import Metal
import os

enum IsSorted: Demo {
    static let source = #"""
        #include <metal_stdlib>
        #include <metal_logging>
    
        using namespace metal;
    
        uint thread_position_in_threadgroup [[thread_position_in_threadgroup]];
        uint threadgroup_position_in_grid [[threadgroup_position_in_grid]];
        uint thread_position_in_grid  [[thread_position_in_grid]];
        uint threads_per_threadgroup [[threads_per_threadgroup]];
    
        kernel void is_sorted(
            device const float* input [[buffer(0)]],
            constant uint& count [[buffer(1)]],
            device atomic_uint* isSorted [[buffer(2)]]
        ) {
            //os_log_default.log("thread_position_in_grid: %d, threads_per_grid: %f", thread_position_in_grid, input[thread_position_in_grid]);
            if (thread_position_in_grid < (count - 1) && input[thread_position_in_grid] > input[thread_position_in_grid + 1]) {
                //os_log_default.log("IS NOT SORTED %d %d", thread_position_in_grid, thread_position_in_grid + 1);
                atomic_store_explicit(isSorted, 0, memory_order_relaxed);
            }                            
        }
    
    
        kernel void is_sorted_complex(
            device const float* input [[buffer(0)]],
            constant uint& count [[buffer(1)]],
            device atomic_bool* isSorted [[buffer(2)]]
        ) {
            uint groupStart = threadgroup_position_in_grid * threads_per_threadgroup;
            
            // Start one before if not the first group
            uint start = (groupStart == 0) ? groupStart + thread_position_in_threadgroup : groupStart - 1 + threadgroup_position_in_grid;
            uint end = metal::min(count - 1, groupStart + threads_per_threadgroup);
            
            for (uint i = start; i < end; i += threads_per_threadgroup) {
                if (input[i] > input[i + 1]) {
                    atomic_store_explicit(isSorted, false, metal::memory_order_relaxed);
                }
            }
        }
    """#

    static func main() async throws {
        try await simple()
        try await complex()
    }

    static func simple() async throws {
        let capture = false

        let device = MTLCreateSystemDefaultDevice()!
        try device.capture(enabled: capture) {
            let logger = Logger()
            let compute = try Compute(device: device, logger: logger, useLogState: !capture)
            let library = ShaderLibrary.source(source, enableLogging: !capture)
            var values = (0..<1_000_000).map { Float($0) }.sorted()
            //
            values.swapAt(1, 2)
            //
            let isSorted = try device.makeBuffer(bytesOf: UInt32(1), options: [.storageModeShared])
            isSorted.label = "isSorted"

            var pipeline = try compute.makePipeline(function: library.is_sorted)
            pipeline.arguments.input = .buffer(values, label: "input")
            pipeline.arguments.count = .int(values.count)
            pipeline.arguments.isSorted = .buffer(isSorted)

            try compute.run(pipeline: pipeline, threads: [values.count, 1, 1], threadsPerThreadgroup: [pipeline.maxTotalThreadsPerThreadgroup, 1, 1])

            print(isSorted.contentsBuffer(of: UInt32.self)[0])
        }

    }

    static func complex() async throws {
        let capture = false

        let device = MTLCreateSystemDefaultDevice()!
        try device.capture(enabled: capture) {
        let logger = Logger()
        let compute = try Compute(device: device, logger: logger, useLogState: !capture)
        let library = ShaderLibrary.source(source, enableLogging: !capture)
            var values = (0..<2_000).map { Float($0) }.sorted()
//
            values.swapAt(1, 2)
//
            let isSorted = try device.makeBuffer(bytesOf: UInt32(1), options: [.storageModeShared])
            isSorted.label = "isSortedComplex"

            var pipeline = try compute.makePipeline(function: library.is_sorted_complex)
            pipeline.arguments.input = .buffer(values, label: "input")
            pipeline.arguments.count = .int(values.count)
            pipeline.arguments.isSorted = .buffer(isSorted)

            let threadsPerThreadgroup = pipeline.maxTotalThreadsPerThreadgroup

            try compute.run(pipeline: pipeline, threads: [(values.count + threadsPerThreadgroup - 1) / threadsPerThreadgroup, 1, 1], threadsPerThreadgroup: [threadsPerThreadgroup, 1, 1])

            print(isSorted.contentsBuffer(of: UInt32.self)[0])
        }

    }
}
