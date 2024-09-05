import Algorithms
import Compute
import Metal
import MetalSupport
import os

enum SIMDPrefixExclusiveSumDemo: Demo {
    static let source = #"""
        #include <metal_stdlib>
        #include <metal_logging>

        using namespace metal;

        const uint thread_position_in_grid [[thread_position_in_grid]];
        const uint threads_per_simdgroup [[threads_per_simdgroup]];

        kernel void sum(
            constant uint *input [[buffer(0)]],
            device uint *output [[buffer(1)]],
            device uint *totals [[buffer(2)]]

        ) {
            const uint simd_position_in_grid = thread_position_in_grid / threads_per_simdgroup;

            output[thread_position_in_grid] = simd_prefix_exclusive_sum(input[thread_position_in_grid]);

            simdgroup_barrier(mem_flags::mem_threadgroup);

            if (simd_is_first() && simd_position_in_grid != 0) {
                uint index = simd_position_in_grid * threads_per_simdgroup - 1;
                totals[simd_position_in_grid] = output[index] + input[index];
            }
        }

        kernel void add_totals(
            device uint *output [[buffer(0)]],
            constant uint *totals [[buffer(1)]]

        ) {
            output[thread_position_in_grid] += totals[thread_position_in_grid / threads_per_simdgroup];
        }
"""#

    static func main() async throws {
        let device = MTLCreateSystemDefaultDevice()!
        let compute = try Compute(device: device, logger: Logger())
        let library = ShaderLibrary.source(source, enableLogging: true)
        var sumPipeline = try compute.makePipeline(function: library.sum)
        let count = 65
        let input = try device.makeTypedBuffer(data: Array<UInt32>(1...UInt32(count)))
//        let input = try device.makeTypedBuffer(data: Array<UInt32>((1...count).map({ _ in .random(in: 0..<10) })))
//        let input = try device.makeTypedBuffer(data: Array<UInt32>((1...count).map({ $0 % 3 })))
//        let input = try device.makeTypedBuffer(data: Array<UInt32>(repeating: 1, count: count))
        let output = try device.makeTypedBuffer(data: Array<UInt32>(repeating: 0, count: input.count))
        let totals = try device.makeTypedBuffer(data: Array<UInt32>(repeating: 0, count: ceildiv(input.count, sumPipeline.threadExecutionWidth)))
        sumPipeline.arguments.input = .buffer(input)
        sumPipeline.arguments.output = .buffer(output)
        sumPipeline.arguments.totals = .buffer(totals)
        try compute.run(pipeline: sumPipeline, width: input.count)
        let expectedResult = Array(input).prefixSum()
        let intermediate = Array(output)

        var totalsPipeline = try compute.makePipeline(function: library.add_totals)
        totalsPipeline.arguments.output = .buffer(output)
        totalsPipeline.arguments.totals = .buffer(totals)
        try compute.run(pipeline: totalsPipeline, width: input.count)

        let result = Array(output)
        print("INTERMEDIATE", Array(intermediate.chunks(ofCount: 32)).map(Array.init))
        print("TOTALS", Array(totals))
        print("EXPECTED", Array(expectedResult.chunks(ofCount: 32).map(Array.init)))
        print("RESULT", Array(result.chunks(ofCount: 32)).map(Array.init))
        print(result == expectedResult)


    }
}
