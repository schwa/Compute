import Compute
import CoreImage
import Metal
import MetalKit
import os
import UniformTypeIdentifiers

enum MaxValue {

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



    // TODO: This will _occasionally_ fail and it's unclear why!???
    static func multipass(values: [Int32], expectedValue: Int32) throws {
        let source = #"""
            #include <metal_stdlib>

            using namespace metal;

            uint thread_position_in_grid [[thread_position_in_grid]];
            uint threads_per_simdgroup [[threads_per_simdgroup]];
            uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]];

            kernel void maxValue(
                device int *input [[buffer(0)]],
                constant uint &count [[buffer(1)]],
                constant uint &span [[buffer(2)]]
            ) {
                const uint index = thread_position_in_grid * span;
                uint value = index >= count ? -1 : input[index];
                for (uint offset = threads_per_simdgroup >> 1; offset > 0; offset >>= 1) {
                    const uint remoteValue = simd_shuffle_down(value, offset);
                    value = max(value, remoteValue);
                }
                if (simd_is_first()) {
                    input[index] = value;
                }
            }
        """#
        let device = MTLCreateSystemDefaultDevice()!
        let input = device.makeBuffer(bytes: values, length: MemoryLayout<Int32>.stride * values.count, options: [])!
        let compute = try Compute(device: device)
        let library = ShaderLibrary.source(source)
        var pipeline = try compute.makePipeline(function: library.maxValue)
        pipeline.arguments.input = .buffer(input)

        let threadsPerSIMDGroup = 32
        let maxTotalThreadsPerThreadgroup = pipeline.computePipelineState.maxTotalThreadsPerThreadgroup

        try timeit(#function) {
            var span = 1

            try compute.task { task in
                try task { dispatch in
                    while span <= values.count {
                        pipeline.arguments.count = .int(Int32(values.count))
                        pipeline.arguments.span = .int(UInt32(span))
                        try dispatch(pipeline: pipeline, threads: MTLSize(width: values.count / span, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: maxTotalThreadsPerThreadgroup, height: 1, depth: 1))
                        span *= threadsPerSIMDGroup
                    }
                }
            }
        }

        let result = input.contents().assumingMemoryBound(to: Int32.self)[0]
        print(result)
        assert(result == expectedValue)
    }



    static func main() throws {
        //        var values = Array(Array(repeating: Int32.zero, count: 1000))
        let expectedValue: Int32 = 123456789
        var values = Array(Int32.zero ..< 1_000_000)
        values[Int.random(in: 0..<values.count)] = expectedValue

        timeit("Array.max") {
            print(values.max())
        }
        try badIdea(values: values, expectedValue: expectedValue)
        try simpleAtomic(values: values, expectedValue: expectedValue)
        try multipass(values: values, expectedValue: expectedValue)
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
