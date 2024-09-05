import Compute
import Metal
import MetalSupport
import os

struct YAPrefixSum: Demo {
    static func main() async throws {

        let device = MTLCreateSystemDefaultDevice()!
        try device.capture(enabled: false) {

            let demo = try Self()
            //        try demo.slow()
            try demo.simd()
        }
    }

    let device: MTLDevice
    let count: Int
    let input: TypedMTLBuffer<UInt32>
    let output: TypedMTLBuffer<UInt32>
    let compute: Compute
    let library: ShaderLibrary

    init() throws {
        device = MTLCreateSystemDefaultDevice()!
        count = 32
        // let input = try device.makeTypedBuffer(data: Array<UInt32>(1...UInt32(count)))
//        input = try device.makeTypedBuffer(data: Array<UInt32>((1...count).map({ _ in .random(in: 0..<10) })))
        // let input = try device.makeTypedBuffer(data: Array<UInt32>((1...count).map({ $0 % 3 })))
        input = try device.makeTypedBuffer(data: Array<UInt32>(repeating: 1, count: count))
        output = try device.makeTypedBuffer(data: Array<UInt32>(repeating: 0, count: input.count))
        compute = try Compute(device: device, logger: nil)
        library = ShaderLibrary.bundle(.module, name: "debug")
    }

    func slow() throws {
        var pipeline = try compute.makePipeline(function: library.function(name: "YAPrefixSum::prefix_sum_slow"))
        pipeline.arguments.input = .buffer(input)
        pipeline.arguments.count = .int(UInt32(count))
        pipeline.arguments.output = .buffer(output)
        try timeit("GPU") {
            try device.capture {
                try compute.run(pipeline: pipeline, width: input.count)
            }
        }
        let expectedResult = timeit("CPU") {
            Array(input).prefixSum()
        }
        let result = Array(output)
//        print("EXPECTED", Array(expectedResult.chunks(ofCount: 32).map(Array.init)).prefix(3))
//        print("RESULT", Array(result.chunks(ofCount: 32)).map(Array.init).prefix(3))
        print(result == expectedResult)
    }


    func simd() throws {
        var pipeline = try compute.makePipeline(function: library.function(name: "YAPrefixSum::prefix_sum_simd"))
        pipeline.arguments.input = .buffer(input)
        pipeline.arguments.count = .int(UInt32(count))
        pipeline.arguments.output = .buffer(output)

        let totals_count = ceildiv(count, pipeline.threadExecutionWidth)

        pipeline.arguments.totals = .threadgroupMemory(type: UInt32.self, count: totals_count)
        pipeline.arguments.offsets = .threadgroupMemory(type: UInt32.self, count: totals_count)
        pipeline.arguments.totals_count = .int(totals_count)

//        threadgroup uint *totals [[threadgroup(0)]],
//        threadgroup uint *offsets [[threadgroup(1)]],
//        device uint *totals_count [[buffer(3)]]

//        assert(count <= pipeline.maxTotalThreadsPerThreadgroup)

        try timeit("GPU") {
            try compute.run(pipeline: pipeline, width: input.count)
        }
        let result = Array(output)
        let expectedResult = timeit("CPU") {
            Array(input).prefixSum()
        }
//        print("EXPECTED", Array(expectedResult.chunks(ofCount: 32).map(Array.init)))
//        print("RESULT", Array(result.chunks(ofCount: 32)).map(Array.init).map(\.description).joined(separator: "\n"))
        print(result == expectedResult)
    }
}
