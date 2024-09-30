import AppKit
import Compute
import Metal
import MetalSupportLite
import os
import TabularData

struct YAPrefixSum: Demo {

    static let logging = false
    static let capture = false
    static let exporting = false

    static func main() async throws {
        let device = MTLCreateSystemDefaultDevice()!

        let count = 1_500_000
//        let values = Array<UInt32>(repeating: 1, count: count)
//        let values = Array<UInt32>(1...UInt32(count))
        let values = Array<UInt32>((1...count).map({ _ in .random(in: 0..<10) }))

        let input = try device.makeTypedBuffer(data: values).labelled("Input")
        let expectedResult = timeit("CPU") { values.prefixSum() }

        let demo = try Self(device: device)

        let output = try timeit("GPU") {
            try demo.simd2(input: input)
        }
        let result = Array(output)

        print(result == expectedResult)

        if exporting {
            var dataFrame = DataFrame()
            dataFrame.append(column: Column(name: "index", contents: 0..<result.count))
            dataFrame.append(column: Column(name: "expectedResult", contents: expectedResult))
            dataFrame.append(column: Column(name: "result", contents: result))
            let url = URL(filePath: "YAPrefixSum.csv")
            try dataFrame.writeCSV(to: url)
            NSWorkspace.shared.selectFile(url.path, inFileViewerRootedAtPath: "/")
        }
    }


    let device: MTLDevice
    let compute: Compute
    let library: ShaderLibrary

    init(device: MTLDevice) throws {
        self.device = device
        compute = try Compute(device: device, logger: Logger(), logging: YAPrefixSum.logging)
        library = ShaderLibrary.bundle(.module, name: "debug")
    }

    func slow(input: TypedMTLBuffer<UInt32>) throws -> TypedMTLBuffer<UInt32> {
        let output = try device.makeTypedBuffer(data: Array<UInt32>(repeating: 0, count: input.count))
        var pipeline = try compute.makePipeline(function: library.function(name: "YAPrefixSum::prefix_sum_slow"))
        pipeline.arguments.inputs = .buffer(input)
        pipeline.arguments.count = .int(UInt32(input.count))
        pipeline.arguments.outputs = .buffer(output)
        try compute.run(pipeline: pipeline, threads: MTLSize(width: input.count), threadsPerThreadgroup: pipeline.calculateThreadgroupSize(threads: MTLSize(width: input.count)))
        return output
    }

    func simd2(input: TypedMTLBuffer<UInt32>) throws -> TypedMTLBuffer<UInt32> {
        var prefixSumPipeline = try compute.makePipeline(function: library.function(name: "YAPrefixSum::prefix_sum_simd"))
        var gatherPipeline = try compute.makePipeline(function: library.function(name: "YAPrefixSum::gather_totals"))
        var applyOffsetsPipeline = try compute.makePipeline(function: library.function(name: "YAPrefixSum::apply_offsets"))
        let chunkSize = prefixSumPipeline.maxTotalThreadsPerThreadgroup
        let chunkCount = ceildiv(input.count, chunkSize)
        let bufferA: [TypedMTLBuffer<UInt32>] = try stride(from: input.count, dividingBy: chunkSize).map {
            try device.makeTypedBuffer(capacity: $0).labelled("BufferA-\($0)")
        }
        let bufferB: TypedMTLBuffer<UInt32> = try device.makeTypedBuffer(capacity: ceildiv(input.count, prefixSumPipeline.threadExecutionWidth)).labelled("BufferB")
        let bufferC: TypedMTLBuffer<UInt32> = try device.makeTypedBuffer(capacity: ceildiv(input.count, prefixSumPipeline.threadExecutionWidth)).labelled("BufferC")
        let bufferD: TypedMTLBuffer<UInt32> = try device.makeTypedBuffer(capacity: chunkCount).labelled("BufferD")

//        let total = bufferA.map { $0.count }.reduce(0, +) + bufferB.count + bufferC.count + bufferD.count
//        print(Double(total) / Double(input.count))



        func internal_simd2(dispatch: Compute.Dispatcher, input: TypedMTLBuffer<UInt32>, level: Int, count: Int) throws -> TypedMTLBuffer<UInt32> {
            let output = bufferA[level]
            prefixSumPipeline.arguments.inputs = .buffer(input)
            prefixSumPipeline.arguments.count = .int(UInt32(count))
            prefixSumPipeline.arguments.outputs = .buffer(output)
            prefixSumPipeline.arguments.totals = .buffer(bufferB)
            prefixSumPipeline.arguments.offsets = .buffer(bufferC)
            try dispatch(pipeline: prefixSumPipeline, threads: MTLSize(width: count), threadsPerThreadgroup: prefixSumPipeline.calculateThreadgroupSize(threads: MTLSize(width: count)))

            if count <= prefixSumPipeline.maxTotalThreadsPerThreadgroup {
                return output
            }

            let chunkSize = prefixSumPipeline.maxTotalThreadsPerThreadgroup
            let chunkCount = ceildiv(count, chunkSize)
            gatherPipeline.arguments.inputs = .buffer(input)
            gatherPipeline.arguments.outputs = .buffer(output)
            gatherPipeline.arguments.chunk_size = .int(chunkSize)
            gatherPipeline.arguments.totals = .buffer(bufferD)
            try dispatch(pipeline: gatherPipeline, threads: MTLSize(width: chunkCount), threadsPerThreadgroup: gatherPipeline.calculateThreadgroupSize(threads: MTLSize(width: chunkCount)))

            if count <= gatherPipeline.maxTotalThreadsPerThreadgroup {
                return output
            }

            let offsets = try internal_simd2(dispatch: dispatch, input: bufferD, level: level + 1, count: chunkCount)

            applyOffsetsPipeline.arguments.outputs = .buffer(output)
            applyOffsetsPipeline.arguments.chunk_size = .int(chunkSize)
            applyOffsetsPipeline.arguments.offsets = .buffer(offsets)

            try dispatch(pipeline: applyOffsetsPipeline, threads: MTLSize(width: count), threadsPerThreadgroup: applyOffsetsPipeline.calculateThreadgroupSize(threads: MTLSize(width: count)))

            return output
        }

        return try device.capture(enabled: Self.capture) {
            try compute.task { task in
                try task { dispatch in
                    return try internal_simd2(dispatch: dispatch, input: input, level: 0, count: input.count)
                }
            }
        }
    }


}
