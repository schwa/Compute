import Compute
import Metal
import os

enum CountingSortDemo: Demo {

    static let logging = false
    static let capture = true

    static func main() async throws {
        let device = MTLCreateSystemDefaultDevice()!
        let compute = try Compute(device: device, logger: Logger(), logging: logging)
        let library = ShaderLibrary.bundle(.module, name: "debug")
        let count = 1_500_000
        let elements: [UInt32] = (0..<count).map { _ in UInt32.random(in: 0..<1_000_000) }
        //        let elements: [UInt32] = (0..<count).map { UInt32(count - $0 - 1) }
        //        print("Input", elements)

        var histogramPipeline = try compute.makePipeline(function: library.function(name: "CountingSort::histogram"))
        var prefixSumPipeline = try compute.makePipeline(function: library.function(name: "YAPrefixSum::prefix_sum_inclusive_slow"))
        var shufflePipeline = try compute.makePipeline(function: library.function(name: "CountingSort::shuffle"))

        var input = try device.makeTypedBuffer(data: elements)
        var output: TypedMTLBuffer<UInt32> = try device.makeTypedBuffer(count: input.count)
        try device.capture(enabled: capture) {
            try compute.task { task in
                try task { dispatch in

                    for phase in 0..<4 {
                        let shift = phase * 8

                        let histogram: TypedMTLBuffer<UInt32> = try device.makeTypedBuffer(count: 256)
                        histogramPipeline.arguments.input = .buffer(input)
                        histogramPipeline.arguments.output = .buffer(histogram)
                        histogramPipeline.arguments.count = .int(input.count)
                        histogramPipeline.arguments.shift = .int(shift)

                        try dispatch(pipeline: histogramPipeline, threads: MTLSize(width: input.count), threadsPerThreadgroup: histogramPipeline.calculateThreadgroupSize(threads: MTLSize(width: input.count)))

                        let summedHistogram: TypedMTLBuffer<UInt32> = try device.makeTypedBuffer(count: 256)
                        prefixSumPipeline.arguments.inputs = .buffer(histogram)
                        prefixSumPipeline.arguments.outputs = .buffer(summedHistogram)
                        prefixSumPipeline.arguments.count = .int(histogram.count)

                        try dispatch(pipeline: prefixSumPipeline, threads: MTLSize(width: 256), threadsPerThreadgroup: prefixSumPipeline.calculateThreadgroupSize(threads: MTLSize(width: 256)))

                        assert(Array(summedHistogram) == Array(histogram).prefixSumInclusive())

                        shufflePipeline.arguments.input = .buffer(input)
                        shufflePipeline.arguments.count = .int(input.count)
                        shufflePipeline.arguments.output = .buffer(output)
                        shufflePipeline.arguments.shift = .int(shift)
                        shufflePipeline.arguments.histogram = .buffer(summedHistogram)
                        try dispatch(pipeline: shufflePipeline, threads: MTLSize(width: input.count), threadsPerThreadgroup: shufflePipeline.calculateThreadgroupSize(threads: MTLSize(width: input.count)))

                        swap(&input, &output)
                    }
                }
            }
        print(Array(output) == elements.sorted())
        }
    }
}
