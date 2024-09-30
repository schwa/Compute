import Compute
import Metal
import MetalSupportLite
import os

struct CountingSortDemo {

    static let logging = false
    static let capture = false

    var compute: Compute
    var histogramPipeline: Compute.Pipeline
    var prefixSumPipeline: Compute.Pipeline
    var shufflePipeline: Compute.Pipeline

    init(compute: Compute) throws {
        self.compute = compute
        let library = ShaderLibrary.bundle(.module, name: "debug")
        histogramPipeline = try compute.makePipeline(function: library.function(name: "Histogram::histogram3"))
        prefixSumPipeline = try compute.makePipeline(function: library.function(name: "YAPrefixSum::prefix_sum_exclusive_slow"))
        shufflePipeline = try compute.makePipeline(function: library.function(name: "CountingSort::shuffle5"))
    }

    func radixSort(dispatch: Compute.Dispatcher, input: TypedMTLBuffer<UInt32>) throws -> TypedMTLBuffer<UInt32> {
        var input = input
        let device = compute.device
        var output: TypedMTLBuffer<UInt32> = try device.makeTypedBuffer(capacity: input.count)
        for phase in 0...3 {
            let shift = phase * 8
            try countingSort(dispatch: dispatch, input: input, output: output, shift: shift)
            swap(&input, &output)
        }
        return input
    }

    func countingSort1(dispatch: Compute.Dispatcher, input: TypedMTLBuffer<UInt32>, shift: Int) throws -> TypedMTLBuffer<UInt32> {
        let device = compute.device
        let output: TypedMTLBuffer<UInt32> = try device.makeTypedBuffer(capacity: input.count)
        try countingSort(dispatch: dispatch, input: input, output: output, shift: shift)
        return output
    }

    func countingSort(dispatch: Compute.Dispatcher, input: TypedMTLBuffer<UInt32>, output: TypedMTLBuffer<UInt32>, shift: Int) throws {
        var histogramPipeline = histogramPipeline
        var prefixSumPipeline = prefixSumPipeline
        var shufflePipeline = shufflePipeline
        let device = compute.device

        let histogram: TypedMTLBuffer<UInt32> = try device.makeTypedBuffer(capacity: 256)
        histogramPipeline.arguments.input = .buffer(input)
        histogramPipeline.arguments.output = .buffer(histogram)
        histogramPipeline.arguments.count = .int(input.count)
        histogramPipeline.arguments.shift = .int(shift)

        try dispatch(pipeline: histogramPipeline, threads: MTLSize(width: input.count), threadsPerThreadgroup: histogramPipeline.calculateThreadgroupSize(threads: MTLSize(width: input.count)))

        let summedHistogram: TypedMTLBuffer<UInt32> = try device.makeTypedBuffer(capacity: 256)
        prefixSumPipeline.arguments.inputs = .buffer(histogram)
        prefixSumPipeline.arguments.outputs = .buffer(summedHistogram)
        prefixSumPipeline.arguments.count = .int(histogram.count)

        try dispatch(pipeline: prefixSumPipeline, threads: MTLSize(width: 256), threadsPerThreadgroup: prefixSumPipeline.calculateThreadgroupSize(threads: MTLSize(width: 256)))

        shufflePipeline.arguments.input = .buffer(input)
        shufflePipeline.arguments.count = .int(input.count)
        shufflePipeline.arguments.output = .buffer(output)
        shufflePipeline.arguments.shift = .int(shift)
        shufflePipeline.arguments.histogram = .buffer(summedHistogram)
        try dispatch(pipeline: shufflePipeline, threads: MTLSize(width: 256), threadsPerThreadgroup: shufflePipeline.calculateThreadgroupSize(threads: MTLSize(width: 256)))

    }
}

extension CountingSortDemo: Demo {

    static func main() async throws {
        let device = MTLCreateSystemDefaultDevice()!
        let compute = try Compute(device: device, logger: Logger(), logging: logging)
        let count = 150_000
//        let count = 1_500_000
        let elements: [UInt32] = (0..<count).map { _ in UInt32.random(in: 0..<(UInt32.max)) }
//        let elements: [UInt32] = (0..<count).map { UInt32(count - $0 - 1) }
        let input = try device.makeTypedBuffer(data: elements)
        let sort = try CountingSortDemo(compute: compute)

        let output = try device.capture(enabled: Self.capture) {
            try compute.task { task in
                try task { dispatch in
                    try sort.radixSort(dispatch: dispatch, input: input)
                }
            }
        }

        let cpu = RadixSortCPU().radixSort(input: elements)
        print(cpu == elements.sorted())
        let gpu = Array(output)
        print(gpu == elements.sorted())
        print(cpu == gpu)
    }
}
