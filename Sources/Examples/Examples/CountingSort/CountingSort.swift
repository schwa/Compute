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

// MARK: -

struct KeyedIndex {
    var key: UInt32
    var index: UInt32
}

extension Collection where Element == KeyedIndex {
    func histogram(shift: Int) -> [Int] {
        func key(_ value: Element) -> Int {
            (Int(value.key) >> shift) & 0xFF
        }
        return reduce(into: Array<Int>(repeating: 0, count: 256)) { result, value in
            result[key(value)] += 1
        }
    }
}

// MARK: Prefix Sum

extension Array where Element: BinaryInteger {
    @inline(__always) func prefixSumInclusive() -> [Element] {
        var copy = self
        for index in copy.indices.dropFirst() {
            copy[index] += copy[index - 1]
        }
        return copy
    }
}

// MARK: Shuffle

extension MutableCollection where Element == KeyedIndex, Index == Int {
    func shuffle(shift: Int, histogram: [Int], output: inout [Element]) {
        func key(_ value: Element) -> Int {
            (Int(value.key) >> shift) & 0xFF
        }
        var histogram = histogram
        for index in stride(from: self.count - 1, through: 0, by: -1) {
            let value = key(self[index])
            histogram[value] -= 1
            output[histogram[value]] = self[index]
        }
    }
}

// MARK: Counting Sort

extension MutableCollection where Element == KeyedIndex, Index == Int {
    func countingSort(shift: Int, output: inout [Element]) {
        func key(_ value: Element) -> Int {
            (Int(value.key) >> shift) & 0xFF
        }
        // Histogram
        var histogram = histogram(shift: shift)
        // Prefix Sum
        histogram = histogram.prefixSumInclusive()
        // Shuffle
        shuffle(shift: shift, histogram: histogram, output: &output)
    }
}

// MARK: Radix Sort

extension Array where Element == KeyedIndex, Index == Int {
    func radixSorted() -> [Element] {
        var input = self
        var output = self
        for phase in 0..<4 {
            input.countingSort(shift: phase * 8, output: &output)
            swap(&input, &output)
        }
        return input
    }
}

//@main
//enum CLI {
//    static func main() {
//        let values = (0..<1_500_000).map { value in
//            KeyedIndex(key: UInt32.random(in: 0..<UInt32.max), index: value)
//        }
//        let sortedValues = timeit("** Foundation") { values.map(\.key).sorted() }
//        let radixSorted = timeit("** Radix Sort") {
//            values.radixSorted()
//        }
//        assert(sortedValues == radixSorted.map(\.key))
//    }
//}
