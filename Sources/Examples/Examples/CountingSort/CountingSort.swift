import Compute
import Metal
import os

enum CountingSortDemo: Demo {

    static let logging = true
    static let capture = false

    static func main() async throws {
        let device = MTLCreateSystemDefaultDevice()!
        let compute = try Compute(device: device, logger: Logger(), logging: logging)
        let library = ShaderLibrary.bundle(.module, name: "debug")
        let count = 40
        //let elements: [UInt32] = (0..<count).map { _ in UInt32.random(in: 0..<100) }
        let elements: [UInt32] = (0..<count).map { UInt32(count - $0 - 1) }
        print("Input", elements)

        let shift = 0
        let expectedResult = elements
            .reduce(into: Array(repeating: UInt32.zero, count: 256)) { result, value in
                let value = UInt8(value >> shift & 0xFF)
                result[Int(value)] += 1
            }

        let input = try device.makeTypedBuffer(data: elements)
        let histogram: TypedMTLBuffer<UInt32> = try device.makeTypedBuffer(count: 256)
        var histogramPipeline = try compute.makePipeline(function: library.function(name: "CountingSort::histogram"))
        histogramPipeline.arguments.input = .buffer(input)
        histogramPipeline.arguments.output = .buffer(histogram)
        histogramPipeline.arguments.count = .int(input.count)
        histogramPipeline.arguments.shift = .int(shift)
//        try device.capture(enabled: capture) {
            try compute.run(pipeline: histogramPipeline, width: input.count)
//        }

//        print("Histogram: ", Array(histogram), Array(histogram).reduce(0, +))

        let summedHistogram = try YAPrefixSum(compute: compute).prefixSum(input: histogram, inclusive: true)
        print("Summed (GPU): ", Array(summedHistogram))
        print("Summed (CPU): ", Array(histogram).prefixSumInclusive())

        assert(Array(summedHistogram) == Array(histogram).prefixSumInclusive())

        let output: TypedMTLBuffer<UInt32> = try device.makeTypedBuffer(count: input.count)

        var shufflePipeline = try compute.makePipeline(function: library.function(name: "CountingSort::shuffle1"))
        shufflePipeline.arguments.input = .buffer(input)
        shufflePipeline.arguments.count = .int(input.count)
        shufflePipeline.arguments.output = .buffer(output)
        shufflePipeline.arguments.shift = .int(shift)
        shufflePipeline.arguments.histogram = .buffer(summedHistogram)
        try compute.run(pipeline: shufflePipeline, width: output.count)

        print(Array(output))
        print(Array(output) == elements.sorted())

//        let result = Array(histogram)
//        print(result)
//        print(expectedResult)
//        print(result == expectedResult)

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
