import Foundation

postfix operator ++

extension BinaryInteger {
    static postfix func ++(rhs: inout Self) -> Self {
        let oldValue = rhs
        rhs += 1
        return oldValue
    }
}

// From: "A High-Performance Implementation of Counting Sort on CUDA GPU"

struct RadixSortCPU {

    func key(_ value: UInt32, shift: Int) -> Int {
        (Int(value) >> shift) & 0xFF
    }

    func histogram(input: [UInt32], shift: Int) -> [UInt32] {
        input.reduce(into: Array(repeating: 0, count: 256)) { result, value in
            result[key(value, shift: shift)] += 1
        }
    }

    func prefixSumExclusive(_ input: [UInt32]) -> [UInt32] {
        input.prefixSumExclusive()
    }

    func shuffle(_ input: [UInt32], summedHistogram histogram: [UInt32], shift: Int, output: inout [UInt32]) {
        var histogram = histogram
        for i in input.indices {
            let value = input[i]
            let key = key(value, shift: shift)
            let outputIndex = histogram[key]++
            output[Int(outputIndex)] = input[i]
        }
    }

    func countingSort(input: [UInt32], shift: Int, output: inout [UInt32]) {
        let histogram = histogram(input: input, shift: shift)
        let summedHistogram = prefixSumExclusive(histogram)
        shuffle(input, summedHistogram: summedHistogram, shift: shift, output: &output)
    }

    func radixSort(input: [UInt32]) -> [UInt32] {
        var input = input
        var output = Array(repeating: UInt32.zero, count: input.count)
        for phase in 0..<4 {
            countingSort(input: input, shift: phase * 8, output: &output)
            swap(&input, &output)
        }
        return input
    }
}
