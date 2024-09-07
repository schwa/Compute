import Foundation

postfix operator ++

extension Int {
    static postfix func ++(rhs: inout Int) -> Int {
        let oldValue = rhs
        rhs += 1
        return oldValue
    }
}

// From: "A High-Performance Implementation of Counting Sort on CUDA GPU"
func countingSort(input: [UInt32], shift: Int, output: inout [UInt32]) {
    func key(_ value: UInt32) -> Int {
        (Int(value) >> shift) & 0xFF
    }
    // Histogram
    var histogram = input.reduce(into: Array(repeating: 0, count: 256)) { result, value in
        result[key(value)] += 1
    }

    histogram = histogram.prefixSumExclusive()

    // Shuffle
    for i in input.indices {
        let value = key(input[i])
        output[histogram[value]++] = input[i]
    }
}

func radixSort(values: [UInt32]) -> [UInt32] {
    var input = values
    var output = Array(repeating: UInt32.zero, count: input.count)
    for phase in 0..<4 {
        countingSort(input: input, shift: phase * 8, output: &output)
        swap(&input, &output)
        //        print("Phase: \(phase) \(input.map({ String($0, radix: 16) }))")
    }
    return input
}
