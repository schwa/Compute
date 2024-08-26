import Foundation

postfix operator ++

extension Int {
    static postfix func ++(rhs: inout Int) -> Int {
        let oldValue = rhs
        rhs += 1
        return oldValue
    }
}

func shuffle(input: [UInt32], shift: Int, histogram: [UInt32]) -> [UInt32] {
    func key(_ value: UInt32) -> Int {
        (Int(value) >> shift) & 0xFF
    }
    var histogram = histogram.map { Int($0) }
    var output = Array(repeating: UInt32.zero, count: input.count)
    for index in stride(from: input.count - 1, through: 0, by: -1) {
        let value = key(input[index])
        histogram[value] -= 1
        output[histogram[value]] = input[index]
    }
    return output
}

func countingSort(input: [UInt32], shift: Int, output: inout [UInt32]) {
    func key(_ value: UInt32) -> Int {
        (Int(value) >> shift) & 0xFF
    }
    // Histogram
    var histogram = input.reduce(into: Array(repeating: 0, count: 256)) { result, value in
        result[key(value)] += 1
    }
    // Prefix Sum
    for index in histogram.indices.dropFirst() {
        histogram[index] += histogram[index - 1]
    }
    // Shuffle
    for index in stride(from: input.count - 1, through: 0, by: -1) {
        let value = key(input[index])
        histogram[value] -= 1
        output[histogram[value]] = input[index]
    }
}

// From: "A High-Performance Implementation of Counting Sort on CUDA GPU"
func countingSort2(input: [UInt32], shift: Int, output: inout [UInt32]) {
    func key(_ value: UInt32) -> Int {
        (Int(value) >> shift) & 0xFF
    }
    // Histogram
    var histogram = input.reduce(into: Array(repeating: 0, count: 256)) { result, value in
        result[key(value)] += 1
    }

    // Prefix Sum (exclusive)
    var sum = 0
    for i in histogram.indices {
        let t = histogram[i]
        histogram[i] = sum
        sum += t
    }

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
        countingSort2(input: input, shift: phase * 8, output: &output)
        swap(&input, &output)
        //        print("Phase: \(phase) \(input.map({ String($0, radix: 16) }))")
    }
    return input
}
