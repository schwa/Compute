//import AppKit
//import Compute
//import CoreGraphics
//import Foundation
//import Metal
//
//struct RadixSort {
//    let device = MTLCreateSystemDefaultDevice()!
//
//    func main() throws {
//        let capture = false
//
//        let values = (0..<1_500_000).map { _ in UInt32.random(in: 0 ... 1000) }
//        let expectedResult = timeit("Foundation sort") { values.sorted() }
//        let cpuSorted = timeit("CPU Radix Sort") { radixSort(values: values) }
//        print("CPU Sorted?", expectedResult == cpuSorted)
//
//        let compute = try Compute(device: device, logState: capture ? nil : try device.makeDefaultLogState())
//        let library = ShaderLibrary.bundle(.module, name: "debug")
//
//        var input = try device.makeBuffer(bytesOf: values, options: [])
//        var output = try device.makeBuffer(bytesOf: Array(repeating: UInt32.zero, count: values.count), options: [])
//        let histogram = try device.makeBuffer(bytesOf: Array(repeating: UInt32.zero, count: 256), options: [])
//
//        var histogramPass = try compute.makePass(function: library.histogram)
//        var prefixSumPass = try compute.makePass(function: library.prefix_sum_exclusive)
//        var shufflePass = try compute.makePass(function: library.shuffle2)
//
//        try timeit("GPU Radix Sort") {
//            try device.capture(enabled: capture) {
//                for phase in 0..<4 {
//                    let shift = UInt32(phase * 8)
//
//                    histogramPass.arguments.histogram = .buffer(histogram)
//                    histogramPass.arguments.shift = .int(shift)
//                    histogramPass.arguments.input = .buffer(input)
//
//                    try compute.dispatch(label: "Histogram") { dispatch in
//                        let maxTotalThreadsPerThreadgroup = histogramPass.maxTotalThreadsPerThreadgroup
//                        try dispatch(pass: histogramPass, threads: MTLSize(width: 256, height: values.count), threadsPerThreadgroup: MTLSize(width: maxTotalThreadsPerThreadgroup, height: 1))
//                    }
//
//                    // Prefix sum.
//                    prefixSumPass.arguments.histogram = .buffer(histogram)
//                    try compute.dispatch(label: "PrefixSum") { dispatch in
//                        try dispatch(pass: prefixSumPass, threads: MTLSize(width: 1), threadsPerThreadgroup: MTLSize(width: 1))
//                    }
//
//                    // Shuffle.
//                    shufflePass.arguments.histogram = .buffer(histogram)
//                    shufflePass.arguments.input = .buffer(input)
//                    shufflePass.arguments.output = .buffer(output)
//                    shufflePass.arguments.count = .int(values.count)
//                    shufflePass.arguments.shift = .int(shift)
//                    try compute.dispatch(label: "Shuffle") { dispatch in
//                        let maxTotalThreadsPerThreadgroup = shufflePass.maxTotalThreadsPerThreadgroup
//
//                        try dispatch(pass: shufflePass, threads: MTLSize(width: 256), threadsPerThreadgroup: MTLSize(width: maxTotalThreadsPerThreadgroup))
//                    }
//
//                    swap(&input, &output)
//
//                    histogram.clear()
//                }
//            }
//        }
//        print("GPU Sorted?", expectedResult == input.as(UInt32.self))
//    }
//}
