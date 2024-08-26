//import AppKit
//import Compute
//import CoreGraphics
//import Foundation
//import Metal
//
//struct CountingSort {
//    let device = MTLCreateSystemDefaultDevice()!
//
//    func main() throws {
//        let capture = false
//
//        let maxValue: UInt16 = 65535
//
//        let values = (0..<1).map { _ in UInt16.random(in: 0 ... maxValue - 1) }
//        //        let values: [UInt16] = [3, 1, 4, 2, 5, 6, 7, 8, 9, 0]
//        //        print("Values", values)
//        let expectedResult = timeit("Foundation sort") { values.sorted() }
//        //        print("Expected", expectedResult)
//        let cpuHistograms = Array(values.histogram().map(UInt32.init)[..<Int(maxValue)])
//
//        let compute = try Compute(device: device, logState: capture ? nil : try device.makeDefaultLogState())
//        let library = ShaderLibrary.bundle(.module, name: "debug")
//
//        let input = try device.makeBuffer(bytesOf: values, options: [])
//        let output = try device.makeBuffer(bytesOf: Array(repeating: UInt16.zero, count: values.count), options: [])
//        let histogram = try device.makeBuffer(bytesOf: Array(repeating: UInt32.zero, count: Int(maxValue)), options: [])
//
//        var histogramPass = try compute.makePass(function: library.function(name: "CountingSort16::histogram"))
//        var prefixSumPass = try compute.makePass(function: library.function(name: "CountingSort16::prefix_sum_exclusive"))
//        var shufflePass = try compute.makePass(function: library.function(name: "CountingSort16::shuffle"))
//
//        try timeit("GPU Counting Sort") {
//            try device.capture(enabled: capture) {
//                histogramPass.arguments.histogram = .buffer(histogram)
//                histogramPass.arguments.input = .buffer(input)
//                histogramPass.arguments.histogramCount = .int(UInt32(maxValue))
//
//                try compute.dispatch(label: "Histogram") { dispatch in
//                    let maxTotalThreadsPerThreadgroup = histogramPass.maxTotalThreadsPerThreadgroup
//                    try dispatch(pass: histogramPass, threads: MTLSize(width: Int(maxValue), height: values.count), threadsPerThreadgroup: MTLSize(width: maxTotalThreadsPerThreadgroup, height: 1))
//                }
//
//                print(cpuHistograms.firstIndex(of: 1))
//                print(histogram.as(UInt32.self).firstIndex(of: 1))
//
//                assert(histogram.as(UInt32.self) == cpuHistograms)
//
//                //            // Prefix sum.
//                //            prefixSumPass.arguments.histogram = .buffer(histogram)
//                //            prefixSumPass.arguments.histogramCount = .int(UInt32(maxValue))
//                //            try compute.dispatch(label: "PrefixSum") { dispatch in
//                //                try dispatch(pass: prefixSumPass, threads: MTLSize(width: 1), threadsPerThreadgroup: MTLSize(width: 1))
//                //            }
//                ////            print("Prefix Sum", histogram.as(UInt32.self))
//                //
//                //            // Shuffle.
//                //            shufflePass.arguments.histogram = .buffer(histogram)
//                //            shufflePass.arguments.input = .buffer(input)
//                //            shufflePass.arguments.output = .buffer(output)
//                //            shufflePass.arguments.count = .int(values.count)
//                //            try compute.dispatch(label: "Shuffle") { dispatch in
//                //                let maxTotalThreadsPerThreadgroup = shufflePass.maxTotalThreadsPerThreadgroup
//                //
//                //                try dispatch(pass: shufflePass, threads: MTLSize(width: values.count), threadsPerThreadgroup: MTLSize(width: maxTotalThreadsPerThreadgroup))
//                //                }
//            }
//        }
//        ////        print(input.as(UInt16.self))
//        ////
//        ////        print(output.as(UInt16.self))
//        //        print("GPU Sorted?", expectedResult == output.as(UInt16.self))
//    }
//}
