import Compute
import Metal
import os

enum YAPrefixSum: Demo {
    static func main() async throws {
        let device = MTLCreateSystemDefaultDevice()!
        let compute = try Compute(device: device, logger: Logger())
        let library = ShaderLibrary.bundle(.module, name: "debug")
        var prefixSumSmallSlow = try compute.makePipeline(function: library.function(name: "YAPrefixSum::prefix_sum_small_slow"))

        let count = 999981
//        let input = try device.makeTypedBuffer(data: Array<UInt32>(1...UInt32(count)))
        let input = try device.makeTypedBuffer(data: Array<UInt32>((1...count).map({ _ in .random(in: 0..<10) })))
//        let input = try device.makeTypedBuffer(data: Array<UInt32>((1...count).map({ $0 % 3 })))
//        let input = try device.makeTypedBuffer(data: Array<UInt32>(repeating: 1, count: count))
        let output = try device.makeTypedBuffer(data: Array<UInt32>(repeating: 0, count: input.count))
        prefixSumSmallSlow.arguments.input = .buffer(input)
        prefixSumSmallSlow.arguments.count = .int(UInt32(count))
        prefixSumSmallSlow.arguments.output = .buffer(output)
        try compute.run(pipeline: prefixSumSmallSlow, width: input.count)

        let expectedResult = Array(input).prefixSum()

        let result = Array(output)
        print("EXPECTED", Array(expectedResult.chunks(ofCount: 32).map(Array.init)).prefix(3))
        print("RESULT", Array(result.chunks(ofCount: 32)).map(Array.init).prefix(3))
        print(result == expectedResult)

    }
}
