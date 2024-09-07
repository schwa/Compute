@testable import Compute
@testable import Examples
import Metal
import Testing
import os

struct TestPrefixSum {

    func test(values: [UInt32], inclusive: Bool = false) throws -> [UInt32] {
        let device = MTLCreateSystemDefaultDevice()!
        let input = try device.makeTypedBuffer(data: values).labelled("Input")
        let compute = try Compute(device: device, logger: Logger(), logging: true)
        let demo = try YAPrefixSum(compute: compute)
        let output = try demo.prefixSum(input: input, inclusive: inclusive)
        let result = Array(output)
        return result
    }

    @Test
    func somePrefixSumSmall() throws {
//        #expect(try test(values: []).isEmpty)
        #expect(try test(values: [0]) == [0])
        #expect(try test(values: [1, 1, 1, 1]) == [0, 1, 2, 3])
        #expect(try test(values: [1, 2, 3, 4]) == [0, 1, 3, 6])
        #expect(try test(values: [1, 1, 1, 1, 0, 0, 0, 0]) == [0, 1, 2, 3, 4, 4, 4, 4])
        #expect(try test(values: [0, 0, 0, 0, 1, 1, 1, 1]) == [0, 0, 0, 0, 0, 1, 2, 3])
    }

    @Test
    func somePrefixSumSmallInclusive() throws {
//        #expect(try test(values: []).isEmpty)
        #expect(try test(values: [0], inclusive: true) == [0])
        #expect(try test(values: [1, 1, 1, 1], inclusive: true) == [1, 2, 3, 4])
        #expect(try test(values: [1, 2, 3, 4], inclusive: true) == [1, 3, 6, 10])
        #expect(try test(values: [1, 1, 1, 1, 0, 0, 0, 0], inclusive: true) == [1, 2, 3, 4, 4, 4, 4, 4])
        #expect(try test(values: [0, 0, 0, 0, 1, 1, 1, 1], inclusive: true) == [0, 0, 0, 0, 1, 2, 3, 4])
    }

    @Test
    func testOneThreadgroup() throws {
        let values = Array<UInt32>((0..<1024).map({ _ in .random(in: 0..<10) }))
        #expect(try test(values: values) == values.prefixSumExclusive())
    }

    @Test
    func testOneSimdGroupInclusive() throws {
        let values = Array<UInt32>((0..<32).map({ _ in .random(in: 0..<10) }))
        #expect(try test(values: values, inclusive: true) == values.prefixSumInclusive())
    }

    @Test
    func testTwoSimdGroupsInclusive() throws {
        let values = Array<UInt32>((0..<64).map({ _ in .random(in: 0..<10) }))
        #expect(try test(values: values, inclusive: true) == values.prefixSumInclusive())
    }

    @Test
    func testOneThreadgroupInclusive() throws {
        let values = Array<UInt32>((0..<1024).map({ _ in .random(in: 0..<10) }))
        #expect(try test(values: values, inclusive: true) == values.prefixSumInclusive())
    }

    @Test
    func testSubThreadgroup() throws {
        let values = Array<UInt32>((0..<256).map({ _ in .random(in: 0..<10) }))
        #expect(try test(values: values) == values.prefixSumExclusive())
    }
}

@Test
func testStride() {
    #expect(Array(stride(from: 100, dividingBy: 10)) == [100, 10, 1])
    #expect(Array(stride(from: 10, dividingBy: 10)) == [10, 1])
    #expect(Array(stride(from: 0, dividingBy: 10)) == [0]) // Infinite loop
}


@Test
func countingSortShuffles() throws {



    let device = MTLCreateSystemDefaultDevice()!
    let compute = try Compute(device: device, logger: Logger(), logging: true)

    let library = ShaderLibrary.bundle(.module, name: "debug")
    var shufflePipeline = try compute.makePipeline(function: library.function(name: "CountingSort::shuffle"))

    let input: [UInt32] = [3, 2, 1, 0]

    print(radixSort(values: input))

    #expect(result == expectedOutput)

}
