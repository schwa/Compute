@testable import Compute
@testable import Examples
import Metal
import Testing

struct TestPrefixSum {

    func test(values: [UInt32]) throws -> [UInt32] {
        let device = MTLCreateSystemDefaultDevice()!
        let input = try device.makeTypedBuffer(data: values).labelled("Input")
        let compute = try Compute(device: device)
        let demo = try YAPrefixSum(compute: compute)
        let output = try demo.prefixSum(input: input)
        return Array(output)
    }

    @Test
    func somePrefixSumSmall() throws {
//        #expect(try test(values: []).isEmpty)
        #expect(try test(values: [0]) == [0])
        #expect(try test(values: [1, 1, 1, 1]) == [0, 1, 2, 3])
        #expect(try test(values: [1, 2, 3, 4]) == [0, 1, 3, 6])
    }

    @Test
    func testOneThreadgroup() throws {
        let values = Array<UInt32>((0..<1024).map({ _ in .random(in: 0..<10) }))
        #expect(try test(values: values) == values.prefixSum())
    }

    @Test
    func testSubThreadgroup() throws {
        let values = Array<UInt32>((0..<256).map({ _ in .random(in: 0..<10) }))
        #expect(try test(values: values) == values.prefixSum())
    }
}

@Test
func testStride() {
    #expect(Array(stride(from: 100, dividingBy: 10)) == [100, 10, 1])
    #expect(Array(stride(from: 10, dividingBy: 10)) == [10, 1])
    #expect(Array(stride(from: 0, dividingBy: 10)) == [0]) // Infinite loop
}
