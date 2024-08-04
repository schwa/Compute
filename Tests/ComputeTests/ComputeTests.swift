@testable import Compute
import Metal
import Testing

struct ComputeTests {
    let device: MTLDevice
    let compute: Compute

    init() throws {
        let device = MTLCreateSystemDefaultDevice()!
        self.device = device
        self.compute = try Compute(device: device)
    }

    @Test
    func computeInitialization() throws {
        #expect(compute != nil)
        #expect(compute.device === device)
    }

    @Test
    func shaderLibraryCreation() throws {
        let source = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void add(device int* a [[buffer(0)]], device int* b [[buffer(1)]], device int* result [[buffer(2)]], uint id [[thread_position_in_grid]]) {
            result[id] = a[id] + b[id];
        }
        """
        let library = ShaderLibrary.source(source)
        let function = library.add
        #expect(function.name == "add")
    }

    @Test
    func pipelineCreation() throws {
        let source = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void add(device int* a [[buffer(0)]], device int* b [[buffer(1)]], device int* result [[buffer(2)]], uint id [[thread_position_in_grid]]) {
            result[id] = a[id] + b[id];
        }
        """
        let library = ShaderLibrary.source(source)
        let function = library.add

        let pipeline = try compute.makePipeline(function: function)

        #expect(pipeline != nil)
    }

    @Test
    func simpleAddition() throws {
        let source = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void add(device int* a [[buffer(0)]], device int* b [[buffer(1)]], device int* result [[buffer(2)]], uint id [[thread_position_in_grid]]) {
            result[id] = a[id] + b[id];
        }
        """
        let library = ShaderLibrary.source(source)
        let function = library.add

        let count = 1_000
        let a = [Int32](repeating: 1, count: count)
        let b = [Int32](repeating: 2, count: count)

        let bufferA = device.makeBuffer(bytes: a, length: MemoryLayout<Int32>.stride * count, options: [])!
        let bufferB = device.makeBuffer(bytes: b, length: MemoryLayout<Int32>.stride * count, options: [])!
        let bufferResult = device.makeBuffer(length: MemoryLayout<Int32>.stride * count, options: [])!

        let pipeline = try compute.makePipeline(
            function: function,
            arguments: [
                "a": .buffer(bufferA),
                "b": .buffer(bufferB),
                "result": .buffer(bufferResult)
            ]
        )

        try compute.run(pipeline: pipeline, count: count)

        bufferResult.contents().withMemoryRebound(to: Int32.self, capacity: count) { result in
            for i in 0..<count {
                #expect(result[i] == 3, "Expected 3, but got \(result[i]) at index \(i)")
            }
        }
    }
}
