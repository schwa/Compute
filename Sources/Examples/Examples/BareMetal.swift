import Compute
import Metal

enum BareMetalVsCompute: Demo {
    static let source = #"""
        #include <metal_stdlib>

        using namespace metal;

        uint thread_position_in_grid [[thread_position_in_grid]];

        kernel void memset(device uchar* output [[buffer(0)]], constant uchar &value [[buffer(1)]]) {
        output[thread_position_in_grid] = value;
        }
    """#

    static func main() async throws {
        // Get the default Metal device
        let device = MTLCreateSystemDefaultDevice()!
        // Create a buffer and confirm it is zeroed
        let buffer = device.makeBuffer(length: 16_384)!
        assert(UnsafeRawBufferPointer(start: buffer.contents(), count: buffer.length).allSatisfy { $0 == 0x00 })
        // Run compute and confirm the output is correct
        try compute(device: device, buffer: buffer, value: 0x88)
        assert(UnsafeRawBufferPointer(start: buffer.contents(), count: buffer.length).allSatisfy { $0 == 0x88 })
        // Run bareMetal and confirm the output is correct
        try bareMetal(device: device, buffer: buffer, value: 0xFF)
        assert(UnsafeRawBufferPointer(start: buffer.contents(), count: buffer.length).allSatisfy { $0 == 0xFF })
    }

    static func bareMetal(device: MTLDevice, buffer: MTLBuffer, value: UInt8) throws {
        // Create shader library from source
        let library = try device.makeLibrary(source: source, options: .init())
        let function = library.makeFunction(name: "memset")

        // Create compute pipeline for memset operation
        let computePipelineDescriptor = MTLComputePipelineDescriptor()
        computePipelineDescriptor.computeFunction = function
        let (computePipelineState, reflection) = try device.makeComputePipelineState(descriptor: computePipelineDescriptor, options: [.bindingInfo])
        guard let reflection else {
            throw ComputeError.resourceCreationFailure
        }
        guard let outputIndex = reflection.bindings.first(where: { $0.name == "output" })?.index else {
            throw ComputeError.missingBinding("output")
        }
        guard let valueIndex = reflection.bindings.first(where: { $0.name == "value" })?.index else {
            throw ComputeError.missingBinding("value")
        }

        // Execute compute pipeline
        guard let commandQueue = device.makeCommandQueue() else {
            throw ComputeError.resourceCreationFailure
        }
        commandQueue.label = "memcpy-MTLCommandQueue"

        let commandBufferDescriptor = MTLCommandBufferDescriptor()
        guard let commandBuffer = commandQueue.makeCommandBuffer(descriptor: commandBufferDescriptor) else {
            throw ComputeError.resourceCreationFailure
        }
        commandBuffer.label = "memcpy-MTLCommandBuffer"
        guard let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw ComputeError.resourceCreationFailure
        }
        computeCommandEncoder.label = "memcpy-MTLComputeCommandEncoder"
        computeCommandEncoder.setComputePipelineState(computePipelineState)
        computeCommandEncoder.setBuffer(buffer, offset: outputIndex, index: outputIndex)
        var value = value
        computeCommandEncoder.setBytes(&value, length: MemoryLayout.size(ofValue: value), index: valueIndex)
        let threadsPerGrid = MTLSize(width: buffer.length, height: 1, depth: 1)
        let maxTotalThreadsPerThreadgroup = computePipelineState.maxTotalThreadsPerThreadgroup
        let threadsPerThreadgroup = MTLSize(width: maxTotalThreadsPerThreadgroup, height: 1, depth: 1)
        computeCommandEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        computeCommandEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }

    static func compute(device: MTLDevice, buffer: MTLBuffer, value: UInt8) throws {
        // Set up.
        let compute = try Compute(device: device)

        // Create shader library from source
        let library = ShaderLibrary.source(source)

        // Create compute pipeline for memset operation
        var fill = try compute.makePipeline(function: library.memset)

        // Set buffer and fill value arguments
        fill.arguments.output = .buffer(buffer)
        fill.arguments.value = .int(value)

        // Execute compute pipeline
        try compute.run(pipeline: fill, width: buffer.length)
    }
}
