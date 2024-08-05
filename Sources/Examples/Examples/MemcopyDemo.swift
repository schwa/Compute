import Compute
import Metal

enum MemcopyDemo {
    // Metal shader source code as a string
    static let source = #"""
        #include <metal_stdlib>

        using namespace metal;

        // Thread position in the execution grid
        uint thread_position_in_grid [[thread_position_in_grid]];

        // Empty kernel for baseline performance measurement
        kernel void empty()
        {
        }

        // Kernel to fill buffer with thread positions
        kernel void fill(
            device uint* output [[buffer(0)]]  // Output buffer
        )
        {
            output[thread_position_in_grid] = thread_position_in_grid;
        }

        // Kernel to copy data from input buffer to output buffer
        kernel void memcpy(
            const device uint* input [[buffer(0)]],  // Input buffer
            device uint* output [[buffer(1)]]        // Output buffer
        )
        {
            output[thread_position_in_grid] = input[thread_position_in_grid];
        }
    """#

    static func main() throws {
        // Get the default Metal device
        let device = MTLCreateSystemDefaultDevice()!
        // Set count to maximum value of UInt32
        let count = Int(UInt32.max)
        // Calculate the length of the buffers in bytes
        let length = MemoryLayout<UInt32>.stride * count

        // Print the size of the buffers in gigabytes
        print(Measurement<UnitInformationStorage>(value: Double(length), unit: .bytes).converted(to: .gibibytes))

        print("# Allocating")
        // Create two Metal buffers of the calculated length
        let bufferA = device.makeBuffer(length: length)!
        let bufferB = device.makeBuffer(length: length)!

        print("# Preparing")
        // Create a Compute object with the Metal device
        let compute = try Compute(device: device)
        // Create a shader library from the source code
        let library = ShaderLibrary.source(source)
        // Create compute pipelines for each kernel function
        let empty = try compute.makePipeline(function: library.empty)
        let fill = try compute.makePipeline(function: library.fill, arguments: ["output": .buffer(bufferA)])
        let memcopy = try compute.makePipeline(function: library.memcpy, arguments: ["input": .buffer(bufferA), "output": .buffer(bufferB)])

        print("# Empty")
        // Run and time the empty kernel (baseline)
        try timeit(length: length) {
            try compute.run(pipeline: empty, width: count)
        }

        print("# Filling")
        // Run and time the fill kernel
        try timeit(length: length) {
            try compute.run(pipeline: fill, width: count)
        }

        print("# GPU memcpy")
        // Run and time the GPU memcpy kernel
        try timeit(length: length) {
            try compute.run(pipeline: memcopy, width: count)
        }

        print("# CPU memcpy")
        // Run and time CPU memcpy for comparison
        timeit(length: length) {
            memcpy(bufferB.contents(), bufferA.contents(), length)
        }

        print("# DONE")
    }
}
