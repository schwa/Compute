import Compute
import Metal
import os

enum BufferFill: Demo {
    static let source = #"""
        #include <metal_stdlib>
        #include <metal_logging>

        using namespace metal;

        uint thread_position_in_grid [[thread_position_in_grid]];
        uint threads_per_grid [[threads_per_grid]];

        kernel void buffer_fill(
            device uint *data [[buffer(0)]]
        ) {
            //if (thread_position_in_grid == 0) {
            //    os_log_default.log("threads_per_grid: %d", threads_per_grid);
            //}
            data[thread_position_in_grid] = threads_per_grid;
        }
    """#

    static func main() async throws {
        let device = MTLCreateSystemDefaultDevice()!
        let count = 2 ** 24
        let data = device.makeBuffer(length: count * MemoryLayout<UInt32>.stride, options: .storageModeShared)!
        let compute = try Compute(device: device, logger: Logger())
        let library = ShaderLibrary.source(source, enableLogging: true)
        var bufferFill = try compute.makePipeline(function: library.buffer_fill)
        bufferFill.arguments.data = .buffer(data)
        var n = count
        var values = Array<UInt32>(repeating: 0, count: count)
        while n > 0 {
            try compute.run(pipeline: bufferFill, width: n)
            values[0..<n] = Array(repeating: UInt32(n), count: n)[0..<n]
            n >>= 1
        }
        assert(Array<UInt32>(data) == values)
    }
}
