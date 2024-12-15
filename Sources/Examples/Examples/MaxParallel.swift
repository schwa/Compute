import Compute
import Metal
import os

enum MaxParallel: Demo {
    static let source = #"""
        #include <metal_stdlib>
        #include <metal_logging>

        using namespace metal;

        uint thread_position_in_grid [[thread_position_in_grid]];

        kernel void kernel_main(
            device atomic_uint &count [[buffer(0)]],
            device atomic_uint &maximum [[buffer(1)]]
        ) {
            uint current = atomic_fetch_add_explicit(&count, 1, memory_order_relaxed);
            atomic_fetch_max_explicit(&maximum, current, memory_order_relaxed);
            atomic_fetch_add_explicit(&count, -1, memory_order_relaxed);
        }
    """#

    static func main() async throws {
        let device = MTLCreateSystemDefaultDevice()!
        let compute = try Compute(device: device)
        let library = ShaderLibrary.source(source, enableLogging: true)
        let count = device.makeBuffer(length: MemoryLayout<UInt32>.size * 1, options: [])!
        let maximum = device.makeBuffer(length: MemoryLayout<UInt32>.size * 1, options: [])!
        var pipeline = try compute.makePipeline(function: library.kernel_main)
        pipeline.arguments.count = .buffer(count)
        pipeline.arguments.maximum = .buffer(maximum)

        for n in 0...31 {
            let count = Int(pow(2, Double(n+1))) - 1
            let time = try timed() {
                try compute.run(pipeline: pipeline, width: count)
            }
            print(count, Array<UInt32>(maximum)[0], Double(time) / Double(1_000_000))
        }
    }
}
