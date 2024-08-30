import Compute
import Metal
import os

enum IndexTest: Demo {
    static let source = #"""
        #include <metal_stdlib>
        #include <metal_logging>

        using namespace metal;

        uint threads_per_grid [[threads_per_grid]];
        uint thread_position_in_grid [[thread_position_in_grid]];

        [[kernel]]
        void index_test(
            device uint *indices1 [[buffer(0)]],
            device uint *indices2 [[buffer(1)]],
            constant uint &n [[buffer(2)]]
        ) {
            if (thread_position_in_grid == 0) {
                os_log_default.log("count: %d, threads_per_grid: %d", n, threads_per_grid);
            }
            uint k = thread_position_in_grid;
            uint s = threads_per_grid;
            uint index1 = (k * n / s - 1 + (k + 1) * n / s - 1) >> 1;
            uint index2 = (k + 1) * n / s - 1;
            indices1[thread_position_in_grid] = index1;
            indices2[thread_position_in_grid] = index2;
        }

    """#

    static func main() async throws {
        let device = MTLCreateSystemDefaultDevice()!
        let compute = try Compute(device: device, logger: Logger())
        let library = ShaderLibrary.source(source, enableLogging: true)
        var index_test = try compute.makePipeline(function: library.index_test)

        let s = 65536
        let indices1 = device.makeBuffer(length: 131072 * MemoryLayout<UInt32>.stride, options: [])!
        let indices2 = device.makeBuffer(length: 131072 * MemoryLayout<UInt32>.stride, options: [])!
        let n = 131072

        index_test.arguments.indices1 = .buffer(indices1)
        index_test.arguments.indices2 = .buffer(indices2)
        index_test.arguments.n = .int(n)
        try compute.run(pipeline: index_test, width: s)


        for k in stride(from: 0, through: s - 1, by: 1) {
            let index1 = (k * n / s - 1 + (k + 1) * n / s - 1) / 2
            let index2 = (k + 1) * n / s - 1
            let indices1 = Array<UInt32>(indices1)
            let indices2 = Array<UInt32>(indices2)
            if k == 32767 {
                print(indices1[k], index1)
            }
            assert(indices2[k] == index2)
            assert(indices1[k] == index1)


        }

    }
}
