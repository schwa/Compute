import Compute
import Metal
import os

enum ThreadgroupLogging: Demo {
    static let source = #"""
        #include <metal_stdlib>
        #include <metal_logging>

        const uint thread_position_in_grid [[thread_position_in_grid]];
        const uint threadgroup_position_in_grid [[threadgroup_position_in_grid]];
        const uint thread_position_in_threadgroup [[thread_position_in_threadgroup]];

        const uint threads_per_grid [[threads_per_grid]];
        const uint threads_per_threadgroup [[threads_per_threadgroup]];
        const uint threadgroups_per_grid [[threadgroups_per_grid]];

        using namespace metal;

        kernel void threadgroup_test() {
            if (thread_position_in_grid == 0) {
                os_log_default.log("threads_per_grid: %d, threads_per_threadgroup: %d, threadgroups_per_grid: %d", threads_per_grid, threads_per_threadgroup, threadgroups_per_grid);
            }
            os_log_default.log("thread_position_in_grid: %d, thread_position_in_threadgroup: %d, threadgroup_position_in_grid: %d", thread_position_in_grid, thread_position_in_threadgroup, threadgroup_position_in_grid);
        }
    """#

    static func main() async throws {
        let device = MTLCreateSystemDefaultDevice()!
        let compute = try Compute(device: device, logger: Logger())
        let library = ShaderLibrary.source(source, enableLogging: true)
        let pipeline = try compute.makePipeline(function: library.threadgroup_test)
        try compute.run(pipeline: pipeline, threadgroupsPerGrid: MTLSize(width: 3), threadsPerThreadgroup: MTLSize(width: 2))
    }
}
