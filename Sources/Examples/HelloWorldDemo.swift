import Compute
import os
import Metal

enum HelloWorldDemo {
    // Metal shader source code as a string
    static let source = #"""
        #include <metal_stdlib>
        #include <metal_logging>

        using namespace metal;

        kernel void hello_world() {
            os_log_default.log("Hello world (from Metal!)");
        }
    """#

    static func main() throws {
        let device = MTLCreateSystemDefaultDevice()!
        // Create a Compute object with the Metal device
        let logger = Logger()
        logger.log("Hello world (from Swift!)")
        let compute = try Compute(device: device, logger: logger)
        let library: ShaderLibrary
        if #available(macOS 15, *) {
            library = ShaderLibrary.source(source, enableLogging: true)
        }
        else {
            library = ShaderLibrary.source(source)
        }
        let helloWorld = try compute.makePipeline(function: library.hello_world)
        try compute.run(pipeline: helloWorld, count: 1)
    }
}
