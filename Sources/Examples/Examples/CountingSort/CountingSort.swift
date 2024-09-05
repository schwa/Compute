import Compute
import Metal
import os

enum CountingSortDemo: Demo {
    static func main() async throws {
        let device = MTLCreateSystemDefaultDevice()!
        let logger = Logger()
        let compute = try Compute(device: device, logger: Logger(), logging: YAPrefixSum.logging)
        let library = ShaderLibrary.bundle(.module, name: "debug")
        let helloWorld = try compute.makePipeline(function: library.hello_world)
        try compute.run(pipeline: helloWorld, width: 1)
    }
}
