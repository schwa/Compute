import Compute
import CoreGraphics
import Foundation
import Metal
import MetalKit

// swiftlint:disable force_unwrapping

struct RandomFill {
    static let source = #"""
    #include <metal_stdlib>
    #include <simd/simd.h>

    using namespace metal;

    float random(float2 p)
    {
      // We need irrationals for pseudo randomness.
      // Most (all?) known transcendental numbers will (generally) work.
      const float2 r = float2(
        23.1406926327792690,  // e^pi (Gelfond's constant)
         2.6651441426902251); // 2^sqrt(2) (Gelfond–Schneider constant)
      return fract(cos(fmod(123456789.0, 1e-7 + 256.0 * dot(p,r))));
    }

    uint2 thread_position_in_grid [[thread_position_in_grid]];

    [[kernel]]
    void randomFill_float(texture2d<float, access::write> outputTexture [[texture(0)]])
    {
        const float2 id = float2(thread_position_in_grid);

        float value = random(id) > 0.5 ? 1 : 0;

        float4 color = { value, value, value, 1 };

        outputTexture.write(color, thread_position_in_grid);
    }
    """#

    static func main() throws {
        let device = MTLCreateSystemDefaultDevice()!

        let outputTextureDescriptor = MTLTextureDescriptor()
        outputTextureDescriptor.width = 1_024
        outputTextureDescriptor.height = 1_024
        outputTextureDescriptor.pixelFormat = .bgra8Unorm
        outputTextureDescriptor.usage = .shaderWrite
        guard let outputTexture = device.makeTexture(descriptor: outputTextureDescriptor) else {
            throw ComputeError.resourceCreationFailure
        }

        let compute = try Compute(device: device)

        let library = ShaderLibrary.source(source)

        var pipeline = try compute.makePipeline(function: library.randomFill_float)
        pipeline.arguments.outputTexture = .texture(outputTexture)

        try compute.task { task in
            try task { dispatch in
                try dispatch(pipeline: pipeline, threadgroupsPerGrid: MTLSize(width: outputTexture.width, height: outputTexture.height, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            }
        }

        let url = URL(filePath: "/tmp/randomfill.png")
        try outputTexture.export(to: url)
        #if os(macOS)
        NSWorkspace.shared.selectFile(url.path, inFileViewerRootedAtPath: url.deletingLastPathComponent().path)
        #endif



    }
}
