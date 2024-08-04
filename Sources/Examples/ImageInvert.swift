import Compute
import CoreImage
import Metal
import MetalKit
import os
import UniformTypeIdentifiers

enum ImageInvert {
    // Metal shader source code as a string
    static let source = #"""
        #include <metal_stdlib>

        using namespace metal;

        uint2 gid [[thread_position_in_grid]];

        kernel void invertImage(
        texture2d<float, access::read> inputTexture [[texture(0)]],
        texture2d<float, access::write> outputTexture [[texture(1)]]
        ) {
        float4 pixel = inputTexture.read(gid);
        pixel.rgb = 1.0 - pixel.rgb;
        outputTexture.write(pixel, gid);
        }
    """#

    static func main() throws {
        let device = MTLCreateSystemDefaultDevice()!
        let logger = Logger()
        let compute = try Compute(device: device, logger: logger)
        let library = ShaderLibrary.source(source)
        var invertImage = try compute.makePipeline(function: library.invertImage, constants: ["isLinear": .bool(true)])

        let textureLoader = MTKTextureLoader(device: device)
        let inputTexture = try textureLoader.newTexture(name: "baboon", scaleFactor: 1, bundle: .module, options: [.SRGB: false])

        let outputTextureDescriptor = MTLTextureDescriptor()
        outputTextureDescriptor.width = inputTexture.width
        outputTextureDescriptor.height = inputTexture.height
        outputTextureDescriptor.pixelFormat = .bgra8Unorm
        outputTextureDescriptor.usage = .shaderWrite
        guard let outputTexture = device.makeTexture(descriptor: outputTextureDescriptor) else {
            throw ComputeError.resourceCreationFailure
        }

        invertImage.arguments.inputTexture = .texture(inputTexture)
        invertImage.arguments.outputTexture = .texture(outputTexture)
        try compute.run(pipeline: invertImage, width: inputTexture.width, height: inputTexture.height)

        try outputTexture.export(to: URL(filePath: "/tmp/inverted.png"))

        // ksdiff /tmp/inverted.png ~/Projects/Compute/Sources/Examples/Resources/Media.xcassets/baboon.imageset/baboon.png
    }
}
