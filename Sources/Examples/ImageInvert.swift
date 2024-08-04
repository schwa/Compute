import CoreImage
import Compute
import os
import Metal
import MetalKit
import UniformTypeIdentifiers

enum ImageInvert {
    // Metal shader source code as a string
    static let source = #"""
        #include <metal_stdlib>

        using namespace metal;

        uint2 gid [[thread_position_in_grid]];

        constant bool isLinear [[function_constant(0)]];

        float3 srgbToLinear(float3 srgb) {
            float3 linear;
            linear.r = (srgb.r <= 0.04045f) ? srgb.r / 12.92f : pow((srgb.r + 0.055f) / 1.055f, 2.4f);
            linear.g = (srgb.g <= 0.04045f) ? srgb.g / 12.92f : pow((srgb.g + 0.055f) / 1.055f, 2.4f);
            linear.b = (srgb.b <= 0.04045f) ? srgb.b / 12.92f : pow((srgb.b + 0.055f) / 1.055f, 2.4f);
            return linear;
        }

        float3 linearToSRGB(float3 linear) {
            float3 srgb;
            srgb.r = (linear.r <= 0.0031308f) ? linear.r * 12.92f : 1.055f * pow(linear.r, 1.0f/2.4f) - 0.055f;
            srgb.g = (linear.g <= 0.0031308f) ? linear.g * 12.92f : 1.055f * pow(linear.g, 1.0f/2.4f) - 0.055f;
            srgb.b = (linear.b <= 0.0031308f) ? linear.b * 12.92f : 1.055f * pow(linear.b, 1.0f/2.4f) - 0.055f;
            return srgb;
        }

        kernel void invertImage(
            texture2d<float, access::read> inputTexture [[texture(0)]],
            texture2d<float, access::write> outputTexture [[texture(1)]],
            uint2 gid [[thread_position_in_grid]]
        ) {
            float4 pixel = inputTexture.read(gid);
            if (!isLinear) {
                pixel.rgb = srgbToLinear(pixel.rgb);
            }
            pixel.rgb = 1.0 - pixel.rgb;
            if (!isLinear) {
                pixel.rgb = linearToSRGB(pixel.rgb);
            }
            outputTexture.write(pixel, gid);
        }
    """#

    static func main() throws {
        let device = MTLCreateSystemDefaultDevice()!
        let logger = Logger()
        let compute = try Compute(device: device, logger: logger)
        let library = ShaderLibrary.source(source)
        var invertImage = try compute.makePipeline(function: library.invertImage, constants: ["isLinear": .bool(false)])

        let textureLoader = MTKTextureLoader(device: device)
        let inputTexture = try textureLoader.newTexture(name: "baboon", scaleFactor: 1, bundle: .module, options: [.SRGB: true])

        let outputTextureDescriptor = MTLTextureDescriptor()
        outputTextureDescriptor.width = inputTexture.width
        outputTextureDescriptor.height = inputTexture.height
        outputTextureDescriptor.pixelFormat = .bgra8Unorm_srgb
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
