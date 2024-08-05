import Compute
import CoreImage
import Metal
import MetalKit
import os
import SwiftUI
import UniformTypeIdentifiers

enum Checkerboard {
    static let source = #"""
        #include <metal_stdlib>

        using namespace metal;

        uint2 gid [[thread_position_in_grid]];

        kernel void checkerboard(
            texture2d<float, access::write> outputTexture [[texture(0)]],
            constant float4 &color1 [[buffer(0)]],
            constant float4 &color2 [[buffer(1)]],
            constant uint2 &cellSize [[buffer(2)]]
        ) {
            // Get the size of the texture
            uint width = outputTexture.get_width();
            uint height = outputTexture.get_height();

            // Check if the current thread is within the texture bounds
            if (gid.x >= width || gid.y >= height) {
                return;
            }

            // Determine which square this pixel belongs to
            uint squareX = gid.x / cellSize.x;
            uint squareY = gid.y / cellSize.y;

            // Choose color based on whether the sum of squareX and squareY is even or odd
            float4 color = ((squareX + squareY) % 2 == 0) ? color1 : color2;

            // Write the color to the output texture
            outputTexture.write(color, gid);
        }
    """#

    static func main() throws {
        let device = MTLCreateSystemDefaultDevice()!
        let logger = Logger()
        let compute = try Compute(device: device, logger: logger)
        let library = ShaderLibrary.source(source)

        var checkerboard = try compute.makePipeline(function: library.checkerboard)

        let outputTextureDescriptor = MTLTextureDescriptor()
        outputTextureDescriptor.width = 1_024
        outputTextureDescriptor.height = 1_024
        outputTextureDescriptor.pixelFormat = .bgra8Unorm
        outputTextureDescriptor.usage = .shaderWrite
        guard let outputTexture = device.makeTexture(descriptor: outputTextureDescriptor) else {
            throw ComputeError.resourceCreationFailure
        }

        checkerboard.arguments.outputTexture = .texture(outputTexture)
        checkerboard.arguments.color1 = try .color(.gray)
        checkerboard.arguments.color2 = try .color(.mint)
        checkerboard.arguments.cellSize = .vector(SIMD2<UInt32>(64, 64))
        try compute.run(pipeline: checkerboard, width: outputTexture.width, height: outputTexture.height)

        let url = URL(filePath: "/tmp/checkerboard.png")
        try outputTexture.export(to: url)
        #if os(macOS)
        NSWorkspace.shared.selectFile(url.path, inFileViewerRootedAtPath: url.deletingLastPathComponent().path)
        #endif

        // ksdiff /tmp/inverted.png ~/Projects/Compute/Sources/Examples/Resources/Media.xcassets/baboon.imageset/baboon.png
    }
}
