#if os(macOS)
import AVFoundation
import Compute
import Foundation
import Metal
import os

enum GameOfLife {
    static func main(density: Double = 0.5, width: Int = 256, height: Int = 256, frames: Int = 1_200, framesPerSecond: Int = 60) async throws {
        let logger: Logger? = Logger()

        // Calculate total number of pixels
        let pixelCount = width * height

        // Get the default Metal device
        let device = MTLCreateSystemDefaultDevice()!

        // Create and initialize buffer A with random live cells
        logger?.log("Creating buffers")
        let bufferA = device.makeBuffer(length: pixelCount * MemoryLayout<UInt32>.size, options: [])!
        bufferA.contents().withMemoryRebound(to: UInt32.self, capacity: pixelCount) { buffer in
            for n in 0..<pixelCount where Double.random(in: 0...1) <= density {
                buffer[n] = 0xFFFFFFFF
            }
        }
        logger?.log("Creating textures")
        // Create texture descriptor for _both_ textures
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: width, height: height, mipmapped: false)
        textureDescriptor.usage = [.shaderRead, .shaderWrite]
        textureDescriptor.resourceOptions = [.storageModeShared]

        // Create texture A from buffer A
        let textureA = bufferA.makeTexture(descriptor: textureDescriptor, offset: 0, bytesPerRow: width * MemoryLayout<UInt32>.size)!
        textureA.label = "texture-a"

        // Create buffer B and texture B
        let bufferB = device.makeBuffer(length: pixelCount * MemoryLayout<UInt32>.size, options: [])!
        let textureB = bufferB.makeTexture(descriptor: textureDescriptor, offset: 0, bytesPerRow: width * MemoryLayout<UInt32>.size)!
        textureB.label = "texture-b"

        logger?.log("Loading shaders")
        // Initialize Compute and ShaderLibrary
        let compute = try Compute(device: device)
        let library = ShaderLibrary.bundle(.module, name: "debug")

        // Initialize timing variables
        var totalComputeTime: UInt64 = 0
        var totalEncodeTime: UInt64 = 0

        // Create compute pipeline
        var pipeline = try compute.makePipeline(function: library.gameOfLife_float4, constants: ["wrap": .bool(true)])

        // Set up video writer
        let url = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Desktop/GameOfLife.mov")
        let movieWriter = try TextureToVideoWriter(outputURL: url, size: CGSize(width: width, height: height))
        movieWriter.start()

        logger?.log("Encoding")

        let time = CMTimeMakeWithSeconds(0, preferredTimescale: 600)
        movieWriter.writeFrame(texture: textureA, at: time)

        // Main simulation loop
        for frame in 0..<frames {
            logger?.log("\(frame + 1)/\(frames)")
            // Alternate between textures A and B for input and output
            let inputTexture = frame.isMultiple(of: 2) ? textureA : textureB
            let outputTexture = frame.isMultiple(of: 2) ? textureB : textureA

            // Set up arguments for the compute pipeline
            pipeline.arguments.inputTexture = .texture(inputTexture)
            pipeline.arguments.outputTexture = .texture(outputTexture)

            // Run the compute pipeline and measure time
            try timeit {
                try compute.task { task in
                    try task { dispatch in
                        try dispatch(pipeline: pipeline, threadgroupsPerGrid: MTLSize(width: width, height: height, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
                    }
                }
            }
            display: { nanoseconds in
                totalComputeTime += nanoseconds
            }

            // Write frame to video and measure time
            timeit {
                let time = CMTimeMakeWithSeconds(Double(frame + 1) / Double(framesPerSecond), preferredTimescale: 600)
                movieWriter.writeFrame(texture: outputTexture, at: time)
            }
            display: { nanoseconds in
                totalEncodeTime += nanoseconds
            }
        }

        logger?.log("Encoding")
        // Finish writing the video
        try await movieWriter.finish()

        // Print performance statistics
        print("Size: \(width)x\(height)")
        print("Frames", frames)
        print("Compute time", Double(totalComputeTime) / Double(1_000_000_000))
        print("Encode time", Double(totalEncodeTime) / Double(1_000_000_000))
    }
}
#endif
