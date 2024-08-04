import AVFoundation
import Compute
import Foundation
import Metal
import os

enum GameOfLife {
    static func main(density: Double = 0.99, width: Int = 640, height: Int = 480, frames: Int = 600, framesPerSecond: Int = 60) async throws {
        let pixelCount = width * height
        let device = MTLCreateSystemDefaultDevice()!

        let bufferA = device.makeBuffer(length: pixelCount * MemoryLayout<UInt32>.size, options: [])!
        bufferA.contents().withMemoryRebound(to: UInt32.self, capacity: pixelCount) { buffer in
            for _ in 0..<Int(Double(pixelCount) * density) {
                buffer[Int.random(in: 0..<pixelCount)] = 0xFFFFFFFF
            }
        }
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: width, height: height, mipmapped: false)
        textureDescriptor.usage = [.shaderRead, .shaderWrite]
        textureDescriptor.resourceOptions = [.storageModeShared]

        let textureA = bufferA.makeTexture(descriptor: textureDescriptor, offset: 0, bytesPerRow: width * MemoryLayout<UInt32>.size)!
        textureA.label = "texture-a"

        let bufferB = device.makeBuffer(length: pixelCount * MemoryLayout<UInt32>.size, options: [])!
        let textureB = bufferB.makeTexture(descriptor: textureDescriptor, offset: 0, bytesPerRow: width * MemoryLayout<UInt32>.size)!
        textureB.label = "texture-b"

        let compute = try Compute(device: device)
        let library = ShaderLibrary.bundle(.module, name: "debug")

        let url = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Desktop/GameOfLife.mp4")
        let movieWriter = try TextureToVideoWriter(outputURL: url, size: CGSize(width: width, height: height))
        movieWriter.start()

        var totalComputeTime: UInt64 = 0
        var totalEncodeTime: UInt64 = 0

        var pass = try compute.makePass(function: library.gameOfLife_float4, constants: ["wrap": .bool(true)])
        for frame in 0..<frames {
            let inputTexture = frame.isMultiple(of: 2) ? textureA : textureB
            let outputTexture = frame.isMultiple(of: 2) ? textureB : textureA

            pass.arguments.inputTexture = .texture(inputTexture)
            pass.arguments.outputTexture = .texture(outputTexture)
            try timeit {
                try compute.task { task in
                    try task { dispatch in
                        try dispatch(pass: pass, threadgroupsPerGrid: MTLSize(width: width, height: height, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
                    }
                }
            }
            display: {
                totalComputeTime += $0
            }

            timeit {
                let time = CMTimeMakeWithSeconds(Double(frame) / Double(framesPerSecond), preferredTimescale: 600)
                movieWriter.writeFrame(texture: outputTexture, at: time)
            }
            display: {
                totalEncodeTime += $0
            }
        }
        try await movieWriter.finish()

        print("Frames", frames)
        print("Compute time", Double(totalComputeTime) / Double(1_000_000_000))
        print("Encode time", Double(totalEncodeTime) / Double(1_000_000_000))
    }
}
