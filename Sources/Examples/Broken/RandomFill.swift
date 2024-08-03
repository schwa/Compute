// #if os(macOS)
// import AppKit
// import BaseSupport
// import Compute
// import CoreGraphics
// import Foundation
// import Metal
//
//// swiftlint:disable force_unwrapping
//
// struct RandomFill {
//    let width = 512
//    let height = 512
//    let device = MTLCreateSystemDefaultDevice()!
//
//    func main() throws {
//        let testImage = try CGImage.makeTestImage(width: 512, height: 512)
//        let testTexture = try device.newTexture(with: testImage)
//        let outImage = try testTexture.cgImage()
//        try outImage.write(to: URL(filePath: "/tmp/test-image.png"))
//        URL(filePath: "/tmp/test-image.png").reveal()
//
//        try testPixelFormats()
//
//        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r8Uint, width: width, height: height, mipmapped: false)
//        textureDescriptor.usage = [.shaderWrite]
//        textureDescriptor.resourceOptions = .storageModeShared
//        guard let bytesPerRow = textureDescriptor.bytesPerRow else {
//            throw BaseError.resourceCreationFailure
//        }
//        let bufferSize = bytesPerRow * height
//        let buffer = try device.makeBufferEx(length: bufferSize, options: [.storageModeShared])
//        // let alignment = device.minimumLinearTextureAlignment(for: textureDescriptor.pixelFormat)
//        let texture = buffer.makeTexture(descriptor: textureDescriptor, offset: 0, bytesPerRow: bytesPerRow)!
//
//        let compute = try Compute(device: device)
//
//        let library = ShaderLibrary.bundle(.module)
//
//        var randomFillPass = try compute.makePass(function: library.randomFill_uint)
//        randomFillPass.arguments.outputTexture = .texture(texture)
//
//        try compute.task { task in
//            try task { dispatch in
//                try dispatch(pass: randomFillPass, threadgroupsPerGrid: MTLSize(width: width, height: height, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
//            }
//        }
//
//        let colorSpace = CGColorSpace(name: CGColorSpace.linearGray)
//        assert(texture.depth == 1)
//
//        let data = UnsafeMutableRawBufferPointer(start: buffer.contents(), count: buffer.length)
//
//        let context = try CGContext.bitmapContext(data: data, definition: .init(width: texture.width, height: height, pixelFormat: PixelFormat(bitsPerComponent: texture.pixelFormat.bits!, numberOfComponents: 1, alphaInfo: .none, byteOrder: .orderDefault, colorSpace: colorSpace)))
//
//        let image = context.makeImage()!
//
//        let url = URL(filePath: "/tmp/image.png")
//        let destination = try ImageDestination(url: url)
//        destination.addImage(image)
//        try destination.finalize()
//        url.reveal()
//    }
// }
//
// func testPixelFormats() throws {
//    let device = MTLCreateSystemDefaultDevice()!
//    for pixelFormat in MTLPixelFormat.allCases {
//        guard let pixelFormat2 = PixelFormat(pixelFormat) else {
//            continue
//        }
//        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: pixelFormat, width: 10, height: 10, mipmapped: false)
//        textureDescriptor.usage = [.shaderWrite]
//        textureDescriptor.resourceOptions = .storageModeShared
//        let alignment = device.minimumLinearTextureAlignment(for: textureDescriptor.pixelFormat)
//        guard let unalignedBytesPerRow = textureDescriptor.bytesPerRow else {
//            continue
//        }
//        let bytesPerRow = align(unalignedBytesPerRow, alignment: alignment)
//        let bufferSize = bytesPerRow * textureDescriptor.height
//        let buffer = try device.makeBufferEx(length: bufferSize, options: [.storageModeShared])
//        let texture = buffer.makeTexture(descriptor: textureDescriptor, offset: 0, bytesPerRow: bytesPerRow)!
//
//        let data = UnsafeMutableRawBufferPointer(start: buffer.contents(), count: buffer.length)
//        _ = try CGContext.bitmapContext(data: data, definition: .init(width: texture.width, height: texture.height, bytesPerRow: bytesPerRow, pixelFormat: pixelFormat2))
//    }
// }
// #endif