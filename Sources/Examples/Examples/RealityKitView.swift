//
//  Untitled.swift
//  Compute
//
//  Created by Nisse Bergman on 2024-09-15.
//

import Compute
import Metal
import MetalKit
import RealityKit
import SwiftUI

struct RealityKitView: View {
    private static let source =
        #"""
        #include <metal_stdlib>
        using namespace metal;
        
        static constant float radiusSquared = 50 * 50;
        uint2 threadPosition [[thread_position_in_grid]];
        
        kernel void drawCircle(texture2d<float, access::write> outputTexture [[texture(0)]],
                              constant float2 &position [[buffer(0)]]) {
        
           // Get the size of the texture
            uint width = outputTexture.get_width();
            uint height = outputTexture.get_height();
        
        
            // Check if the current thread is within the texture bounds
            if (threadPosition.x >= width || threadPosition.y >= height) {
                return;
            }
        
            // draw red if pixel is close to tap position
            if (distance_squared( float2(threadPosition), position) <= radiusSquared ) {
                outputTexture.write(float4(1, 0, 0, 1), threadPosition);
            }
        }
        """#

    @State private var realityKitTexture: LowLevelTexture
    @State private var metalTexture: MTLTexture
    @State private var drawCircle: Compute.Pipeline
    @State private var compute: Compute

    init() {
        guard let metalDevice = MTLCreateSystemDefaultDevice() else {
            fatalError("not able to create metal device")
        }

        (self.realityKitTexture, self.metalTexture) = Self.createOutputTexture(metalDevice: metalDevice)

        let compute: Compute
        do {
            compute = try Compute(device: metalDevice)
        } catch {
            fatalError("failed creating compute with error: \(error)")
        }
        self.compute = compute

        let library = ShaderLibrary.source(Self.source)

        do {
            self.drawCircle = try compute.makePipeline(function: library.drawCircle)
        } catch {
            fatalError("failed creating pileline with error: \(error)")
        }
    }

    var body: some View {
        RealityView { content in

            let entity: Entity = await Self.createEntity(texture: realityKitTexture)
            content.add(entity)
        }
        .realityViewCameraControls(.orbit)
        .gesture(SpatialTapGesture().targetedToAnyEntity().onEnded { value in
            guard let hitPosition = self.convertHitPositionToTexturePosition(value: value) else {
                return
            }
            handleTap(at: hitPosition)
        })
    }

    private func handleTap(at hitPosition: SIMD2<Float>) {
        let threadsPerThreadgroupWidth = Int(sqrt(Double(drawCircle.maxTotalThreadsPerThreadgroup)))
        let threadsPerThreadgroupHeight = drawCircle.maxTotalThreadsPerThreadgroup / threadsPerThreadgroupWidth

        do {
            try compute.task { task in
                try task { dispatch in
                    drawCircle.arguments.outputTexture = .texture(metalTexture)
                    drawCircle.arguments.position = .vector(hitPosition)

                    try dispatch(pipeline: drawCircle,
                                 threadgroupsPerGrid: MTLSize(width: metalTexture.width, height: metalTexture.height, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: threadsPerThreadgroupWidth, height: threadsPerThreadgroupHeight, depth: 1))
                }
            }
        } catch {
            fatalError("failed dispatching compute with error: \(error)")
        }
    }

    private func convertHitPositionToTexturePosition(value: EntityTargetValue<SpatialTapGesture.Value>) -> SIMD2<Float>? {
        guard let ray = value.ray(through: value.location, in: .local, to: .scene) else {
            return nil
        }

        guard let scene = value.entity.scene else {
            return nil
        }

        guard let hit = scene.raycast(origin: ray.origin, direction: ray.direction).first else {
            return nil
        }

        let localEntityPosition = value.entity.convert(position: hit.position, from: nil)

        return (SIMD2<Float>(localEntityPosition.x, localEntityPosition.z) + [0.5, 0.5]) * [Float(metalTexture.width), Float(metalTexture.height)]
    }

    private static func createOutputTexture(metalDevice: MTLDevice) -> (LowLevelTexture, MTLTexture) {
        var textureDescriptor = LowLevelTexture.Descriptor()
        textureDescriptor.textureType = .type2D
        textureDescriptor.width = 1024
        textureDescriptor.height = 1024
        textureDescriptor.pixelFormat = .bgra8Unorm
        textureDescriptor.textureUsage = [.shaderRead, .shaderWrite]

        let lowLevelTexture: LowLevelTexture
        do {
            lowLevelTexture = try LowLevelTexture(descriptor: textureDescriptor)
        } catch {
            fatalError("failed creating low level texture with error: \(error.localizedDescription)")
        }

        guard let commandQueue = metalDevice.makeCommandQueue() else {
            fatalError("not possible to create command queue")
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            fatalError("not able to create command buffer")
        }

        let metalTexture = lowLevelTexture.replace(using: commandBuffer)

        return (lowLevelTexture, metalTexture)
    }

    private static func createEntity(texture: LowLevelTexture) async -> Entity {
        let textureResource: TextureResource
        do {
            textureResource = try await TextureResource(from: texture)
        } catch {
            fatalError("failed creating texture resource with error: \(error)")
        }
        let material = UnlitMaterial(texture: textureResource)

        let mesh = MeshResource.generatePlane(width: 1, depth: 1)
        let entity = ModelEntity(mesh: mesh, materials: [material])

        entity.transform.rotation = simd_quatf(angle: .pi / 2, axis: SIMD3<Float>(1, 0, 0))
        entity.components.set(InputTargetComponent(allowedInputTypes: .all))
        do {
            entity.collision = try await CollisionComponent(shapes: [.generateStaticMesh(from: mesh)])
        } catch {
            fatalError("failed adding collision component with error: \(error)")
        }

        return entity
    }
}
