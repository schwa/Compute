import Metal
import MetalKit
import MetalSupportLite
import Compute

/**
    * Perform a parallel prefix sum on the given data buffer
    *
    * Based on "Parallel Prefix Sum (Scan) with CUDA"
    * https://www.eecs.umich.edu/courses/eecs570/hw/parprefix.pdf
    */

enum PrefixSum: Demo {
    static func main() async throws {

        // TODO: Succeeds at up to 512 elements, fails at 513+ elements.
        let input = (0..<513).map(UInt32.init)
        let device = MTLCreateSystemDefaultDevice()!

        let inputBuffer = try device.makeBuffer(bytesOf: input, options: [])
        let compute = try Compute(device: device)
        let library = ShaderLibrary.bundle(.module, name: "debug")
        let kernel = try PrefixSumKernel(device: device, dataBuffer: inputBuffer, count: input.count,library: library.make(device))
        try compute.task { task in
            try task { dispatcher in
                try kernel.dispatch(computeEncoder: dispatcher.commandEncoder)
            }
        }
        let output = Array(inputBuffer.contentsBuffer(of: UInt32.self))
        let expected = input.prefixSumExclusive()
        print("Input count: \(input.count)")
        print("Output count: \(output.count)")
        print("Expected count: \(expected.count)")
        print("Expected prefix sum: \(expected.prefix(upTo: min(20, expected.count)))")
        print("Actual output: \(output.prefix(upTo: min(20, output.count)))")
        print("Count assertion: \(output.count == input.count)")
        print("Values match: \(output == expected)")
        if output != expected {
            let differences = zip(output, expected).enumerated().first { $0.element.0 != $0.element.1 }
            if let diff = differences {
                print("First difference at index: \(diff.offset)")
                print("Expected: \(diff.element.1), Got: \(diff.element.0)")
            }
        }

        if input.count < 60 {
            print("Input: \(input)")
            print("Output: \(output)")
        }
        else {
            print("Input: \(input.prefix(upTo: 20)) ... \(input.suffix(20))")
            print("Output: \(output.prefix(upTo: 20)) ...  \(output.suffix(20))")
        }
    }
}


struct PrefixSumUniforms {
    var threads_per_workgroup: UInt32
    var items_per_workgroup: UInt32
    var element_count: UInt32
}

struct PrefixSumPass {
    let pipeline: MTLComputePipelineState
    let bindGroup: MTLBuffer
    let blockSumBuffer: MTLBuffer
    let dispatchSize: MTLSize
    var uniforms: PrefixSumUniforms
}

class PrefixSumKernel {
    private let device: MTLDevice
    private let workgroupSize: MTLSize
    private let threadsPerWorkgroup: Int
    private let itemsPerWorkgroup: Int
    private let library: MTLLibrary

    private var passes: [PrefixSumPass] = []

    init(device: MTLDevice,
         dataBuffer: MTLBuffer,
         count: Int,
         workgroupSize: MTLSize = MTLSize(width: 16, height: 16, depth: 1),
         avoidBankConflicts: Bool = false,
         library: MTLLibrary
    ) throws {

        self.device = device
        self.workgroupSize = workgroupSize
        self.threadsPerWorkgroup = workgroupSize.width * workgroupSize.height
        self.itemsPerWorkgroup = 2 * threadsPerWorkgroup

        // Validate workgroup size is power of 2
        guard threadsPerWorkgroup.nonzeroBitCount == 1 else {
            throw PrefixSumError.invalidWorkgroupSize
        }

        self.library = library

        try createPassRecursive(dataBuffer: dataBuffer, count: count)
    }

    private func createPassRecursive(dataBuffer: MTLBuffer, count: Int) throws {
        let workgroupCount = (count + itemsPerWorkgroup - 1) / itemsPerWorkgroup
        let dispatchSize = findOptimalDispatchSize(workgroupCount: workgroupCount)

        // Create buffer for block sums
        guard let blockSumBuffer = device.makeBuffer(
            length: workgroupCount * 4,
            options: .storageModeShared
        ) else {
            throw PrefixSumError.bufferCreationError
        }

        let uniforms = PrefixSumUniforms(
            threads_per_workgroup: UInt32(threadsPerWorkgroup),
            items_per_workgroup: UInt32(itemsPerWorkgroup),
            element_count: UInt32(count)
        )

        // Create scan pipeline
        guard let scanFunction = library.makeFunction(name: "reduce_downsweep") else {
            throw PrefixSumError.functionError
        }

        let scanPipeline = try device.makeComputePipelineState(function: scanFunction)

        let scanPass = PrefixSumPass(
            pipeline: scanPipeline,
            bindGroup: dataBuffer,
            blockSumBuffer: blockSumBuffer,
            dispatchSize: dispatchSize,
            uniforms: uniforms
        )

        passes.append(scanPass)

        if workgroupCount > 1 {
            // Recursively create prefix sum for block sums
            try createPassRecursive(dataBuffer: blockSumBuffer, count: workgroupCount)

            // Create add block sums pipeline
            guard let addBlockSumsFunction = library.makeFunction(name: "add_block_sums") else {
                throw PrefixSumError.functionError
            }

            let addBlockSumsPipeline = try device.makeComputePipelineState(function: addBlockSumsFunction)

            let addBlockSumsPass = PrefixSumPass(
                pipeline: addBlockSumsPipeline,
                bindGroup: dataBuffer,
                blockSumBuffer: blockSumBuffer,
                dispatchSize: dispatchSize,
                uniforms: uniforms
            )

            passes.append(addBlockSumsPass)
        }
    }

    private func findOptimalDispatchSize(workgroupCount: Int) -> MTLSize {
        // Find best dispatch dimensions to minimize unused threads
        let maxThreadsPerThreadgroup = device.maxThreadsPerThreadgroup
        let width = min(workgroupCount, maxThreadsPerThreadgroup.width)
        let height = min((workgroupCount + width - 1) / width, maxThreadsPerThreadgroup.height)

        return MTLSize(width: width, height: height, depth: 1)
    }

    func getDispatchChain() -> [UInt32] {
        return passes.flatMap { pass in
            [UInt32(pass.dispatchSize.width), UInt32(pass.dispatchSize.height), 1]
        }
    }

    func dispatch(computeEncoder: MTLComputeCommandEncoder,
                  dispatchSizeBuffer: MTLBuffer? = nil,
                  offset: Int = 0) throws {

        for (index, pass) in passes.enumerated() {
            var pass = pass
            computeEncoder.setComputePipelineState(pass.pipeline)
            computeEncoder.setBuffer(pass.bindGroup, offset: 0, index: 0)
            computeEncoder.setBuffer(pass.blockSumBuffer, offset: 0, index: 1)
            computeEncoder.setBytes(&pass.uniforms, length: MemoryLayout<PrefixSumUniforms>.size, index: 2)

            let threadsPerThreadgroup = MTLSize(
                width: workgroupSize.width,
                height: workgroupSize.height,
                depth: 1
            )

            if let dispatchBuffer = dispatchSizeBuffer {
                // Indirect dispatch
                let bufferOffset = offset + index * 3 * 4
                computeEncoder.dispatchThreadgroups(
                    indirectBuffer: dispatchBuffer,
                    indirectBufferOffset: bufferOffset,
                    threadsPerThreadgroup: threadsPerThreadgroup
                )
            } else {
                // Direct dispatch
                let threadgroupsPerGrid = MTLSize(
                    width: (pass.dispatchSize.width + workgroupSize.width - 1) / workgroupSize.width,
                    height: (pass.dispatchSize.height + workgroupSize.height - 1) / workgroupSize.height,
                    depth: 1
                )

                computeEncoder.dispatchThreadgroups(
                    threadgroupsPerGrid,
                    threadsPerThreadgroup: threadsPerThreadgroup
                )
            }
        }
    }
}

enum PrefixSumError: Error {
    case invalidWorkgroupSize
    case libraryError
    case functionError
    case bufferCreationError
}
