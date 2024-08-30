import Compute
import Metal
import os

// swiftlint:disable force_unwrapping

public struct BitonicSortDemo: Demo {

    public static func main() async throws {
        let logger: Logger? = Logger()

        logger?.debug("Creating random buffer")
        var entries: [UInt32] = (0 ..< 10_000).shuffled()

        logger?.debug("Copying buffer to GPU.")
        let device = MTLCreateSystemDefaultDevice()!
        let numEntries = entries.count
        let buffer: MTLBuffer = try entries.withUnsafeMutableBufferPointer { buffer in
            let buffer = UnsafeMutableRawBufferPointer(buffer)
            return try device.makeBufferEx(bytes: buffer.baseAddress!, length: buffer.count)
        }
        logger?.debug("Preparing compute.")

        let function = ShaderLibrary.bundle(.module).bitonicSort
        let numStages = log2(nextPowerOfTwo(numEntries))

        let compute = try Compute(device: device)

        var pipeline = try compute.makePipeline(function: function, arguments: [
            "numEntries": .int(numEntries),
            "entries": .buffer(buffer),
        ])

        let start = CFAbsoluteTimeGetCurrent()
        logger?.debug("Running \(numStages) compute stages")

        var threadgroupsPerGrid = (entries.count + pipeline.maxTotalThreadsPerThreadgroup - 1) / pipeline.maxTotalThreadsPerThreadgroup
        threadgroupsPerGrid = (threadgroupsPerGrid + pipeline.threadExecutionWidth - 1) / pipeline.threadExecutionWidth * pipeline.threadExecutionWidth

        try compute.task { task in
            try task { dispatch in
                var n = 0
                for stageIndex in 0 ..< numStages {
                    for stepIndex in 0 ..< (stageIndex + 1) {
                        let groupWidth = 1 << (stageIndex - stepIndex)
                        let groupHeight = 2 * groupWidth - 1

                        pipeline.arguments.groupWidth = .int(groupWidth)
                        pipeline.arguments.groupHeight = .int(groupHeight)
                        pipeline.arguments.stepIndex = .int(stepIndex)

                        logger?.debug("\(n), \(stageIndex)/\(numStages), \(stepIndex)/\(stageIndex + 1), \(groupWidth), \(groupHeight)")
                        try dispatch(
                            pipeline: pipeline,
                            threadgroupsPerGrid: MTLSize(width: threadgroupsPerGrid),
                            threadsPerThreadgroup: MTLSize(width: pipeline.maxTotalThreadsPerThreadgroup)
                        )
                        n += 1
                    }
                }
            }
        }

        let current = CFAbsoluteTimeGetCurrent()
        logger?.debug("GPU \(current - start), \(1 / (current - start))")

        logger?.debug("Running on CPU for comparison")
        entries.sort()

        logger?.debug("Confirming output is sorted,")

        let result = Array<UInt32>(buffer)
        assert(entries == result)
    }
}
