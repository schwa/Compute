import Compute
import Metal
import os

// swiftlint:disable force_unwrapping

public struct BitonicSortDemo: Demo {

    static let capture = true

    public static func main() async throws {

        var entries: [UInt32] = timeit("Creating entries") {
            (0 ..< 1_500_000).shuffled()
        }

        let device = MTLCreateSystemDefaultDevice()!
        let numEntries = entries.count
        let buffer: MTLBuffer = try entries.withUnsafeMutableBufferPointer { buffer in
            let buffer = UnsafeMutableRawBufferPointer(buffer)
            return device.makeBuffer(bytes: buffer.baseAddress!, length: buffer.count)!
        }

        let function = ShaderLibrary.bundle(.module).bitonicSort
        let numStages = log2(nextPowerOfTwo(numEntries))

        let compute = try Compute(device: device)

        var pipeline = try compute.makePipeline(function: function, arguments: [
            "numEntries": .int(numEntries),
            "entries": .buffer(buffer),
        ])

        print("Running \(numStages) compute stages")

        var threadgroupsPerGrid = (entries.count + pipeline.maxTotalThreadsPerThreadgroup - 1) / pipeline.maxTotalThreadsPerThreadgroup
        threadgroupsPerGrid = (threadgroupsPerGrid + pipeline.threadExecutionWidth - 1) / pipeline.threadExecutionWidth * pipeline.threadExecutionWidth

        try device.capture(enabled: capture) {
            try timeit("GPU") {
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
                                //                            print("\(n), \(stageIndex)/\(numStages), \(stepIndex)/\(stageIndex + 1), \(groupWidth), \(groupHeight)")
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
            }
        }

        timeit("CPU") {
            entries.sort()
        }

        let result = Array<UInt32>(buffer)
        assert(entries == result)
    }
}
