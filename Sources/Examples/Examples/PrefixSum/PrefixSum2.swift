import Compute
import Metal
import MetalSupport
import os

struct PrefixSum2: Demo {

    static func main() async throws {
        let values: [UInt32] = [3, 1, 7, 0, 4, 1, 6, 3]
        print(values.prefixSum())

//        let values: [UInt32] = (0..<1_000_000).map { _ in UInt32.random(in: 0..<1000) }
//        assert(values.prefixSum() == [0, 3, 4, 11, 11, 15, 16, 22])

        let device = MTLCreateSystemDefaultDevice()!

        let buffer = try device.makeTypedBuffer(data: values, options: .storageModeShared)

//        let buffer = try device.makeBuffer(bytesOf: values, options: .storageModeShared)

        let prefixSum = try PrefixSum2(device: device)
        try prefixSum.create_pass_recursive(data: buffer, count: UInt32(values.count))

        let result = buffer.withUnsafeBufferPointer { buffer in
            Array(buffer)
        }
        print(result)

    }


    var device: MTLDevice
    var workgroup_size: SIMD2<UInt32>
    var compute: Compute
    var library: ShaderLibrary

    var threads_per_workgroup: UInt32 {
        workgroup_size.x * workgroup_size.y
    }

    var items_per_workgroup: UInt32 {
        threads_per_workgroup * 2
    }

    init(device: MTLDevice, workgroup_size: SIMD2<UInt32> = [16, 16]) throws {
        self.device = device
        self.workgroup_size = workgroup_size

        compute = try Compute(device: device, logger: nil)
        library = ShaderLibrary.bundle(.module, name: "debug")

        assert(log2(threads_per_workgroup).isMultiple(of: 2))
    }


    func create_pass_recursive(data: TypedMTLBuffer<UInt32>, count: UInt32) throws {
        print("A", count)
        let workgroup_count = ceildiv(count, items_per_workgroup)
        print(workgroup_count)
        let dispatchSize = device.find_optimal_dispatch_size(workgroup_count: Int(workgroup_count))
        let blockSumBuffer: TypedMTLBuffer<UInt32> = try device.makeTypedBuffer(count: Int(workgroup_count))

        var reduceDownsweepPipeline = try compute.makePipeline(function: library.reduce_downsweep)

        data.withUnsafeMTLBuffer { data in
            reduceDownsweepPipeline.arguments.items = .buffer(data)
        }

        blockSumBuffer.withUnsafeMTLBuffer { blockSumBuffer in
            reduceDownsweepPipeline.arguments.blockSums = .buffer(blockSumBuffer)
        }
        reduceDownsweepPipeline.arguments.WORKGROUP_SIZE_X = .int(workgroup_size.x)
        reduceDownsweepPipeline.arguments.WORKGROUP_SIZE_Y = .int(workgroup_size.y)
        reduceDownsweepPipeline.arguments.THREADS_PER_WORKGROUP = .int(threads_per_workgroup)
        reduceDownsweepPipeline.arguments.ITEMS_PER_WORKGROUP = .int(items_per_workgroup)
        reduceDownsweepPipeline.arguments.ELEMENT_COUNT = .int(count)
        reduceDownsweepPipeline.arguments.temp = .threadgroupMemoryLength(Int(items_per_workgroup * 2))

        try compute.run(pipeline: reduceDownsweepPipeline, width: dispatchSize.x, height: dispatchSize.y)

        if workgroup_count > 1 {
            try create_pass_recursive(data: blockSumBuffer, count: workgroup_count)

            var blocksumPipeline = try compute.makePipeline(function: library.add_block_sums)


            data.withUnsafeMTLBuffer { data in
                blocksumPipeline.arguments.items = .buffer(data)
            }
            blockSumBuffer.withUnsafeMTLBuffer { blockSumBuffer in
                blocksumPipeline.arguments.blockSums = .buffer(blockSumBuffer)
            }
            blocksumPipeline.arguments.WORKGROUP_SIZE_X = .int(workgroup_size.x)
            blocksumPipeline.arguments.WORKGROUP_SIZE_Y = .int(workgroup_size.y)
            blocksumPipeline.arguments.THREADS_PER_WORKGROUP = .int(threads_per_workgroup)
            blocksumPipeline.arguments.ELEMENT_COUNT = .int(count)

            print("B")
            try compute.run(pipeline: blocksumPipeline, width: dispatchSize.x, height: dispatchSize.y)
        }
    }

    func dispatch() throws {

    }

}

extension MTLDevice {
    func find_optimal_dispatch_size(workgroup_count: Int) -> SIMD2<Int> {
        if workgroup_count > 256 { // TODO: FIXME
            let x = Int(floor(sqrt(Double(workgroup_count))))
            let y = ceildiv(workgroup_count, x)
            return [x, y]
        }
        else {
            return [workgroup_count, 1]
        }
    }
}
