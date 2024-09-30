import Compute
import Metal
import MetalSupportLite
import os

class PrefixSum3: Demo {

    static func main() async throws {
        let device = MTLCreateSystemDefaultDevice()!
        //let values: [UInt32] = Array(0..<4096)
        let values: [UInt32] = Array(repeating: 1, count: 1000000)
        print(values.count)
        let expectedResult = values.prefixSum()
        print(expectedResult.prefix(100))
        let data = try device.makeTypedBuffer(data: values, options: .storageModeShared)
        let demo = try PrefixSum3(device: device)
        print("####################################")
        print("PREFIX SUM", demo.threads_per_workgroup, demo.items_per_workgroup)
        try demo.create_pass_recursive(data: data, count: values.count)

        for block in demo.blocks {
            try block()
        }

        let result = Array<UInt32>(data)
        print(result.prefix(100))
        print(result == expectedResult)
    }

    var device: MTLDevice
    var compute: Compute
    var library: ShaderLibrary
    var workgroup_size: MTLSize
    var threads_per_workgroup: Int
    var items_per_workgroup: Int

    var blocks: [() throws -> Void] = []

    init(device: MTLDevice) throws {
        self.device = device
        let logger = Logger()
        compute = try Compute(device: device, logger: nil)
        library = ShaderLibrary.bundle(.module, name: "debug")
        workgroup_size = MTLSize(width: 16, height: 16, depth: 1)
        threads_per_workgroup = workgroup_size.width * workgroup_size.height
        items_per_workgroup = 2 * threads_per_workgroup // 2 items are processed per thread
        if !log2(threads_per_workgroup).isMultiple(of: 2) {
            fatalError("workgroup_size.x * workgroup_size.y must be a power of two.")
        }
    }

    func create_pass_recursive(data: TypedMTLBuffer<UInt32>, count: Int) throws {
        let workgroup_count = ceildiv(count, items_per_workgroup)
        let blockSumBuffer: TypedMTLBuffer<UInt32> = try device.makeTypedBuffer(capacity: workgroup_count)

        var reduceDownsweepPipeline = try compute.makePipeline(function: library.reduce_downsweep)
        reduceDownsweepPipeline.arguments.items = .buffer(data)
        reduceDownsweepPipeline.arguments.blockSums = .buffer(blockSumBuffer)
        reduceDownsweepPipeline.arguments.ELEMENT_COUNT = .int(count)
        reduceDownsweepPipeline.arguments.temp = .threadgroupMemoryLength(items_per_workgroup * 2 * MemoryLayout<UInt32>.stride)

        let dispatchSize = find_optimal_dispatch_size(reduceDownsweepPipeline, workgroup_count)

        blocks.append {
            print("reduceDownsweepPipeline dispatchSize: \(dispatchSize)")
            try self.compute.run(pipeline: reduceDownsweepPipeline, threadgroupsPerGrid: dispatchSize, threadsPerThreadgroup: MTLSize(width: self.threads_per_workgroup, height: 1, depth: 1))
        }

        if workgroup_count > 1 {
            try create_pass_recursive(data: data, count: workgroup_count)
//            device uint* items [[buffer(0)]],
//            device uint* blockSums [[buffer(1)]],
//            constant uint& THREADS_PER_WORKGROUP [[buffer(4)]],
//            constant uint& ELEMENT_COUNT [[buffer(6)]]

            var add_block_sums = try compute.makePipeline(function: library.add_block_sums)
            add_block_sums.arguments.items = .buffer(data)
            add_block_sums.arguments.blockSums = .buffer(blockSumBuffer)
            add_block_sums.arguments.ELEMENT_COUNT = .int(count)
            blocks.append {
                print("add_block_sums dispatchSize: \(dispatchSize)")
                try self.compute.run(pipeline: add_block_sums, threadgroupsPerGrid: dispatchSize, threadsPerThreadgroup: MTLSize(width: self.threads_per_workgroup, height: 1, depth: 1))
            }
        }

    }
}

func find_optimal_dispatch_size(_ pipeline: Compute.Pipeline, _ workgroup_count: Int) -> MTLSize {



//    if workgroup_count > 256 { // TODO: FIXME
//        let x = Int(floor(sqrt(Double(workgroup_count))))
//        let y = ceildiv(workgroup_count, x)
//        return MTLSize(x, y, 1)
//    }
//    else {
        return MTLSize(workgroup_count, 1, 1)
//    }
}
