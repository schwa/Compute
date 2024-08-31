import Compute
import Metal
import MetalSupport
import os

// https://www.eecs.umich.edu/courses/eecs570/hw/parprefix.pdf

enum PrefixSum: Demo {

    static func main() async throws {
        //let values: [UInt32] = [3, 1, 7, 0, 4, 1, 6, 3] // [0, 3, 4, 11, 11, 15, 16, 22]
        let values: [UInt32] = (0..<(2 ** 20)).map { _ in UInt32.random(in: 0..<1024)}
        print(values.count)
        let prefixSum = values.prefixSum()
//        let values: [UInt32] = [0, 1, 2, 3, 4, 5, 6, 7] // [0, 0, 1, 3, 6, 10, 15, 21]
//        let values: [UInt32] = [1, 1, 1, 1, 1, 1, 1, 1] // [0, 1, 2, 3, 4, 5, 6, 7]
//        let values: [UInt32] = [3, 1, 7, 0, 4, 1, 6, 3] // [0, 3, 4, 11, 11, 15, 16, 22]

        let cpu = try await cpumain(values)
        let gpu = try await gpumain(values)

        assert(cpu == prefixSum)
        //assert(gpu == prefixSum)
    }


    static let source = #"""
        #include <metal_stdlib>
        #include <metal_logging>

        using namespace metal;

        uint threads_per_grid [[threads_per_grid]];
        uint thread_position_in_grid [[thread_position_in_grid]];

        [[kernel]]
        void reduce(
            device uint *data [[buffer(0)]],
            constant uint &count [[buffer(1)]]
        ) {
            if (thread_position_in_grid == 0) {
                os_log_default.log("count: %d, threads_per_grid: %d", count, threads_per_grid);
            }
            uint n = count;
            uint k = thread_position_in_grid;
            uint s = threads_per_grid;
            uint index1 = (k * (n / s) + (k + 1) * (n / s) - 2) / 2;
            uint index2 = (k + 1) * (n / s) - 1;
            data[index2] = data[index1] + data[index2];
        }

        [[kernel]]
        void down_sweep(
            device uint *data [[buffer(0)]],
            constant uint &count [[buffer(1)]]
        ) {
            uint n = count;
            uint k = thread_position_in_grid;
            uint s = threads_per_grid;
            uint index1 = (k * (n / s) + (k + 1) * (n / s) - 2) / 2;
            uint index2 = (k + 1) * (n / s) - 1;
            uint temp = data[index1];
            data[index1] = data[index2];
            data[index2] = temp + data[index2];
        }
    """#
    static func gpumain(_ values: [UInt32]) async throws -> [UInt32] {
        let device = MTLCreateSystemDefaultDevice()!
        let data = values.withUnsafeBufferPointer { buffer in
            device.makeBuffer(bytes: buffer.baseAddress!, length: buffer.count * MemoryLayout<UInt32>.stride)!
        }

        let logger = Logger()
        let compute = try Compute(device: device, logger: logger)
        let library: ShaderLibrary
        library = ShaderLibrary.source(source, enableLogging: true)
        var reduce = try compute.makePipeline(function: library.reduce)
        reduce.arguments.data = .buffer(data)
        reduce.arguments.count = .int(values.count)
        let n = values.count
        for d in stride(from: log2(n) - 1, through: 0, by: -1) {
            let s = 2 ** (d)
            try compute.run(pipeline: reduce, width: s)
        }


        data.contents().assumingMemoryBound(to: UInt32.self)[n - 1] = 0
        var downsweep = try compute.makePipeline(function: library.down_sweep)
        downsweep.arguments.data = .buffer(data)
        downsweep.arguments.count = .int(values.count)
        for d in stride(from: 0, through: log2(n) - 1, by: 1) {
            let s = 2 ** (d)
            try compute.run(pipeline: downsweep, width: s)
        }
        return Array<UInt32>(data)
    }

    static func cpureduce(_ values: [UInt32]) -> [UInt32] {
        var x = values
        let n = UInt32(x.count)
        for d in stride(from: UInt32(log2(n) - 1), through: 0, by: -1) {
            let s = 2 ** d
            for k in stride(from: 0, through: s - 1, by: 1) {
                let index1 = (k * (n / s) + (k + 1) * (n / s) - 2) / 2
                let index2 = (k + 1) * (n / s) - 1
                x[Int(index2)] = x[Int(index1)] + x[Int(index2)]
            }
        }
        return x
    }

    static func cpudownsweep(_ values: [UInt32]) -> [UInt32] {
        var x = values
        let n = x.count
        x[n - 1] = 0
        for d in stride(from: 0, through: log2(n) - 1, by: 1) {
            let s = 2 ** (d)
            for k in stride(from: 0, through: s - 1, by: 1) {
                let index1 = (k * (n / s) + (k + 1) * (n / s) - 2) / 2
                let index2 = (k + 1) * (n / s) - 1
                let temp = x[index1]
                x[index1] = x[index2]
                x[index2] = temp + x[index2]
            }
        }
        return x
    }

    static func cpumain(_ values: [UInt32]) async throws -> [UInt32] {
        var values = cpureduce(values)
        values = cpudownsweep(values)
        return values
    }
}
