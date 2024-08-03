import Compute
import Metal

// Metal shader source code
let source = """
#include <metal_stdlib>
using namespace metal;

kernel void add(device int* inA [[buffer(0)]],
                device int* inB [[buffer(1)]],
                device int* result [[buffer(2)]],
                uint id [[thread_position_in_grid]]) {
    result[id] = inA[id] + inB[id];
}
"""


// Create input data
let count = 1000
let inA = [Int32](repeating: 1, count: count)
let inB = [Int32](repeating: 2, count: count)
var result = [Int32](repeating: 0, count: count)

let device = MTLCreateSystemDefaultDevice()!

// Create Metal buffers
let bufferA = device.makeBuffer(bytes: inA, length: MemoryLayout<Int32>.stride * count, options: [])!
let bufferB = device.makeBuffer(bytes: inB, length: MemoryLayout<Int32>.stride * count, options: [])!
let bufferResult = device.makeBuffer(length: MemoryLayout<Int32>.stride * count, options: [])!
print(bufferResult)

//// Set up the compute environment
let compute = try Compute(device: device)
// Create a shader library and function
let library = ShaderLibrary.source(source)
let function = library.add

// Create a compute pass
var pass = try compute.makePass(function: function)
pass.arguments.inA = .buffer(bufferA)
pass.arguments.inB = .buffer(inB)
pass.arguments.result = .buffer(bufferResult)

// Run the compute pass
try compute.run(pass: pass, count: count)

// Read back the results
let resultData = Data(bytesNoCopy: bufferResult.contents(), count: MemoryLayout<Int32>.stride * count, deallocator: .none)
result = resultData.withUnsafeBytes { $0.bindMemory(to: Int32.self) }

// Verify the results
for i in 0..<count {
    assert(result[i] == inA[i] + inB[i], "Computation error at index \(i)")
}
