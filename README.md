# Metal Compute Framework

This project provides a high-level Swift framework for working with Metal compute shaders. It simplifies the process of setting up and executing compute tasks on GPU using Apple's Metal API.

## Features

- Easy-to-use abstraction layer over Metal's compute pipeline
- Support for custom shader functions and libraries
- Flexible argument passing to compute shaders
- Logging support for debugging and performance monitoring

## Usage Example

Here's a simple example of how to use the Metal Compute Framework to perform a basic computation:

```swift
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

// Set up the compute environment
let device = MTLCreateSystemDefaultDevice()!
let compute = try Compute(device: device)

// Create input data
let count = 1000
let inA = [Int32](repeating: 1, count: count)
let inB = [Int32](repeating: 2, count: count)
var result = [Int32](repeating: 0, count: count)

// Create Metal buffers
let bufferA = device.makeBuffer(bytes: inA, length: MemoryLayout<Int32>.stride * count, options: [])!
let bufferB = device.makeBuffer(bytes: inB, length: MemoryLayout<Int32>.stride * count, options: [])!
let bufferResult = device.makeBuffer(length: MemoryLayout<Int32>.stride * count, options: [])!

// Create a shader library and function
let library = ShaderLibrary.source(source)
let function = library.add

// Create a compute pass
let pass = try compute.makePass(
    function: function,
    arguments: [
        "inA": .buffer(bufferA),
        "inB": .buffer(bufferB),
        "result": .buffer(bufferResult)
    ]
)

// Run the compute pass
try compute.run(pass: pass, count: count)

// Read back the results
let resultData = Data(bytesNoCopy: bufferResult.contents(), count: MemoryLayout<Int32>.stride * count, deallocator: .none)
result = resultData.withUnsafeBytes { $0.bindMemory(to: Int32.self) }

// Verify the results
for i in 0..<count {
    assert(result[i] == inA[i] + inB[i], "Computation error at index \(i)")
}
```

This example demonstrates how to:
1. Define a simple Metal shader for adding two arrays
2. Set up the compute environment
3. Create input data and Metal buffers
4. Create a shader library and function
5. Set up and run a compute pass
6. Read back and verify the results

## Requirements

- iOS 17.0+ / macOS 15+
- Swift 6.0+

## Installation

### Swift Package Manager

To integrate the Metal Compute Framework into your Xcode project using Swift Package Manager, add it to the dependencies value of your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/schwa/Compute.git", .from(from: "0.1"))
]
```

Then, specify it as a dependency of your target:

```swift
targets: [
    .target(name: "YourTarget", dependencies: ["MetalComputeFramework"]),
]
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions to the Compute Framework.

Note: Some of our initial documentation and tests were AI-generated. We encourage contributors to review, improve, and expand upon this foundation with real-world expertise and use cases.
