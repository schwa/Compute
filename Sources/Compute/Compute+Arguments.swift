import Metal
import simd
import SwiftUI

public extension Compute {
    /// A structure that holds and manages arguments for a compute pipeline.
    ///
    /// This structure uses dynamic member lookup to provide a convenient way to access and set arguments.
    @dynamicMemberLookup
    struct Arguments {
        /// The underlying dictionary storing the arguments.
        internal var arguments: [String: Argument]

        /// Provides access to arguments using dynamic member lookup.
        ///
        /// - Parameter name: The name of the argument.
        /// - Returns: The argument value if it exists, or nil if it doesn't.
        public subscript(dynamicMember name: String) -> Argument? {
            get {
                arguments[name]
            }
            set {
                arguments[name] = newValue
                // NOTE: It would be nice to assign name as a label to buffers/textures that have no name.
            }
        }
    }

    /// Represents an argument that can be passed to a compute shader.
    ///
    /// This struct encapsulates the logic for encoding the argument to a compute command encoder
    /// and setting it as a constant value in a function constant values object.
    struct Argument {
        /// A closure that encodes the argument to a compute command encoder.
        internal var encode: (MTLComputeCommandEncoder, Int) -> Void

        /// A closure that sets the argument as a constant value in a function constant values object.
        internal var constantValue: (MTLFunctionConstantValues, String) -> Void

        public init(encode: @escaping (MTLComputeCommandEncoder, Int) -> Void, constantValue: @escaping (MTLFunctionConstantValues, String) -> Void) {
            self.encode = encode
            self.constantValue = constantValue
        }
    }

}


public extension Compute.Argument {

        /// Creates an integer argument.
        ///
        /// - Parameter value: The integer value.
        /// - Returns: An `Argument` instance representing the integer.
        static func int(_ value: some BinaryInteger) -> Self {
            .init { encoder, index in
                withUnsafeBytes(of: value) { buffer in
                    guard let baseAddress = buffer.baseAddress else {
                        fatalError("Could not get baseAddress.")
                    }
                    encoder.setBytes(baseAddress, length: buffer.count, index: index)
                }
            }
            constantValue: { constants, name in
                withUnsafeBytes(of: value) { buffer in
                    guard let baseAddress = buffer.baseAddress else {
                        fatalError("Could not get baseAddress.")
                    }
                    switch value {
                    case is Int8:
                        constants.setConstantValue(baseAddress, type: .char, withName: name)

                    case is UInt8:
                        constants.setConstantValue(baseAddress, type: .uchar, withName: name)

                    case is Int16:
                        constants.setConstantValue(baseAddress, type: .short, withName: name)

                    case is UInt16:
                        constants.setConstantValue(baseAddress, type: .ushort, withName: name)

                    case is Int32:
                        constants.setConstantValue(baseAddress, type: .int, withName: name)

                    case is UInt32:
                        constants.setConstantValue(baseAddress, type: .uint, withName: name)

                    default:
                        fatalError("Unsupported integer type.")
                    }
                }
            }
        }

        /// Creates a float argument.
        ///
        /// - Parameter value: The float value.
        /// - Returns: An `Argument` instance representing the float.
        static func float(_ value: Float) -> Self {
            .init { encoder, index in
                withUnsafeBytes(of: value) { buffer in
                    guard let baseAddress = buffer.baseAddress else {
                        fatalError("Could not get baseAddress.")
                    }
                    encoder.setBytes(baseAddress, length: buffer.count, index: index)
                }
            }
            constantValue: { constants, name in
                withUnsafeBytes(of: value) { buffer in
                    guard let baseAddress = buffer.baseAddress else {
                        fatalError("Could not get baseAddress.")
                    }
                    constants.setConstantValue(baseAddress, type: .float, withName: name)
                }
            }
        }

        /// Creates a boolean argument.
        ///
        /// - Parameter value: The boolean value.
        /// - Returns: An `Argument` instance representing the boolean.
        static func bool(_ value: Bool) -> Self {
            .init { encoder, index in
                withUnsafeBytes(of: value) { buffer in
                    guard let baseAddress = buffer.baseAddress else {
                        fatalError("Could not get baseAddress.")
                    }
                    encoder.setBytes(baseAddress, length: buffer.count, index: index)
                }
            }
            constantValue: { constants, name in
                withUnsafeBytes(of: value) { buffer in
                    guard let baseAddress = buffer.baseAddress else {
                        fatalError("Could not get baseAddress.")
                    }
                    constants.setConstantValue(baseAddress, type: .bool, withName: name)
                }
            }
        }

        /// Creates a buffer argument.
        ///
        /// - Parameters:
        ///   - buffer: The Metal buffer to be used as an argument.
        ///   - offset: The offset within the buffer. Defaults to 0.
        /// - Returns: An `Argument` instance representing the buffer.
        static func buffer(_ buffer: MTLBuffer, offset: Int = 0) -> Self {
            .init { encoder, index in
                encoder.setBuffer(buffer, offset: offset, index: index)
            }
            constantValue: { _, _ in
                fatalError("Unimplemented")
            }
        }

        /// Creates a texture argument.
        ///
        /// - Parameter texture: The Metal texture to be used as an argument.
        /// - Returns: An `Argument` instance representing the texture.
        static func texture(_ texture: MTLTexture) -> Self {
            Self { encoder, index in
                encoder.setTexture(texture, index: index)
            }
            constantValue: { _, _ in
                fatalError("Unimplemented")
            }
        }

        /// Creates an argument from a simd vector
        ///
        /// - Parameter value: The vector value.
        /// - Returns: An `Argument` instance representing the boolean.
        static func vector<V>(_ value: V) -> Self where V: SIMD {
            .init { encoder, index in
                withUnsafeBytes(of: value) { buffer in
                    guard let baseAddress = buffer.baseAddress else {
                        fatalError("Could not get baseAddress.")
                    }
                    encoder.setBytes(baseAddress, length: buffer.count, index: index)
                }
            }
            constantValue: { _, _ in
                fatalError("Unimplemented")
            }
        }

        /// Creates an argument from a simd vector
        ///
        /// - Parameter value: The vector value.
        /// - Returns: An `Argument` instance representing the boolean.
        static func float4(_ value: SIMD4<Float>) -> Self {
            .vector(value)
        }

        /// Creates an argument from a simd vector
        ///
        /// - Parameter value: The vector value.
        /// - Returns: An `Argument` instance representing the boolean.
        static func color(_ value: Color) throws -> Self {
            let cgColor = value.resolve(in: .init()).cgColor
            guard let colorSpace = CGColorSpace(name: CGColorSpace.genericRGBLinear) else {
                throw ComputeError.resourceCreationFailure
            }
            guard let components = cgColor.converted(to: colorSpace, intent: .defaultIntent, options: nil)?.components else {
                throw ComputeError.resourceCreationFailure
            }

            // TODO: This assumes the pass wants a SIMD4<Float> - we can use reflection to work out what is needed and convert appropriately. This will mean we need to refactor Compute.Argument

            let vector = SIMD4<Float>([Float(components[0]), Float(components[1]), Float(components[2]), Float(components[3])])
            return .vector(vector)
        }

        /// Creates an argument from a threadgroup memory length
        static func threadgroupMemoryLength(_ value: Int) -> Self {
            Self { encoder, index in
                encoder.setThreadgroupMemoryLength(value, index: index)
            }
            constantValue: { _, _ in
                fatalError("Unimplemented")
            }
        }

    static func buffer<T>(_ array: [T], offset: Int = 0, label: String? = nil) -> Self {
        .init { encoder, index in
            let buffer = array.withUnsafeBufferPointer { buffer in
                return encoder.device.makeBuffer(bytes: buffer.baseAddress!, length: buffer.count * MemoryLayout<T>.stride, options: [])
            }
            if let label {
                buffer?.label = label
            }
            encoder.setBuffer(buffer, offset: offset, index: index)
        }
        constantValue: { _, _ in
            fatalError("Unimplemented")
        }
    }

}
