import Metal

public extension Compute {
    /// Represents a compute pass, encapsulating all the information needed to execute a compute operation.
    ///
    /// A `Pass` includes the shader function, its arguments, and the associated compute pipeline state.
    /// It provides the necessary context for dispatching compute operations on the GPU.
    struct Pass {
        /// The shader function associated with this pass.
        public let function: ShaderFunction

        /// A dictionary mapping argument names to their binding indices.
        internal let bindings: [String: Int]

        /// The arguments to be passed to the shader function.
        public var arguments: Arguments

        /// The compute pipeline state created from the shader function.
        public let computePipelineState: MTLComputePipelineState

        /// Initializes a new compute pass.
        ///
        /// This initializer creates a compute pipeline state from the provided shader function and sets up the necessary bindings and arguments.
        ///
        /// - Parameters:
        ///   - device: The Metal device on which the compute pass will be executed.
        ///   - function: The shader function to be used in this pass.
        ///   - constants: A dictionary of constant values to be used when compiling the shader function. Defaults to an empty dictionary.
        ///   - arguments: A dictionary of arguments to be passed to the shader function. Defaults to an empty dictionary.
        /// - Throws: An error if the compute pipeline state cannot be created or if there's an issue with the shader function.
        internal init(device: MTLDevice, function: ShaderFunction, constants: [String: Argument] = [:], arguments: [String: Argument] = [:]) throws {
            self.function = function

            let constantValues = MTLFunctionConstantValues()
            for (name, constant) in constants {
                constant.constantValue(constantValues, name)
            }

            let library = try function.library.makelibrary(device: device)

            let function = try library.makeFunction(name: function.name, constantValues: constantValues)
            function.label = "\(function.name)-MTLFunction"
            let computePipelineDescriptor = MTLComputePipelineDescriptor()
            computePipelineDescriptor.label = "\(function.name)-MTLComputePipelineState"
            computePipelineDescriptor.computeFunction = function
            let (computePipelineState, reflection) = try device.makeComputePipelineState(descriptor: computePipelineDescriptor, options: [.bindingInfo])
            guard let reflection else {
                throw ComputeError.resourceCreationFailure
            }
            bindings = Dictionary(uniqueKeysWithValues: reflection.bindings.map { binding in
                (binding.name, binding.index)
            })

            self.computePipelineState = computePipelineState
            self.arguments = Arguments(arguments: arguments)
        }

        /// The maximum total number of threads per threadgroup for this compute pipeline state.
        public var maxTotalThreadsPerThreadgroup: Int {
            computePipelineState.maxTotalThreadsPerThreadgroup
        }

        /// The thread execution width for this compute pipeline state.
        public var threadExecutionWidth: Int {
            computePipelineState.threadExecutionWidth
        }

        /// Binds the arguments to the provided compute command encoder.
        ///
        /// This method sets up the arguments for the compute operation, associating each argument with its corresponding binding point.
        ///
        /// - Parameter commandEncoder: The compute command encoder to which the arguments should be bound.
        /// - Throws: `ComputeError.missingBinding` if a required binding is not found for an argument.
        func bind(_ commandEncoder: MTLComputeCommandEncoder) throws {
            for (name, value) in arguments.arguments {
                guard let index = bindings[name] else {
                    throw ComputeError.missingBinding(name)
                }
                value.encode(commandEncoder, index)
            }
        }
    }
}
