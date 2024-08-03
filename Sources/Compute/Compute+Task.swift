import Metal
import os

public extension Compute {
    /// Represents a compute task that can be executed on the GPU.
    ///
    /// A `Task` encapsulates a Metal command buffer and provides methods to execute compute operations.
    struct Task {
        /// The label for this task, used for debugging and profiling.
        let label: String?

        /// The logger used for logging information about the task execution.
        let logger: Logger?

        /// The Metal command buffer associated with this task.
        let commandBuffer: MTLCommandBuffer

        /// Executes a block of code with a `Dispatcher`.
        ///
        /// This method is a convenience wrapper around the `run` method.
        ///
        /// - Parameter block: A closure that takes a `Dispatcher` and returns a result.
        /// - Returns: The result of the block execution.
        /// - Throws: Any error that occurs during the execution of the block.
        public func callAsFunction<R>(_ block: (Dispatcher) throws -> R) throws -> R {
            try run(block)
        }

        /// Runs a block of code with a `Dispatcher`.
        ///
        /// This method creates a compute command encoder and executes the provided block with a `Dispatcher`.
        ///
        /// - Parameter block: A closure that takes a `Dispatcher` and returns a result.
        /// - Returns: The result of the block execution.
        /// - Throws: `ComputeError.resourceCreationFailure` if unable to create a compute command encoder,
        ///           or any error that occurs during the execution of the block.
        public func run<R>(_ block: (Dispatcher) throws -> R) throws -> R {
            guard let commandEncoder = commandBuffer.makeComputeCommandEncoder() else {
                throw ComputeError.resourceCreationFailure
            }
            commandEncoder.label = "\(label ?? "Unlabeled")-MTLComputeCommandEncoder"

            defer {
                commandEncoder.endEncoding()
            }
            let dispatcher = Dispatcher(label: label, logger: logger, commandEncoder: commandEncoder)
            return try block(dispatcher)
        }
    }

    /// Handles the dispatching of compute operations to the GPU.
    ///
    /// A `Dispatcher` is responsible for setting up and executing compute operations using the provided `Pass`.
    struct Dispatcher {
        /// The label for this dispatcher, used for debugging and profiling.
        public let label: String?

        /// The logger used for logging information about the dispatch operation.
        public let logger: Logger?

        /// The Metal compute command encoder used to encode compute commands.
        public let commandEncoder: MTLComputeCommandEncoder

        /// Dispatches a compute operation using threadgroups.
        ///
        /// - Parameters:
        ///   - pass: The `Pass` containing the compute pipeline state and arguments.
        ///   - threadgroupsPerGrid: The number of threadgroups to dispatch in each dimension.
        ///   - threadsPerThreadgroup: The number of threads in each threadgroup.
        /// - Throws: Any error that occurs during the binding of arguments or dispatch.
        public func callAsFunction(pass: Pass, threadgroupsPerGrid: MTLSize, threadsPerThreadgroup: MTLSize) throws {
            logger?.info("maxTotalThreadsPerThreadgroup: \(pass.computePipelineState.maxTotalThreadsPerThreadgroup), threadExecutionWidth: \(pass.computePipelineState.threadExecutionWidth)")
            logger?.info("Dispatching \(String(describing: threadgroupsPerGrid)) threads with \(String(describing: threadsPerThreadgroup)) threads per threadgroup")
            commandEncoder.setComputePipelineState(pass.computePipelineState)
            try pass.bind(commandEncoder)
            commandEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        }

        /// Dispatches a compute operation using a specific number of threads.
        ///
        /// - Parameters:
        ///   - pass: The `Pass` containing the compute pipeline state and arguments.
        ///   - threads: The total number of threads to dispatch in each dimension.
        ///   - threadsPerThreadgroup: The number of threads in each threadgroup.
        /// - Throws: Any error that occurs during the binding of arguments or dispatch.
        public func callAsFunction(pass: Pass, threads: MTLSize, threadsPerThreadgroup: MTLSize) throws {
            logger?.info("maxTotalThreadsPerThreadgroup: \(pass.computePipelineState.maxTotalThreadsPerThreadgroup), threadExecutionWidth: \(pass.computePipelineState.threadExecutionWidth)")
            logger?.info("Dispatching \(String(describing: threads)) threads with \(String(describing: threadsPerThreadgroup)) threads per threadgroup")
            commandEncoder.setComputePipelineState(pass.computePipelineState)
            try pass.bind(commandEncoder)
            commandEncoder.dispatchThreads(threads, threadsPerThreadgroup: threadsPerThreadgroup)
        }
    }
}
