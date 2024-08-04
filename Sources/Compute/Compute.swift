import Metal
import os

/// The main struct that encapsulates the Metal compute environment.
///
/// This struct provides the core functionality for creating and executing compute tasks on the GPU.
public struct Compute {
    /// The Metal device used for compute operations.
    public let device: MTLDevice

    /// The logger used for debugging and performance monitoring.
    let logger: Logger?

    /// The Metal command queue used for submitting command buffers.
    let commandQueue: MTLCommandQueue

    /// Initializes a new Compute instance.
    ///
    /// - Parameters:
    ///   - device: The Metal device to use for compute operations.
    ///   - logger: An optional logger for debugging and performance monitoring.
    /// - Throws: `ComputeError.resourceCreationFailure` if unable to create the command queue.
    public init(device: MTLDevice, logger: Logger? = nil) throws {
        self.device = device
        self.logger = logger
        guard let commandQueue = device.makeCommandQueue() else {
            throw ComputeError.resourceCreationFailure
        }
        commandQueue.label = "Compute-MTLCommandQueue"
        self.commandQueue = commandQueue
    }

    /// Executes a compute task.
    ///
    /// This method creates a command buffer and executes the provided block with a `Task` instance.
    ///
    /// - Parameters:
    ///   - label: An optional label for the task, useful for debugging.
    ///   - block: A closure that takes a `Task` instance and returns a result.
    /// - Returns: The result of the block execution.
    /// - Throws: `ComputeError.resourceCreationFailure` if unable to create the command buffer,
    ///           or any error thrown by the provided block.
    public func task<R>(label: String? = nil, _ block: (Task) throws -> R) throws -> R {
        let commandBufferDescriptor = MTLCommandBufferDescriptor()
        guard let commandBuffer = commandQueue.makeCommandBuffer(descriptor: commandBufferDescriptor) else {
            throw ComputeError.resourceCreationFailure
        }
        commandBuffer.label = "\(label ?? "Unlabeled")-MTLCommandBuffer"
        defer {
            commandBuffer.commit()
            logger?.log("waitUntilCompleted")
            commandBuffer.waitUntilCompleted()
        }
        let task = Task(label: label, logger: logger, commandBuffer: commandBuffer)
        return try block(task)
    }

    /// Creates a compute pipeline.
    ///
    /// This method creates a `Pipeline` instance, which encapsulates a compute pipeline state and its associated arguments.
    ///
    /// - Parameters:
    ///   - function: The shader function to use for the pipeline.
    ///   - constants: A dictionary of constant values to be used when compiling the shader function.
    ///   - arguments: A dictionary of arguments to be passed to the shader function.
    /// - Returns: A new `Pipeline` instance.
    /// - Throws: Any error that occurs during the creation of the compute pipeline state.
    public func makePipeline(function: ShaderFunction, constants: [String: Argument] = [:], arguments: [String: Argument] = [:]) throws -> Pipeline {
        try Pipeline(device: device, function: function, constants: constants, arguments: arguments)
    }
}
