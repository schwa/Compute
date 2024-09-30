import Metal

/// Enumerates the possible errors that can occur in the Compute framework.
public enum ComputeError: Error {
    /// Indicates that a required binding for an argument is missing.
    /// - Parameter String: The name of the missing binding.
    case missingBinding(String)

    case nonuniformThreadgroupsSizeNotSupported

    /// Indicates a failure in creating a required resource, such as a command queue or buffer.
    case resourceCreationFailure

}

public extension Compute {
    /// Dispatches a compute operation using a more convenient syntax.
    ///
    /// This method wraps the `task` method, providing a simpler interface for dispatching compute operations.
    ///
    /// - Parameters:
    ///   - label: An optional label for the dispatch operation, useful for debugging.
    ///   - block: A closure that takes a `Dispatcher` and returns a result.
    /// - Returns: The result of the block execution.
    /// - Throws: Any error that occurs during the execution of the block or the underlying task.
    func dispatch<R>(label: String? = nil, _ block: (Dispatcher) throws -> R) throws -> R {
        try task(label: label) { task in
            try task { dispatch in
                try block(dispatch)
            }
        }
    }
}

public extension Compute {
    func run(pipeline: Pipeline, arguments: [String: Argument]? = nil, threads: MTLSize, threadsPerThreadgroup: MTLSize) throws {
        var pipeline = pipeline
        if let arguments {
            var existing = pipeline.arguments.arguments
            existing.merge(arguments) { $1 }
            pipeline.arguments = .init(arguments: existing)
        }
        try task { task in
            try task { dispatch in
                try dispatch(pipeline: pipeline, threads: threads, threadsPerThreadgroup: threadsPerThreadgroup)
            }
        }
    }

    func run(pipeline: Pipeline, arguments: [String: Argument]? = nil, threadgroupsPerGrid: MTLSize, threadsPerThreadgroup: MTLSize) throws {
        var pipeline = pipeline
        if let arguments {
            var existing = pipeline.arguments.arguments
            existing.merge(arguments) { $1 }
            pipeline.arguments = .init(arguments: existing)
        }
        try task { task in
            try task { dispatch in
                try dispatch(pipeline: pipeline, threadgroupsPerGrid: threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
            }
        }
    }


    /// Runs a compute pipeline with the specified arguments and thread count.
    ///
    /// This method provides a convenient way to execute a single compute pipeline with optional additional arguments.
    ///
    /// - Parameters:
    ///   - pipeline: The compute pipeline to run.
    ///   - arguments: Optional additional arguments to merge with the pipeline's existing arguments.
    ///   - width: The number of threads to dispatch.
    /// - Throws: Any error that occurs during the execution of the compute pipeline.
    @available(*, deprecated, message: "Deprecated")
    func run(pipeline: Pipeline, arguments: [String: Argument]? = nil, width: Int) throws {
        var pipeline = pipeline
        if let arguments {
            var existing = pipeline.arguments.arguments
            existing.merge(arguments) { $1 }
            pipeline.arguments = .init(arguments: existing)
        }
        try task { task in
            try task { dispatch in
                let threadPerThreadgroup = min(ceildiv(width, pipeline.threadExecutionWidth) * pipeline.threadExecutionWidth, pipeline.maxTotalThreadsPerThreadgroup)
                try dispatch(pipeline: pipeline, threads: MTLSize(width: width, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: threadPerThreadgroup , height: 1, depth: 1))
            }
        }
    }

    @available(*, deprecated, message: "Deprecated")
    func run(pipeline: Pipeline, arguments: [String: Argument]? = nil, width: Int, height: Int) throws {
        var pipeline = pipeline
        if let arguments {
            var existing = pipeline.arguments.arguments
            existing.merge(arguments) { $1 }
            pipeline.arguments = .init(arguments: existing)
        }
        try task { task in
            try task { dispatch in
                let maxTotalThreadsPerThreadgroup = pipeline.computePipelineState.maxTotalThreadsPerThreadgroup

                let threadsPerThreadgroupWidth = Int(sqrt(Double(maxTotalThreadsPerThreadgroup)))
                let threadsPerThreadgroupHeight = maxTotalThreadsPerThreadgroup / threadsPerThreadgroupWidth

                try dispatch(pipeline: pipeline, threads: MTLSize(width: width, height: height, depth: 1), threadsPerThreadgroup: MTLSize(width: threadsPerThreadgroupWidth, height: threadsPerThreadgroupHeight, depth: 1))
            }
        }
    }
}

internal extension MTLSize {
    var shortDescription: String {
        return "[\(width), \(height), \(depth)]"
    }
}

internal func ceildiv <T>(_ x: T, _ y: T) -> T where T: BinaryInteger {
    (x + y - 1) / y
}

internal func align(_ value: Int, alignment: Int) -> Int {
    (value + alignment - 1) / alignment * alignment
}

public extension Compute.Pipeline {
    func calculateThreadgroupSize(threads: MTLSize) -> MTLSize {
        switch (threads.width, threads.height, threads.depth) {
        case (_, 1, 1):
            if threads.width < threadExecutionWidth {
                return threads
            }
            let threadPerThreadgroup = min(align(threads.width, alignment: threadExecutionWidth), maxTotalThreadsPerThreadgroup)
            return MTLSize(width: threadPerThreadgroup, height: 1, depth: 1)
        case (_, _, 1):
            fatalError()
        default:
            fatalError()
        }
    }
}

public extension MTLCommandEncoder {
    func withDebugGroup<R>(_ string: String, enabled: Bool = true, _ closure: () throws -> R) rethrows -> R {
        if enabled {
            pushDebugGroup(string)
        }
        defer {
            if enabled {
                popDebugGroup()
            }
        }
        return try closure()
    }
}
