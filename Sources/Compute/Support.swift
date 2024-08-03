import Metal

/// Enumerates the possible errors that can occur in the Compute framework.
public enum ComputeError: Error {
    /// Indicates a failure in creating a required resource, such as a command queue or buffer.
    case resourceCreationFailure

    /// Indicates that a required binding for an argument is missing.
    /// - Parameter String: The name of the missing binding.
    case missingBinding(String)
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

    /// Runs a compute pass with the specified arguments and thread count.
    ///
    /// This method provides a convenient way to execute a single compute pass with optional additional arguments.
    ///
    /// - Parameters:
    ///   - pass: The compute pass to run.
    ///   - arguments: Optional additional arguments to merge with the pass's existing arguments.
    ///   - count: The number of threads to dispatch.
    /// - Throws: Any error that occurs during the execution of the compute pass.
    func run(pass: Pass, arguments: [String: Argument]? = nil, count: Int) throws {
        var pass = pass
        if let arguments {
            var existing = pass.arguments.arguments
            existing.merge(arguments) { $1 }
            pass.arguments = .init(arguments: existing)
        }
        try task { task in
            try task { dispatch in
                let maxTotalThreadsPerThreadgroup = pass.computePipelineState.maxTotalThreadsPerThreadgroup
                try dispatch(pass: pass, threads: MTLSize(width: count, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: maxTotalThreadsPerThreadgroup, height: 1, depth: 1))
            }
        }
    }
}
