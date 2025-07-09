import Metal

/// Represents a Metal shader library.
@dynamicMemberLookup
public struct ShaderLibrary: Identifiable, Sendable {
    public enum ID: Hashable, Sendable {
        case bundle(URL, String?)
        case source(String)
    }

    /// The default shader library loaded from the main bundle.
    public static let `default` = Self.bundle(.main)

    /// Creates a shader library from a bundle.
    ///
    /// - Parameters:
    ///   - bundle: The bundle containing the shader library.
    ///   - name: The name of the metallib file (without extension). If nil, uses the default library.
    /// - Returns: A new ShaderLibrary instance.
    public static func bundle(_ bundle: Bundle, name: String? = nil) -> Self {
        return ShaderLibrary(id: .bundle(bundle.bundleURL, name)) { device in
            if let name {
                guard let url = bundle.url(forResource: name, withExtension: "metallib") else {
                    fatalError("Could not load metallib.")
                }
                let library = try device.makeLibrary(URL: url)
                library.label = "\(name).MTLLibrary"
                return library
            }
            let library = try device.makeDefaultLibrary(bundle: bundle)
            library.label = "Default.MTLLibrary"
            return library
        }
    }

    /// Creates a shader library from source code.
    ///
    /// - Parameters:
    ///   - source: The Metal shader source code as a string.
    ///   - enableLogging: Enable Metal shader logging.
    /// - Returns: A new ShaderLibrary instance.
    public static func source(_ source: String, enableLogging: Bool = false) -> Self {
        return ShaderLibrary(id: .source(source)) { device in
            let options = MTLCompileOptions()
            #if !targetEnvironment(simulator)
            if enableLogging {
                if #available(macOS 15, iOS 18, *) {
                    options.enableLogging = true
                }
                else {
                    fatalError("Metal logging is not available on this platform.")
                }
            }
            #endif
            return try device.makeLibrary(source: source, options: options)
        }
    }

    public var id: ID

    /// A closure that creates an MTLLibrary given an MTLDevice.
    public var make: @Sendable (MTLDevice) throws -> MTLLibrary

    /// Creates a ShaderFunction with the given name.
    ///
    /// - Parameter name: The name of the shader function.
    /// - Returns: A new ShaderFunction instance.
    public func function(name: String) -> ShaderFunction {
        ShaderFunction(library: self, name: name)
    }

    /// Allows accessing shader functions using dynamic member lookup.
    ///
    /// - Parameter name: The name of the shader function.
    /// - Returns: A new ShaderFunction instance.
    public subscript(dynamicMember name: String) -> ShaderFunction {
        ShaderFunction(library: self, name: name)
    }

    /// Creates an MTLLibrary for the given device.
    ///
    /// - Parameter device: The Metal device to create the library for.
    /// - Returns: The created MTLLibrary.
    /// - Throws: An error if the library creation fails.
    internal func makelibrary(device: MTLDevice) throws -> MTLLibrary {
        try make(device)
    }
}

/// Represents a compute shader function within a shader library.
public struct ShaderFunction: Identifiable {
    /// A unique identifier for the shader function.
    public let id: Composite<ShaderLibrary.ID, String>

    /// The shader library containing this function.
    public let library: ShaderLibrary

    /// The name of the shader function.
    public let name: String

    /// An array of shader constants associated with this function.
    public let constants: [ShaderConstant]

    /// Initializes a new shader function.
    ///
    /// - Parameters:
    ///   - library: The shader library containing this function.
    ///   - name: The name of the shader function.
    ///   - constants: An array of shader constants associated with this function. Defaults to an empty array.
    public init(library: ShaderLibrary, name: String, constants: [ShaderConstant] = []) {
        // BUG: https://github.com/schwa/Compute/issues/17 Make shader constants part of name id
        self.id = Composite(library.id, name)
        self.library = library
        self.name = name
        self.constants = constants
    }
}

/// Represents a constant value that can be passed to a shader function.
public struct ShaderConstant {
    /// The data type of the constant.
    var dataType: MTLDataType

    /// A closure that provides access to the constant's value.
    var accessor: ((UnsafeRawPointer) -> Void) -> Void

    /// Initializes a new shader constant with an array value.
    ///
    /// - Parameters:
    ///   - dataType: The Metal data type of the constant.
    ///   - value: The array value of the constant.
    // BUG: https://github.com/schwa/Compute/issues/16 Get away from Any and use an enum.
    public init(dataType: MTLDataType, value: [some Any]) {
        self.dataType = dataType
        accessor = { (callback: (UnsafeRawPointer) -> Void) in
            value.withUnsafeBytes { pointer in
                guard let baseAddress = pointer.baseAddress else {
                    fatalError("Could not get baseAddress.")
                }
                callback(baseAddress)
            }
        }
    }

    /// Initializes a new shader constant with a single value.
    ///
    /// - Parameters:
    ///   - dataType: The Metal data type of the constant.
    ///   - value: The value of the constant.
    // BUG: https://github.com/schwa/Compute/issues/16 Get away from Any and use an enum.
    public init(dataType: MTLDataType, value: some Any) {
        self.dataType = dataType
        accessor = { (callback: (UnsafeRawPointer) -> Void) in
            withUnsafeBytes(of: value) { pointer in
                guard let baseAddress = pointer.baseAddress else {
                    fatalError("Could not get baseAddress.")
                }
                callback(baseAddress)
            }
        }
    }

    /// Adds the constant value to a MTLFunctionConstantValues object.
    ///
    /// - Parameters:
    ///   - values: The MTLFunctionConstantValues object to add the constant to.
    ///   - name: The name of the constant in the shader code.
    public func add(to values: MTLFunctionConstantValues, name: String) {
        accessor { pointer in
            values.setConstantValue(pointer, type: dataType, withName: name)
        }
    }
}
