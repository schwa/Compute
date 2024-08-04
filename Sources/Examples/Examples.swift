@main
enum Examples {
    static func main() async throws {
        let argument = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : ""

        switch argument {
#if os(macOS)
        case "life":
            try await GameOfLife.main()
#endif
        case "baremetal":
            try BareMetalVsCompute.main()
        case "memcopy":
            try BareMetalVsCompute.main()

        default:
            try BareMetalVsCompute.main()
        }
    }
}
