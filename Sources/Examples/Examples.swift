@main
enum Examples {
    static func main() async throws {
        let argument = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : ""

        switch argument {
        case "life":
            try await GameOfLife.main()
        case "memcopy":
            try MemcopyDemo.main()

        default:
            try await GameOfLife.main()
        }
    }
}
