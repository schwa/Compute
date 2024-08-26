import Foundation

@main
enum Examples {
    static func main() async throws {
        let demos: [(String, () async throws -> Void)] = [
            ("life", { try await GameOfLife.main() }),
            ("baremetal", { try BareMetalVsCompute.main() }),
            ("memcopy", { try BareMetalVsCompute.main() }),
            ("helloworld", { try HelloWorldDemo.main() }),
            ("ImageInvert", { try ImageInvert.main() }),
            ("Checkerboard", { try Checkerboard.main() }),
            ("MaxValue", { try MaxValue.main() }),
        ]
        let argument: String? = CommandLine.arguments.count > 1 ? CommandLine.arguments[1].lowercased() : nil
        if let argument {
            guard let demo = demos.first(where: { $0.0.lowercased() == argument }) else {
                fatalError("No demo with name: \(argument)")
            }
            try await demo.1()
        }
        else {
            for (index, name) in demos.map(\.0).enumerated() {
                print("\(index): \(name)")
            }
            print("Choice", terminator: ": ")
            let choice = Int(readLine()!)!
            let demo = demos[choice]
            print("Running \(demo.0)...")
            try await demo.1()
        }
    }
}
