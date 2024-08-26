import Foundation

protocol Demo {
    static func main() async throws
}

@main
enum Examples {
    static func main() async throws {
        let demos: [Demo.Type] = [
            PrefixSum.self,
            GameOfLife.self,
            BareMetalVsCompute.self,
            HelloWorldDemo.self,
            ImageInvert.self,
            Checkerboard.self,
            MaxValue.self,
            Reduce.self,
            RandomFill.self,
        ]

        let argument: String? = CommandLine.arguments.count > 1 ? CommandLine.arguments[1].lowercased() : nil
        if let argument {
            guard let demo = demos.first(where: { String(describing: $0).lowercased() == argument }) else {
                fatalError("No demo with name: \(argument)")
            }
            try await demo.main()
        }
        else {
            for (index, demo) in demos.enumerated() {
                print("\(index): \(String(describing: demo))")
            }
            print("Choice", terminator: ": ")
            let choice = Int(readLine()!)!
            let demo = demos[choice]
            print("Running \(String(describing: demo))...")
            try await demo.main()
        }
    }
}
