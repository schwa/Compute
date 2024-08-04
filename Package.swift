// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "Compute",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(name: "Compute", targets: ["Compute"])
    ],
    dependencies: [
        .package(url: "https://github.com/schwa/MetalCompilerPlugin", from: "0.0.3")
    ],
    targets: [
        .target(name: "Compute"),
        .executableTarget(
            name: "Examples",
            dependencies: ["Compute"],
            resources: [
                .copy("Bundle.txt")
            ],
            plugins: [
                .plugin(name: "MetalCompilerPlugin", package: "MetalCompilerPlugin")
            ]
        ),
        .testTarget(name: "ComputeTests", dependencies: ["Compute"])
    ]
)
