// swift-tools-version: 6.0

import PackageDescription

// swiftlint:disable:next explicit_top_level_acl
let package = Package(
    name: "Compute",
    platforms: [.macOS(.v15), .iOS(.v18)],
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
                .copy("Bundle.txt"),
                .process("Resources/Media.xcassets")
            ],
            plugins: [
                .plugin(name: "MetalCompilerPlugin", package: "MetalCompilerPlugin")
            ]
        ),
        .testTarget(name: "ComputeTests", dependencies: ["Compute"])
    ]
)
