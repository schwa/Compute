// swift-tools-version: 6.0

import PackageDescription

// swiftlint:disable:next explicit_top_level_acl
let package = Package(
    name: "Compute",
    platforms: [.macOS(.v15), .iOS(.v18)],
    products: [
        .library(name: "Compute", targets: ["Compute"]),
    ],
    dependencies: [
        .package(url: "https://github.com/schwa/MetalCompilerPlugin", branch: "jwight/develop"),
        .package(url: "https://github.com/schwa/SwiftGraphics", branch: "jwight/develop")
    ],
    targets: [
        .target(name: "Compute"),
        .executableTarget(
            name: "Examples",
            dependencies: [
                "Compute",
                .product(name: "MetalSupport", package: "SwiftGraphics"),
            ],
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
