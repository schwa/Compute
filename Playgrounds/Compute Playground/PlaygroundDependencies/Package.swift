// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "PlaygroundDependencies",
    products: [
        // Products define the executables and libraries a package produces, making them visible to other packages.
        .library(
            name: "PlaygroundDependencies",
            targets: ["PlaygroundDependencies"]),
    ],
    targets: [
        // Targets are the basic building blocks of a package, defining a module or a test suite.
        // Targets can depend on other targets in this package and products from dependencies.
        .target(
            name: "PlaygroundDependencies"),
        .testTarget(
            name: "PlaygroundDependenciesTests",
            dependencies: ["PlaygroundDependencies"]
        ),
    ]
)

package.dependencies = [
    .package(url: "https://github.com/schwa/Compute", from: "0.0.2")
]
package.targets = [
    .target(name: "PlaygroundDependencies",
        dependencies: [
            .product(name: "Compute", package: "Compute")
        ]
    )
]
package.platforms = [
    .iOS("17.0"),
    .macOS("14.0")
]
