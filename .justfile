list:
    @just --list

build-docs:
    xcrun xcodebuild docbuild -scheme Compute -derivedDataPath /tmp/compute-docbuild -destination platform=macOS,arch=arm64
    cp -r /tmp/compute-docbuild/Build/Products/Debug/Compute.doccarchive ~/Desktop

    xcrun docc process-archive transform-for-static-hosting ~/Desktop/Compute.doccarchive --hosting-base-path / --output-path ~/Desktop/Compute-HTML/

concurrency-check:
    swift clean
    swift build -Xswiftc -strict-concurrency=complete

swift-six-check:
    swift clean
    SWIFT_VERSION=6 swift build --verbose

run-prefix-sum:
    env DEVELOPER_DIR=/Applications/Xcode-16.4.0.app xcrun swift run Examples PrefixSum
