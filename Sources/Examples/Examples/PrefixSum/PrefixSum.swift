public extension Collection where Element: BinaryInteger {
    @inline(__always) func prefixSumExclusive() -> [Element] {
        reduce(into: [0]) { result, value in
            result.append(result.last! + value)
        }.dropLast()
    }
}
