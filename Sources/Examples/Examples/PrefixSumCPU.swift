
/// - Note: Prefix sums have various applications, including:
///   - Range sum queries
///   - Finding continuous subarrays with a given sum
///   - Image processing (integral images)
///   - Financial applications (cumulative returns)
///   - Computational geometry
///
/// - SeeAlso: Other related concepts:
///   1. 2D Prefix Sum: Used for efficient range sum queries in 2D arrays.
///   2. Parallel Prefix Sum: An algorithm for efficient parallel computation of prefix sums.
///   3. Fenwick Tree (Binary Indexed Tree): A data structure that can efficiently update elements and calculate prefix sums.

public extension Collection where Element: BinaryInteger {
    /// Calculates the inclusive prefix sum (cumulative sum) of the collection.
    ///
    /// This method computes a new array where each element is the sum of all elements
    /// up to and including that position in the original collection.
    ///
    /// - Complexity: O(n), where n is the number of elements in the collection.
    ///
    /// - Returns: An array containing the inclusive prefix sums of the collection.
    ///
    /// - Example:
    ///   ```
    ///   let numbers = [1, 2, 3, 4]
    ///   let prefixSums = numbers.prefixSumInclusive()
    ///   // prefixSums is [1, 3, 6, 10]
    ///   ```
    @inline(__always) func prefixSumInclusive() -> [Element] {
        reduce(into: []) { result, value in
            result.append((result.last ?? 0) + value)
        }
    }

    /// Calculates the exclusive prefix sum of the collection.
    ///
    /// This method computes a new array where each element is the sum of all previous elements
    /// in the original collection, excluding the current element.
    ///
    /// - Complexity: O(n), where n is the number of elements in the collection.
    ///
    /// - Returns: An array containing the exclusive prefix sums of the collection.
    ///            The returned array has the same number of elements as the original collection,
    ///            with the first element always being zero.
    ///
    /// - Example:
    ///   ```
    ///   let numbers = [1, 2, 3, 4]
    ///   let exclusivePrefixSums = numbers.prefixSumExclusive()
    ///   // exclusivePrefixSums is [0, 1, 3, 6]
    ///   ```
    @inline(__always) func prefixSumExclusive() -> [Element] {
        reduce(into: [0]) { result, value in
            result.append(result.last! + value)
        }.dropLast()
    }
}
