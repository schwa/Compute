#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

uint3 thread_position_in_grid [[thread_position_in_grid]];

[[kernel]]
void bitonicSort(
    constant uint &numEntries [[buffer(0)]],
    constant uint &groupWidth [[buffer(1)]],
    constant uint &groupHeight [[buffer(2)]],
    constant uint &stepIndex [[buffer(3)]],
    device uint *entries [[buffer(4)]]
) {
    const uint index = thread_position_in_grid.x;
    const uint hIndex = index & (groupWidth - 1);
    const uint indexLeft = hIndex + (groupHeight + 1) * (index / groupWidth);
    const uint stepSize = stepIndex == 0 ? groupHeight - 2 * hIndex : (groupHeight + 1) / 2;
    const uint indexRight = indexLeft + stepSize;
    // Exit if out of bounds (for non-power of 2 input sizes)
    if (indexRight >= numEntries) {
        return;
    }
    const uint valueLeft = entries[indexLeft];
    const uint valueRight = entries[indexRight];
    // Swap entries if value is descending
    if (valueLeft > valueRight) {
        entries[indexLeft] = valueRight;
        entries[indexRight] = valueLeft;
    }
}
