#include <metal_stdlib>
#include <simd/simd.h>
#include <metal_logging>

using namespace metal;

inline uchar key(uint value, uint shift) {
    return (value >> shift) & 0xFF;
}

// MARK: -

[[kernel]]
void histogram(
    uint2 thread_position_in_grid [[thread_position_in_grid]],
    device uint *input [[buffer(0)]],
    constant uint &shift [[buffer(2)]],
    device atomic_uint *histogram [[buffer(3)]]
)
{
    const uchar bucket = thread_position_in_grid.x;
    const uint index = thread_position_in_grid.y;
    if (bucket == key(input[index], shift)) {
        atomic_fetch_add_explicit(&histogram[bucket], 1, memory_order_relaxed);
    }
}

// MARK: -

[[kernel]]
void prefix_sum_inclusive(
    device uint *histogram [[buffer(0)]]
)
{
    for (int index = 1; index != 256; index++) {
        histogram[index] += histogram[index - 1];
    }
}

[[kernel]]
void prefix_sum_exclusive(
    device uint *histogram [[buffer(0)]]
)
{
    uint sum = 0;
    for (int i = 0; i != 256; i++) {
        uint t = histogram[i];
        histogram[i] = sum;
        sum += t;
    }
}

// MARK: -

[[kernel]]
void shuffle(
    uint2 thread_position_in_grid [[thread_position_in_grid]],
    device atomic_uint *histogram [[buffer(0)]],
    device uint *input [[buffer(1)]],
    device uint *output [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    constant uint &shift [[buffer(4)]]
)
{
    const uchar bucket = thread_position_in_grid.x;
    for (int index = count - 1; index >= 0; --index) {
        if (bucket == key(input[index], shift)) {
            auto old = atomic_fetch_add_explicit(&histogram[bucket], -1, memory_order_relaxed);
            output[old - 1] = input[index];
        }
    }
}

[[kernel]]
void shuffle2(
    uint2 thread_position_in_grid [[thread_position_in_grid]],
    device atomic_uint *histogram [[buffer(0)]],
    device uint *input [[buffer(1)]],
    device uint *output [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    constant uint &shift [[buffer(4)]]
)
{
    const uchar bucket = thread_position_in_grid.x;
    for (uint index = 0; index != count; index++) {
        if (bucket == key(input[index], shift)) {
            auto old = atomic_fetch_add_explicit(&histogram[bucket], 1, memory_order_relaxed);
            output[old] = input[index];
        }
    }
}

[[kernel]]
void shuffle3(
    uint thread_position_in_grid [[thread_position_in_grid]],
    uint threads_per_grid [[threads_per_grid]],
    device atomic_uint *histogram [[buffer(0)]],
    device uint *input [[buffer(1)]],
    device uint *output [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    constant uint &shift [[buffer(4)]]
)
{
    const uint global_id = thread_position_in_grid;

    if (global_id < count) {
        const uchar bucket = key(input[global_id], shift);
        const auto old = atomic_fetch_add_explicit(&histogram[bucket], 1, memory_order_relaxed);
        output[old] = input[global_id];
    }
}
