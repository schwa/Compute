#include <metal_stdlib>
#include <simd/simd.h>
#include <metal_logging>

using namespace metal;

namespace CountingSort16 {
    // MARK: -

    [[kernel]]
    void histogram(
       uint2 thread_position_in_grid [[thread_position_in_grid]],
       device short *input [[buffer(0)]],
       device atomic_uint *histogram [[buffer(1)]],
        constant uint &histogramCount [[buffer(2)]]

       )
    {
        const uchar bucket = thread_position_in_grid.x;
        if (bucket >= histogramCount) {
            return;
        }
        const uint index = thread_position_in_grid.y;
        if (bucket == input[index]) {
            atomic_fetch_add_explicit(&histogram[bucket], 1, memory_order_relaxed);
        }
    }

    // MARK: -

    [[kernel]]
    void prefix_sum_inclusive(
        device uint *histogram [[buffer(0)]],
        constant uint &histogramCount [[buffer(1)]]
        )
    {
        for (uint index = 1; index != histogramCount; index++) {
            histogram[index] += histogram[index - 1];
        }
    }

    [[kernel]]
    void prefix_sum_exclusive(
        device uint *histogram [[buffer(0)]],
        constant uint &histogramCount [[buffer(1)]]
    )
    {
        uint sum = 0;
        for (uint i = 0; i != histogramCount; i++) {
            uint t = histogram[i];
            histogram[i] = sum;
            sum += t;
        }
    }

    // MARK: -

    [[kernel]]
    void shuffle(
         uint thread_position_in_grid [[thread_position_in_grid]],
         device atomic_uint *histogram [[buffer(0)]],
         device short *input [[buffer(1)]],
         device short *output [[buffer(2)]],
         constant uint &count [[buffer(3)]]
         )
    {
        const uint index = thread_position_in_grid;

//        for (uint index = 0; index != count; ++index) {
            const auto bucket = input[index];
            const auto old = atomic_fetch_add_explicit(&histogram[bucket], 1, memory_order_relaxed);
            output[old] = input[index];
//        }
    }

}
