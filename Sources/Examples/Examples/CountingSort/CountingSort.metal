#include <metal_stdlib>
#include <metal_logging>

using namespace metal;

namespace CountingSort {

    const uint thread_position_in_grid [[thread_position_in_grid]];
    const uint thread_position_in_threadgroup [[thread_position_in_threadgroup]];
    const uint threads_per_simdgroup [[threads_per_simdgroup]];



    // MARK: -

    kernel void shuffle1(
        device uint *input [[buffer(0)]],
        constant uint &count [[buffer(1)]],
        device uint *output [[buffer(2)]],
        device uint *histogram [[buffer(3)]],
        constant uint &shift [[buffer(4)]]
    ) {
        if (thread_position_in_grid != 0) {
            return;
        }

        for (uint i = 0; i != count; ++i) {
            auto value = input[i];
            auto key =  (value >> shift) & 0xFF;
            auto outputIndex = histogram[key]++;
            output[outputIndex] = input[i];
        }
    }


    kernel void shuffle(
        device uint *input [[buffer(0)]],
        constant uint &count [[buffer(1)]],
        device uint *output [[buffer(2)]],
        device atomic_uint *histogram [[buffer(3)]],
        constant uint &shift [[buffer(4)]]
    ) {
        const uint index = count - thread_position_in_grid - 1;

        const uint value = input[index];
        const uchar bucket = (value >> shift) & 0xFF;

        const uint newIndex = atomic_fetch_add_explicit(&histogram[bucket], -1, memory_order_relaxed) - 1;
        output[newIndex] = value;
    }
}
