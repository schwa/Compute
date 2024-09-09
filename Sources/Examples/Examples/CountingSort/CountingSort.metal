#include <metal_stdlib>
#include <metal_logging>

using namespace metal;

namespace CountingSort {

    const uint thread_position_in_grid [[thread_position_in_grid]];
    const uint thread_position_in_threadgroup [[thread_position_in_threadgroup]];
    const uint threads_per_simdgroup [[threads_per_simdgroup]];



    // MARK: -

    // VERY VERY SLOW
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
            auto key = (value >> shift) & 0xFF;
            auto outputIndex = histogram[key]++;
            output[outputIndex] = input[i];
        }
    }

    // BROKEN:
    kernel void shuffle2(
        device uint *input [[buffer(0)]],
        constant uint &count [[buffer(1)]],
        device uint *output [[buffer(2)]],
        device atomic_uint *histogram [[buffer(3)]],
        constant uint &shift [[buffer(4)]]
    ) {
        const uint i = thread_position_in_grid;
        auto value = input[i];
        auto key = (value >> shift) & 0xFF;
        auto outputIndex = atomic_fetch_add_explicit(&histogram[key], 1, memory_order_relaxed);
        output[outputIndex] = input[i];
    }

    // VERY SLOW
    kernel void shuffle3(
        device uint *input [[buffer(0)]],
        constant uint &count [[buffer(1)]],
        device uint *output [[buffer(2)]],
        device atomic_uint *histogram [[buffer(3)]],
        constant uint &shift [[buffer(4)]]
    ) {
        if (thread_position_in_grid >= 256) {
            return;
        }
        for (uint i = 0; i != count; ++i) {
            auto value = input[i];
            auto key = (value >> shift) & 0xFF;
            if (key == thread_position_in_grid) {
                auto outputIndex = atomic_fetch_add_explicit(&histogram[key], 1, memory_order_relaxed);
                output[outputIndex] = input[i];
            }
        }
    }

    kernel void shuffle4(
        device uint *input [[buffer(0)]],
        constant uint &count [[buffer(1)]],
        device uint *output [[buffer(2)]],
        device uint *histogram [[buffer(3)]],
        constant uint &shift [[buffer(4)]]
    ) {
        threadgroup atomic_uint threadgroupHistogram[256];

        if (thread_position_in_grid == 0) {
            for (uint i = 0; i != 256; ++i) {
                atomic_store_explicit(&threadgroupHistogram[i], histogram[i], memory_order_relaxed);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (thread_position_in_grid >= 256) {
            return;
        }
        for (uint i = 0; i != count; ++i) {
            auto value = input[i];
            auto key = (value >> shift) & 0xFF;
            if (key == thread_position_in_grid) {
                auto outputIndex = atomic_fetch_add_explicit(&threadgroupHistogram[key], 1, memory_order_relaxed);
                output[outputIndex] = input[i];
            }
        }
    }

    kernel void shuffle5(
        device uint *input [[buffer(0)]],
        constant uint &count [[buffer(1)]],
        device uint *output [[buffer(2)]],
        device uint *histogram [[buffer(3)]],
        constant uint &shift [[buffer(4)]]
    ) {
        if (thread_position_in_grid >= 256) {
            return;
        }

        threadgroup uint h;
        h = histogram[thread_position_in_grid];

        for (uint i = 0; i != count; ++i) {
            auto value = input[i];
            auto key = (value >> shift) & 0xFF;
            if (key == thread_position_in_grid) {
                auto outputIndex = h++;
                output[outputIndex] = input[i];
            }
        }
    }

    kernel void shuffle6(
        device uint *input [[buffer(0)]],
        constant uint &count [[buffer(1)]],
        device uint *output [[buffer(2)]],
        device uint *histogram [[buffer(3)]],
        constant uint &shift [[buffer(4)]]
     ) {
     }
}
