#include <metal_stdlib>
#include <metal_logging>

using namespace metal;

namespace CountingSort {

    const uint thread_position_in_grid [[thread_position_in_grid]];
    const uint thread_position_in_threadgroup [[thread_position_in_threadgroup]];
    const uint threads_per_simdgroup [[threads_per_simdgroup]];

    // M1 Ultra @ 1.5m input = 202.01 us
    kernel void histogram1(
        constant uint *input [[buffer(0)]],
        constant uint &count [[buffer(1)]],
        constant uint &shift [[buffer(2)]],
        device atomic_uint *output [[buffer(3)]]
   ) {
        const uint value = input[thread_position_in_grid];
        const uchar bucket = (value >> shift) & 0xFF;
        atomic_fetch_add_explicit(&output[bucket], 1, memory_order_relaxed);
   }

    // M1 Ultra @ 1.5m input = 276.87 us
    kernel void histogram2(
        constant uint *input [[buffer(0)]],
        constant uint &count [[buffer(1)]],
        constant uint &shift [[buffer(2)]],
        device atomic_uint *output [[buffer(3)]]
    ) {
        threadgroup atomic_uint temp[256];

        // < M3 does not seem to clear threadgroup memory. Manually clear it.
        if (thread_position_in_threadgroup == 0) {
           for (uint bucket = 0; bucket != 256; ++bucket) {
               atomic_store_explicit(&temp[bucket], 0, memory_order_relaxed);
           }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint value = input[thread_position_in_grid];
        const uchar bucket = (value >> shift) & 0xFF;
        atomic_fetch_add_explicit(&temp[bucket], 1, memory_order_relaxed);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (thread_position_in_threadgroup == 0) {
           for (uint bucket = 0; bucket != 256; ++bucket) {
               const uint count = atomic_load_explicit(&temp[bucket], memory_order_relaxed);
               atomic_fetch_add_explicit(&output[bucket], count, memory_order_relaxed);
           }
        }
    }

    // M1 Ultra @ 1.5m input = 38.70 us
    kernel void histogram(
        constant uint *input [[buffer(0)]],
        constant uint &count [[buffer(1)]],
        constant uint &shift [[buffer(2)]],
        device atomic_uint *output [[buffer(3)]]
    ) {
        threadgroup atomic_uint temp[256];

        // < M3 does not seem to clear threadgroup memory. Manually clear it.
        if (thread_position_in_threadgroup == 0) {
           for (uint bucket = 0; bucket != 256; ++bucket) {
               atomic_store_explicit(&temp[bucket], 0, memory_order_relaxed);
           }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint value = input[thread_position_in_grid];
        const uchar bucket = (value >> shift) & 0xFF;
        atomic_fetch_add_explicit(&temp[bucket], 1, memory_order_relaxed);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (thread_position_in_threadgroup < 256) {
           const uchar bucket = thread_position_in_threadgroup;
           const uint count = atomic_load_explicit(&temp[bucket], memory_order_relaxed);
           atomic_fetch_add_explicit(&output[bucket], count, memory_order_relaxed);
        }
    }

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

        for (int index = count - 1; index >= 0; --index) {
            const uint value = input[index];
            const uchar bucket = (value >> shift) & 0xFF;
            histogram[bucket] -= 1;
            const uint newIndex = histogram[bucket];
            output[newIndex] = value;
//            os_log_default.log("%d %d %d", index, newIndex, value);
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

//        for (int index = count - 1; index >= 0; --index) {
//            const uint value = input[index];
//            const uchar bucket = (value >> shift) & 0xFF;
//            histogram[bucket] -= 1;
//            const uint newIndex = histogram[bucket];
//            output[newIndex] = value;
//        }
    }
}
