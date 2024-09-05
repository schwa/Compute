#include <metal_stdlib>
#include <metal_logging>

using namespace metal;

namespace YAPrefixSum {

    const uint thread_position_in_grid [[thread_position_in_grid]];
    const uint threads_per_simdgroup [[threads_per_simdgroup]];
    const uint threadgroup_position_in_grid [[threadgroup_position_in_grid]];
    const uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]];

    void log_if_outside(uint value, uint lower, uint upper) {
        if (value < lower || value >= upper) {
            os_log_default.log("Value (%d) is out of range (%d..<%d)", value, lower, upper);
        }
    }

    // MARK: -

    // Very inefficient. Do not use. Does work with any size input however. But is not at all parallel. Beware.
    kernel void prefix_sum_slow(
        constant uint *input [[buffer(0)]],
        constant uint &count [[buffer(1)]],
        device uint *output [[buffer(2)]]
    ) {
        for (uint n = 1; n != count; ++n) {
            output[n] = input[n - 1] + output[n - 1];
        }
    }

    // Performs a parallel prefix sum (scan) operation using SIMD instructions.
    // 1. SIMD-level prefix sum within each SIMD group
    // 2. Threadgroup-level prefix sum to combine SIMD group results
    // Only computes the prefix sum for a single threadgroup.

    kernel void prefix_sum_simd(
        constant uint *input [[buffer(0)]],
        constant uint &count [[buffer(1)]],
        device uint *output [[buffer(2)]],
        threadgroup uint *totals [[threadgroup(0)]],
        threadgroup uint *offsets [[threadgroup(1)]],
        constant uint &totals_count [[buffer(3)]]
    ) {
        // Calculate the position of this SIMD group within the grid
        const uint simdgroup_position_in_grid = thread_position_in_grid / threads_per_simdgroup;

        // Early exit if this thread is beyond the input size
        if (thread_position_in_grid >= count) {
            return;
        }

        // Perform an exclusive prefix sum within each SIMD group
        output[thread_position_in_grid] = simd_prefix_exclusive_sum(input[thread_position_in_grid]);

        if (count <= threads_per_simdgroup) {
            return;
        }

        // Ensure all threads in the SIMD group have completed their work
        // TODO: is simdgroup_barrier necessary if we just called simd_prefix_exclusive_sum?
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // The first thread in each SIMD group computes the total sum for that group.
        if (simd_is_first()) {
            // Index of the last element in this SIMD group
            const uint total_index = min(simdgroup_position_in_grid * threads_per_simdgroup + threads_per_simdgroup - 1, count - 1);
            // Compute total sum for this SIMD group
            const uint total = output[total_index] + input[total_index];
            totals[simdgroup_position_in_grid] = total;
        }

        // Synchronize all threads in the threadgroup. TODO: Code seems to work without this :-)
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Perform a prefix sum on the SIMD group totals to calculate offsets...
        if (threadgroup_position_in_grid == 0) {
            if (thread_position_in_grid < totals_count) {
                offsets[thread_position_in_grid] = simd_prefix_exclusive_sum(totals[thread_position_in_grid]);
            }
        }

        // Ensure the sequential prefix sum is complete TODO: Code seems to work without this :-)
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Apply the computed offsets to each thread's result
        output[thread_position_in_grid] += offsets[simdgroup_position_in_grid];
    }

    // MARK: -

    // Applies offsets to groups of input elements.
    // This kernel adds an offset to each input element. The offset for each element is determined by its position within a group of size `group_size`. Each group shares a single offset from the `offsets` array.
    kernel void offset(
        device uint *input [[buffer(0)]],
        constant uint &count [[buffer(1)]],
        constant uint *offsets [[buffer(2)]],
        constant uint &group_size [[buffer(3)]]
    ) {
        if (thread_position_in_grid >= count) {
            return;
        }
        input[thread_position_in_grid] += offsets[thread_position_in_grid / group_size];
    }
}
