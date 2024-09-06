#include <metal_stdlib>
#include <metal_logging>

using namespace metal;

namespace YAPrefixSum {

    const uint thread_position_in_grid [[thread_position_in_grid]];
    const uint threads_per_simdgroup [[threads_per_simdgroup]];
    const uint threadgroup_position_in_grid [[threadgroup_position_in_grid]];
    const uint thread_position_in_threadgroup [[thread_position_in_threadgroup]];

    constant bool prefix_sum_inclusive [[function_constant(12345)]];

    uint simd_prefix_sum(uint n) {
        if (prefix_sum_inclusive) {
            return simd_prefix_inclusive_sum(n);
        }
        else {
            return simd_prefix_exclusive_sum(n);
        }
    }

    // MARK: -

    // Very inefficient. Do not use. Does work with any size input however. But is not at all parallel. Beware.
    kernel void prefix_sum_exclusive_slow(
        constant uint *inputs [[buffer(0)]],
        constant uint &count [[buffer(1)]],
        device uint *outputs [[buffer(2)]]
    ) {
        for (uint n = 1; n != count; ++n) {
            outputs[n] = inputs[n - 1] + outputs[n - 1];
        }
    }

    kernel void prefix_sum_inclusive_slow(
        constant uint *inputs [[buffer(0)]],
        constant uint &count [[buffer(1)]],
        device uint *outputs [[buffer(2)]]
    ) {
        outputs[0] = inputs[0];
        for (uint n = 1; n != count; ++n) {
            outputs[n] = inputs[n] + outputs[n - 1];
        }
    }

    // Performs a parallel prefix sum (scan) operation using SIMD instructions.
    // 1. SIMD-level prefix sum within each SIMD group
    // 2. Threadgroup-level prefix sum to combine SIMD group results
    // Only computes the prefix sum for a single threadgroup.
    kernel void prefix_sum_simd(
        constant uint *inputs [[buffer(0)]],
        constant uint &count [[buffer(1)]],
        device uint *outputs [[buffer(2)]],
        device uint *totals [[buffer(3)]],
        device uint *offsets [[buffer(4)]]
    ) {
        // Calculate the position of this SIMD group within the grid
        const uint simdgroup_position_in_grid = thread_position_in_grid / threads_per_simdgroup;

        // Early exit if this thread is beyond the input size
        if (thread_position_in_grid >= count) {
            return;
        }

        // Perform an exclusive prefix sum within each SIMD group
        const uint t = simd_prefix_sum(inputs[thread_position_in_grid]);
        outputs[thread_position_in_grid] = t;

        if (count <= threads_per_simdgroup) {
            return;
        }

        // Ensure all threads in the SIMD group have completed their work
        // TODO: is simdgroup_barrier necessary if we just called simd_prefix_exclusive_sum?
//        simdgroup_barrier(mem_flags::mem_threadgroup);

        // The first thread in each SIMD group computes the total sum for that group.
        if (simd_is_first()) {
            // Index of the last element in this SIMD group
            const uint total_index = min(simdgroup_position_in_grid * threads_per_simdgroup + threads_per_simdgroup - 1, count - 1);
            // Compute total sum for this SIMD group
            const uint total = outputs[total_index] + inputs[total_index];
            totals[simdgroup_position_in_grid] = total;
        }

        // Synchronize all threads in the threadgroup.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Perform a prefix sum on the SIMD group totals to calculate offsets...
        if (thread_position_in_threadgroup < threads_per_simdgroup) {
            const uint totals_index = threadgroup_position_in_grid * threads_per_simdgroup + thread_position_in_threadgroup;
            const uint offset = simd_prefix_exclusive_sum(totals[totals_index]);
            offsets[totals_index] = offset;
        }

        // Ensure the sequential prefix sum is complete
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Apply the computed offsets to each thread's result]
        outputs[thread_position_in_grid] += offsets[simdgroup_position_in_grid];
    }

    // MARK: -

    kernel void gather_totals(
        constant uint *inputs [[buffer(0)]],
        device uint *outputs [[buffer(1)]],
        constant uint &chunk_size [[buffer(2)]],
        device uint *totals [[buffer(3)]]
        )
    {
        const uint input_index = thread_position_in_grid * chunk_size + chunk_size - 1;
        const uint total = inputs[input_index] + outputs[input_index];
        totals[thread_position_in_grid] = total;
    }

    kernel void apply_offsets(
        device uint *outputs [[buffer(0)]],
        constant uint &chunk_size [[buffer(1)]],
        device uint *offsets [[buffer(2)]]
    )
    {
        const uint offset_index = thread_position_in_grid / chunk_size;
        const uint offset = offsets[offset_index];
        const uint output = outputs[thread_position_in_grid];
        outputs[thread_position_in_grid] = output + offset;
    }
}
