#include <metal_stdlib>
using namespace metal;

uint3 w_id [[threadgroup_position_in_grid]];
uint3 w_dim [[threadgroups_per_grid]];
uint TID [[thread_index_in_threadgroup]];

kernel void reduce_downsweep(
    device uint* items [[buffer(0)]],
    device uint* blockSums [[buffer(1)]],
    constant uint& WORKGROUP_SIZE_X [[buffer(2)]],
    constant uint& WORKGROUP_SIZE_Y [[buffer(3)]],
    constant uint& THREADS_PER_WORKGROUP [[buffer(4)]],
    constant uint& ITEMS_PER_WORKGROUP [[buffer(5)]],
    constant uint& ELEMENT_COUNT [[buffer(6)]],
    threadgroup uint *temp[[threadgroup(0)]] // threadgroup uint temp[ITEMS_PER_WORKGROUP * 2];
    )
{
    const uint WORKGROUP_ID = w_id.x + w_id.y * w_dim.x;
    const uint WID = WORKGROUP_ID * THREADS_PER_WORKGROUP;
    const uint GID = WID + TID; // Global thread ID

    const uint ELM_TID = TID * 2; // Element pair local ID
    const uint ELM_GID = GID * 2; // Element pair global ID

    // Load input to shared memory
    temp[ELM_TID] = (ELM_GID < ELEMENT_COUNT) ? items[ELM_GID] : 0;
    temp[ELM_TID + 1] = (ELM_GID + 1 < ELEMENT_COUNT) ? items[ELM_GID + 1] : 0;

    uint offset = 1;

    // Up-sweep (reduce) phase
    for (uint d = ITEMS_PER_WORKGROUP >> 1; d > 0; d >>= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (TID < d) {
            uint ai = offset * (ELM_TID + 1) - 1;
            uint bi = offset * (ELM_TID + 2) - 1;
            temp[bi] += temp[ai];
        }

        offset *= 2;
    }

    // Save workgroup sum and clear last element
    if (TID == 0) {
        const uint last_offset = ITEMS_PER_WORKGROUP - 1;
        blockSums[WORKGROUP_ID] = temp[last_offset];
        temp[last_offset] = 0;
    }

    // Down-sweep phase
    for (uint d = 1; d < ITEMS_PER_WORKGROUP; d *= 2) {
        offset >>= 1;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (TID < d) {
            uint ai = offset * (ELM_TID + 1) - 1;
            uint bi = offset * (ELM_TID + 2) - 1;

            uint t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Copy result from shared memory to global memory
    if (ELM_GID < ELEMENT_COUNT) {
        items[ELM_GID] = temp[ELM_TID];
    }

    if (ELM_GID + 1 < ELEMENT_COUNT) {
        items[ELM_GID + 1] = temp[ELM_TID + 1];
    }
}

kernel void add_block_sums(
    device uint* items [[buffer(0)]],
    device uint* blockSums [[buffer(1)]],
    constant uint& WORKGROUP_SIZE_X [[buffer(2)]],
    constant uint& WORKGROUP_SIZE_Y [[buffer(3)]],
    constant uint& THREADS_PER_WORKGROUP [[buffer(4)]],
    constant uint& ITEMS_PER_WORKGROUP [[buffer(5)]],
    constant uint& ELEMENT_COUNT [[buffer(6)]]
    )
{
    uint WORKGROUP_ID = w_id.x + w_id.y * w_dim.x;
    uint WID = WORKGROUP_ID * THREADS_PER_WORKGROUP;
    uint GID = WID + TID; // Global thread ID
    uint ELM_ID = GID * 2;

    if (ELM_ID >= ELEMENT_COUNT) {
        return;
    }

    uint blockSum = blockSums[WORKGROUP_ID];
    items[ELM_ID] += blockSum;

    if (ELM_ID + 1 >= ELEMENT_COUNT) {
        return;
    }

    items[ELM_ID + 1] += blockSum;
}
