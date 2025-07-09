#include <metal_stdlib>
using namespace metal;

struct PrefixSumUniforms {
    uint threads_per_workgroup;
    uint items_per_workgroup;
    uint element_count;
};

kernel void reduce_downsweep(
    device uint* items [[buffer(0)]],
    device uint* blockSums [[buffer(1)]],
    constant PrefixSumUniforms& uniforms [[buffer(2)]],
    uint3 workgroup_id [[threadgroup_position_in_grid]],
    uint3 num_workgroups [[threadgroups_per_grid]],
    uint thread_index_in_threadgroup [[thread_index_in_threadgroup]]
) {
    const uint THREADS_PER_WORKGROUP = uniforms.threads_per_workgroup;
    const uint ITEMS_PER_WORKGROUP = uniforms.items_per_workgroup;
    const uint ELEMENT_COUNT = uniforms.element_count;
    
    // Threadgroup shared memory - dynamically sized based on items per workgroup
    threadgroup uint temp[1024]; // Support up to 512 threads per workgroup (1024 items)
    
    const uint WORKGROUP_ID = workgroup_id.x + workgroup_id.y * num_workgroups.x;
    const uint WID = WORKGROUP_ID * THREADS_PER_WORKGROUP;
    const uint GID = WID + thread_index_in_threadgroup;
    const uint TID = thread_index_in_threadgroup;
    
    const uint ELM_TID = TID * 2;
    const uint ELM_GID = GID * 2;
    
    // Load input to shared memory
    temp[ELM_TID] = (ELM_GID >= ELEMENT_COUNT) ? 0 : items[ELM_GID];
    temp[ELM_TID + 1] = (ELM_GID + 1 >= ELEMENT_COUNT) ? 0 : items[ELM_GID + 1];
    
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
    if (ELM_GID >= ELEMENT_COUNT) {
        return;
    }
    items[ELM_GID] = temp[ELM_TID];
    
    if (ELM_GID + 1 >= ELEMENT_COUNT) {
        return;
    }
    items[ELM_GID + 1] = temp[ELM_TID + 1];
}

kernel void add_block_sums(
    device uint* items [[buffer(0)]],
    device uint* blockSums [[buffer(1)]],
    constant PrefixSumUniforms& uniforms [[buffer(2)]],
    uint3 workgroup_id [[threadgroup_position_in_grid]],
    uint3 num_workgroups [[threadgroups_per_grid]],
    uint thread_index_in_threadgroup [[thread_index_in_threadgroup]]
) {
    const uint THREADS_PER_WORKGROUP = uniforms.threads_per_workgroup;
    const uint ELEMENT_COUNT = uniforms.element_count;
    
    const uint WORKGROUP_ID = workgroup_id.x + workgroup_id.y * num_workgroups.x;
    const uint WID = WORKGROUP_ID * THREADS_PER_WORKGROUP;
    const uint GID = WID + thread_index_in_threadgroup;
    
    const uint ELM_ID = GID * 2;
    
    if (ELM_ID >= ELEMENT_COUNT) {
        return;
    }
    
    const uint blockSum = blockSums[WORKGROUP_ID];
    
    items[ELM_ID] += blockSum;
    
    if (ELM_ID + 1 >= ELEMENT_COUNT) {
        return;
    }
    
    items[ELM_ID + 1] += blockSum;
}
