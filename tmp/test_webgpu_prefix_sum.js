// Test the WebGPU prefix sum implementation with Node.js
import { create } from 'webgpu';

// Get the prefix sum shader source
const prefixSumSource = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> items: array<u32>;
@group(0) @binding(1) var<storage, read_write> blockSums: array<u32>;

override WORKGROUP_SIZE_X: u32;
override WORKGROUP_SIZE_Y: u32;
override THREADS_PER_WORKGROUP: u32;
override ITEMS_PER_WORKGROUP: u32;
override ELEMENT_COUNT: u32;

var<workgroup> temp: array<u32, ITEMS_PER_WORKGROUP>;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn reduce_downsweep(
    @builtin(workgroup_id) w_id: vec3<u32>,
    @builtin(num_workgroups) w_dim: vec3<u32>,
    @builtin(local_invocation_index) TID: u32,
) {
    let WORKGROUP_ID = w_id.x + w_id.y * w_dim.x;
    let WID = WORKGROUP_ID * THREADS_PER_WORKGROUP;
    let GID = WID + TID;
    
    let ELM_TID = TID * 2;
    let ELM_GID = GID * 2;
    
    // Load input to shared memory
    temp[ELM_TID]     = select(items[ELM_GID], 0, ELM_GID >= ELEMENT_COUNT);
    temp[ELM_TID + 1] = select(items[ELM_GID + 1], 0, ELM_GID + 1 >= ELEMENT_COUNT);

    var offset: u32 = 1;

    // Up-sweep (reduce) phase
    for (var d: u32 = ITEMS_PER_WORKGROUP >> 1; d > 0; d >>= 1) {
        workgroupBarrier();

        if (TID < d) {
            var ai: u32 = offset * (ELM_TID + 1) - 1;
            var bi: u32 = offset * (ELM_TID + 2) - 1;
            temp[bi] += temp[ai];
        }

        offset *= 2;
    }

    // Save workgroup sum and clear last element
    if (TID == 0) {
        let last_offset = ITEMS_PER_WORKGROUP - 1;

        blockSums[WORKGROUP_ID] = temp[last_offset];
        temp[last_offset] = 0;
    }

    // Down-sweep phase
    for (var d: u32 = 1; d < ITEMS_PER_WORKGROUP; d *= 2) {
        offset >>= 1;
        workgroupBarrier();

        if (TID < d) {
            var ai: u32 = offset * (ELM_TID + 1) - 1;
            var bi: u32 = offset * (ELM_TID + 2) - 1;

            let t: u32 = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    workgroupBarrier();

    // Copy result from shared memory to global memory
    if (ELM_GID >= ELEMENT_COUNT) {
        return;
    }
    items[ELM_GID] = temp[ELM_TID];

    if (ELM_GID + 1 >= ELEMENT_COUNT) {
        return;
    }
    items[ELM_GID + 1] = temp[ELM_TID + 1];
}`;

async function testWebGPUPrefixSum() {
    const gpu = create();
    const adapter = await gpu.requestAdapter();
    const device = await adapter.requestDevice();
    
    console.log('WebGPU device created successfully');
    
    // Test with 513 elements
    const input = new Uint32Array(Array.from({length: 513}, (_, i) => i));
    
    const inputBuffer = device.createBuffer({
        size: input.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });
    
    device.queue.writeBuffer(inputBuffer, 0, input);
    
    const workgroupSize = { x: 16, y: 16 };
    const threadsPerWorkgroup = workgroupSize.x * workgroupSize.y;
    const itemsPerWorkgroup = 2 * threadsPerWorkgroup;
    
    const workgroupCount = Math.ceil(513 / itemsPerWorkgroup);
    const blockSumBuffer = device.createBuffer({
        size: workgroupCount * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });
    
    const shaderModule = device.createShaderModule({
        code: prefixSumSource
    });
    
    const bindGroupLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'storage' }
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'storage' }
            }
        ]
    });
    
    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: inputBuffer } },
            { binding: 1, resource: { buffer: blockSumBuffer } }
        ]
    });
    
    const pipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout]
        }),
        compute: {
            module: shaderModule,
            entryPoint: 'reduce_downsweep',
            constants: {
                'WORKGROUP_SIZE_X': workgroupSize.x,
                'WORKGROUP_SIZE_Y': workgroupSize.y,
                'THREADS_PER_WORKGROUP': threadsPerWorkgroup,
                'ITEMS_PER_WORKGROUP': itemsPerWorkgroup,
                'ELEMENT_COUNT': 513,
            }
        }
    });
    
    const commandEncoder = device.createCommandEncoder();
    const computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(pipeline);
    computePass.setBindGroup(0, bindGroup);
    computePass.dispatchWorkgroups(workgroupCount);
    computePass.end();
    
    const readBuffer = device.createBuffer({
        size: input.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    
    commandEncoder.copyBufferToBuffer(inputBuffer, 0, readBuffer, 0, input.byteLength);
    device.queue.submit([commandEncoder.finish()]);
    
    await readBuffer.mapAsync(GPUMapMode.READ);
    const result = new Uint32Array(readBuffer.getMappedRange().slice());
    readBuffer.unmap();
    
    console.log('WebGPU result:', result.slice(0, 10), '...', result.slice(-10));
    console.log('Last element:', result[512]);
    
    // Expected result
    const expected = [0];
    for (let i = 0; i < 512; i++) {
        expected.push(expected[expected.length - 1] + i);
    }
    
    console.log('Expected last element:', expected[512]);
    console.log('Match:', result[512] === expected[512]);
    
    device.destroy();
}

testWebGPUPrefixSum().catch(console.error);