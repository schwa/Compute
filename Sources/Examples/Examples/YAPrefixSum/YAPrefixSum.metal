#include <metal_stdlib>
#include <metal_logging>

using namespace metal;

namespace YAPrefixSum {

    kernel void prefix_sum_small_slow(
        constant uint *input [[buffer(0)]],
        constant uint &count [[buffer(1)]],
        device uint *output [[buffer(2)]]
    ) {
        for (uint n = 1; n != count; ++n) {
            output[n] = input[n - 1] + output[n - 1];
        }
    }

}
