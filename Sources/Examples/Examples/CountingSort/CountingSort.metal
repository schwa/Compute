#include <metal_stdlib>
#include <metal_logging>

using namespace metal;

kernel void hello_world() {
    os_log_default.log("Hello world (from Metal!)");
}
