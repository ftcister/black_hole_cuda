#include "types.h"
#include <math.h>

__host__ __device__ vec2_t vec2_add(vec2_t v1, vec2_t v2) {
    return (vec2_t) {
        .x = v1.x + v2.x,
        .y = v1.y + v2.y
    };
}

__host__ __device__ vec2_t vec2_sub(vec2_t v1, vec2_t v2) {
    return (vec2_t) {
        .x = v1.x - v2.x,
        .y = v1.y - v2.y
    };
}

__host__ __device__ vec2_t vec2_mul(vec2_t v, float s) {
    return (vec2_t) {
        .x = v.x * s,
        .y = v.y * s
    };
}

__host__ __device__ vec2_t vec2_div(vec2_t v, float s) {
    return (vec2_t) {
        .x = v.x / s,
        .y = v.y / s
    };
}

__host__ __device__ float vec2_dot(vec2_t v1, vec2_t v2) {
    return v1.x * v2.x + v1.y * v2.y;
}

__host__ __device__ float vec2_len(vec2_t v) {
    return sqrtf(vec2_dot(v, v));
}

__host__ __device__ vec2_t vec2_normalize(vec2_t v) {
    return vec2_div(v, vec2_len(v));
}
