#ifndef TYPES_H
#define TYPES_H

#include <stdbool.h>
#include <stdint.h>

#define WINDOW_WIDTH 1600
#define WINDOW_HEIGHT 900

#define MAX_PARTICLE_COUNT 1000000
#define MAX_BLACK_HOLE_COUNT 1000

#define BACKGROUND_COLOR pixel_t { 0x80, 0x80, 0x80, 0xff }

typedef struct pixel_t
{
    uint8_t b, g, r, a;
} pixel_t;

typedef struct vec2_t {
    float x, y;
} vec2_t;

typedef struct particle_t {
    vec2_t position;
    vec2_t velocity;
    pixel_t color;
    bool   deleted;
} particle_t;

int rand_int(int min, int max);

// vector operations
__host__ __device__ vec2_t vec2_add(vec2_t v1, vec2_t v2);
__host__ __device__ vec2_t vec2_sub(vec2_t v1, vec2_t v2);
__host__ __device__ vec2_t vec2_mul(vec2_t v, float s);
__host__ __device__ vec2_t vec2_div(vec2_t v, float s);
__host__ __device__ float vec2_dot(vec2_t v1, vec2_t v2);
__host__ __device__ float vec2_len(vec2_t v);
__host__ __device__ vec2_t vec2_normalize(vec2_t v);


#endif // TYPES_H
