#include <SDL2/SDL.h>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <curand_kernel.h>

#include "types.h"

struct particle_soa_t {
    float*   position_xs;
    float*   position_ys;
    float*   velocity_xs;
    float*   velocity_ys;
    pixel_t* colors;
    bool*    deleted;
};

void Write_times(const char *filename, float *draw, float *update, float *generate, float *frame) {
  FILE *fp;
  fp = fopen(filename, "w");
  fprintf(fp, "iteration,draw,update,generate,frame\n");
  for (int i = 0; i < 1000; i++)
    fprintf(fp, "%d,%f,%f,%f,%f\n", i, draw[i], update[i], generate[i], frame[i]);
  fclose(fp);
}

typedef enum mouse_button_t {
    MOUSE_BUTTON_LEFT = SDL_BUTTON_LEFT,
    MOUSE_BUTTON_MIDDLE = SDL_BUTTON_MIDDLE,
    MOUSE_BUTTON_RIGHT = SDL_BUTTON_RIGHT,

    MOUSE_BUTTON_COUNT,
} mouse_button_t;

__global__ void random_init(curandState* state) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  curand_init(0, tid, 0, &state[tid]);
}

__global__ void generate_particles_cuda(int particle_count, particle_soa_t particles, curandState *state){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < particle_count) {
        particles.position_xs[tid] = ((curand_uniform(&state[tid]) - 1e-14f) * (WINDOW_WIDTH + 1));
        particles.position_ys[tid] = ((curand_uniform(&state[tid]) - 1e-14f) * (WINDOW_HEIGHT + 1));
        particles.velocity_xs[tid] = (curand_uniform(&state[tid]) - 1e-14f) * (15 - (-15) + 1);
        particles.velocity_ys[tid] = (curand_uniform(&state[tid]) - 1e-14f) * (15 - (-15) + 1);
        particles.colors[tid].r = ((curand_uniform(&state[tid]) - 1e-14f) * (255 + 1));
        particles.colors[tid].g = ((curand_uniform(&state[tid]) - 1e-14f) * (255 + 1));
        particles.colors[tid].b = ((curand_uniform(&state[tid]) - 1e-14f) * (255 + 1));
        particles.colors[tid].a = 0xff;
        particles.deleted[tid] = false;
    }
}

__global__ void generate_black_holes_cuda(int black_hole_count, particle_soa_t black_holes, curandState *state){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < black_hole_count) {
        black_holes.position_xs[tid] = ((curand_uniform(&state[tid]) - 1e-14f) * (WINDOW_WIDTH + 1));
        black_holes.position_ys[tid] = ((curand_uniform(&state[tid]) - 1e-14f) * (WINDOW_HEIGHT + 1));
    }
}

__global__ void draw_background(pixel_t* pixels) {
    // set the background color
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < WINDOW_WIDTH * WINDOW_HEIGHT) {
        pixels[tid] = BACKGROUND_COLOR;
    }
}

__global__ void fast_draw(pixel_t* pixels,
                          particle_soa_t particles, int particle_count,
                          particle_soa_t black_holes, int black_hole_count) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    // draw the particles
    if (tid < particle_count) {
        bool deleted = particles.deleted[tid];
        if (!deleted) {
            int particle_x = particles.position_xs[tid];
            int particle_y = particles.position_ys[tid];
            pixel_t color = particles.colors[tid];
            for (int i = 0; i < 4; i++) {
                int pixel_x = i + particle_x;
                for (int j = 0; j < 4; j++) {
                    int pixel_y = j + particle_y;
                    pixels[pixel_y * WINDOW_WIDTH + pixel_x] = color;
                }
            }

        }

    }

    // draw the black holes
    if (tid < black_hole_count) {
        int black_hole_x = black_holes.position_xs[tid];
        int black_hole_y = black_holes.position_ys[tid];

        for (int i = 0; i < 10; i++) {
            int pixel_x = i + black_hole_x;
            for (int j = 0; j < 10; j++) {
                int pixel_y = j + black_hole_y;
                pixels[pixel_y * WINDOW_WIDTH + pixel_x] = pixel_t { 0, 0, 0, 0xff };
            }
        }
    }
}

__global__ void update_particles_cuda(particle_soa_t particles, size_t particle_count, particle_soa_t black_holes, size_t black_hole_count, float dt){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < particle_count) {
        // particle_t* particle = &particles[tid];
        bool deleted = particles.deleted[tid];

        if (!deleted) {
            float pos_x  = particles.position_xs[tid];
            float pos_y  = particles.position_ys[tid];
            float vel_x  = particles.velocity_xs[tid];
            float vel_y  = particles.velocity_ys[tid];

            float delta_vx = 0;
            float delta_vy = 0;
            bool eaten_by_hole = false;
            for (int j = 0; j < black_hole_count; j++) {
                float distance_x = black_holes.position_xs[j] - pos_x;
                float distance_y = black_holes.position_ys[j] - pos_y;
                float distance_sq = distance_x*distance_x + distance_y*distance_y;
                if (distance_sq < 100.0f) {
                    eaten_by_hole = true;
                    deleted = true;
                    break;
                }
                float distance_cb = distance_sq * sqrt(distance_sq);
                delta_vx += distance_x / distance_cb;
                delta_vy += distance_y / distance_cb;
            }
            if (!eaten_by_hole) {
                vel_x += delta_vx * 1000;
                vel_y += delta_vy * 1000;

                pos_x += vel_x * dt / 1000;
                pos_y += vel_y * dt / 1000;

                pos_x = fmod(pos_x, (float)WINDOW_WIDTH);
                pos_y = fmod(pos_y, (float)WINDOW_HEIGHT);
            }

            particles.position_xs[tid] = pos_x;
            particles.position_ys[tid] = pos_y;
            particles.velocity_xs[tid] = vel_x;
            particles.velocity_ys[tid] = vel_y;
            particles.deleted[tid] = deleted;
        }
    }
}

// get the time in milliseconds
float get_time(void) {
    clock_t t = clock();
    return (float)(t * 1000.0 / CLOCKS_PER_SEC);
}

int rand_int(int min, int max) {
    return min + rand() % (max - min + 1);
}

int main(void) {
    int particle_count = MAX_PARTICLE_COUNT;
    int black_hole_count = 5;

    const char *file_times[1] = {"black_hole_render_times.csv"};
    float draw_times[1000];
    float update_times[1000];
    float generate_times[1000];
    float frame_times[1000];

    int block_size = 256;

    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        fprintf(stderr, "failed to init SDL: %s\n", SDL_GetError());
        return 1;
    }

    SDL_Window* window = SDL_CreateWindow(
        "Proyecto CAD",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        WINDOW_WIDTH, WINDOW_HEIGHT,
        0
    );
    if (window == NULL) {
        SDL_Quit();
        fprintf(stderr, "failed to create window: %s\n", SDL_GetError());
        return 1;
    }

    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (renderer == NULL) {
        SDL_Quit();
        fprintf(stderr, "failed to create renderer: %s\n", SDL_GetError());
        return 1;
    }

    SDL_Texture* texture = SDL_CreateTexture(
        renderer,
        SDL_PIXELFORMAT_ARGB8888,
        SDL_TEXTUREACCESS_STREAMING,
        WINDOW_WIDTH, WINDOW_HEIGHT
    );
    if (texture == NULL) {
        SDL_Quit();
        fprintf(stderr, "failed to create texture: %s\n", SDL_GetError());
        return 1;
    }

    pixel_t* pixels = new pixel_t[WINDOW_WIDTH * WINDOW_HEIGHT]();
    pixel_t* gpu_pixels = nullptr;
    cudaMalloc(&gpu_pixels, WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(pixel_t));

    particle_soa_t gpu_particles = {};
    cudaMalloc(&gpu_particles.position_xs, 4 * MAX_PARTICLE_COUNT * sizeof(float));
    cudaMalloc(&gpu_particles.colors, MAX_PARTICLE_COUNT * sizeof(pixel_t));
    cudaMalloc(&gpu_particles.deleted, MAX_PARTICLE_COUNT * sizeof(bool));
    gpu_particles.position_ys = gpu_particles.position_xs + MAX_PARTICLE_COUNT;
    gpu_particles.velocity_xs = gpu_particles.position_ys + MAX_PARTICLE_COUNT;
    gpu_particles.velocity_ys = gpu_particles.velocity_xs + MAX_PARTICLE_COUNT;
    int particles_grid_size = ceil((float)MAX_PARTICLE_COUNT / block_size);


    particle_soa_t gpu_black_holes = {};
    cudaMalloc(&gpu_black_holes.position_xs, 2 * MAX_BLACK_HOLE_COUNT * sizeof(float));
    gpu_black_holes.position_ys = gpu_black_holes.position_xs + MAX_BLACK_HOLE_COUNT;
    int black_hole_grid_size = ceil((float)MAX_BLACK_HOLE_COUNT / block_size);

    curandState* states;
    cudaMalloc(&states, MAX_PARTICLE_COUNT * sizeof(curandState));
    random_init<<<particles_grid_size, block_size>>>(states);
    generate_particles_cuda<<<particles_grid_size, block_size>>>(particle_count, gpu_particles, states);
    generate_black_holes_cuda<<<particles_grid_size, block_size>>>(black_hole_count, gpu_black_holes, states);

    int grid_size = ceil((float)WINDOW_WIDTH * WINDOW_HEIGHT / block_size);

    vec2_t mouse_pos = {};
    bool button_pressed[MOUSE_BUTTON_COUNT] = {};
    float dt = 0;
    SDL_Event event = {};
    bool window_close_requested = false;
    int cont = 0;
    while (!window_close_requested) {
        float start_time = get_time();

        cudaEvent_t e1, e2;
        cudaEventCreate(&e1);
        cudaEventCreate(&e2);
        cudaEventRecord(e1);
        draw_background<<<grid_size, block_size>>>(gpu_pixels);
        fast_draw<<<particles_grid_size, block_size>>>(gpu_pixels, gpu_particles, particle_count, gpu_black_holes, black_hole_count);
        cudaMemcpy(pixels, gpu_pixels, WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(pixel_t), cudaMemcpyDeviceToHost);
        SDL_UpdateTexture(texture, nullptr, (void*)pixels, 4 * WINDOW_WIDTH);
        SDL_RenderCopy(renderer, texture, nullptr, nullptr);
        SDL_RenderPresent(renderer);
        cudaEventRecord(e2);
        cudaEventSynchronize(e2);
        float draw_time = 0;
        cudaEventElapsedTime(&draw_time, e1, e2);
        printf("draw time: %f, ", draw_time);

        cudaEvent_t e3, e4;
        cudaEventCreate(&e3);
        cudaEventCreate(&e4);
        cudaEventRecord(e3);
        update_particles_cuda<<<particles_grid_size, block_size>>>(gpu_particles, particle_count, gpu_black_holes, black_hole_count, dt);
        cudaEventRecord(e4);
        cudaEventSynchronize(e4);
        float update_time = 0;
        cudaEventElapsedTime(&update_time, e3, e4);
        printf("update time: %f, ", update_time);
        
        float generate_start = get_time();
        if (button_pressed[MOUSE_BUTTON_LEFT] && particle_count < MAX_PARTICLE_COUNT) {
            float position_xs[100], position_ys[100], velocity_xs[100], velocity_ys[100];
            pixel_t colors[100];
            bool deleted[100];
            for (int i = 0; i < 100; i++) {
                position_xs[i] = mouse_pos.x + rand_int(-10, 10);
                position_ys[i] = mouse_pos.y + rand_int(-10, 10);
                velocity_xs[i] = rand_int(-15, 15);
                velocity_ys[i] = rand_int(-15, 15);
                deleted[i] = false;
                colors[i] = pixel_t { (uint8_t)rand_int(0, 255), (uint8_t)rand_int(0, 255), (uint8_t)rand_int(0, 255) };
            }
            cudaMemcpy(gpu_particles.position_xs + particle_count, position_xs, 100 * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(gpu_particles.position_ys + particle_count, position_ys, 100 * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(gpu_particles.velocity_xs + particle_count, velocity_xs, 100 * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(gpu_particles.velocity_ys + particle_count, velocity_ys, 100 * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(gpu_particles.colors + particle_count, colors, 100  * sizeof(pixel_t), cudaMemcpyHostToDevice);
            cudaMemcpy(gpu_particles.deleted + particle_count, deleted, 100 * sizeof(bool), cudaMemcpyHostToDevice);
            particle_count += 100;
        } 

        if (button_pressed[MOUSE_BUTTON_RIGHT] && black_hole_count < MAX_BLACK_HOLE_COUNT) {
            cudaMemcpy(gpu_black_holes.position_xs + black_hole_count, &mouse_pos.x, sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(gpu_black_holes.position_ys + black_hole_count, &mouse_pos.y, sizeof(float), cudaMemcpyHostToDevice);
            black_hole_count++;
        }
        float generate_end = get_time();
        float generate_ms = generate_end - generate_start;
        printf("generate time: %f, ", generate_ms);

        // poll events
        while (SDL_PollEvent(&event)) {
            switch (event.type) {
                case SDL_KEYDOWN:
                    if (event.key.keysym.scancode == SDL_SCANCODE_R) {
                        particle_count = 0;
                        black_hole_count = 0;
                    }
                    break;
                case SDL_MOUSEBUTTONDOWN:
                case SDL_MOUSEBUTTONUP:
                    button_pressed[event.button.button] = event.type == SDL_MOUSEBUTTONDOWN;
                    break;
                case SDL_MOUSEMOTION:
                    mouse_pos.x = event.motion.x;
                    mouse_pos.y = event.motion.y;
                    break;
                case SDL_QUIT:
                    window_close_requested = true;
                    break;
                default: break;
            }
        }

        float end_time = get_time();
        dt = end_time - start_time;
        printf("frame time: %f\n", dt);

        if (cont < 1000){
            draw_times[cont] = draw_time;
            update_times[cont] = update_time;
            generate_times[cont] = generate_ms;
            frame_times[cont] = dt;
            
        }
        if (cont == 1000){
            Write_times(file_times[0], draw_times, update_times, generate_times, frame_times);
        }
        cont++;
    }

    cudaFree(states);
    cudaFree(gpu_particles.position_xs);
    cudaFree(gpu_particles.colors);
    cudaFree(gpu_particles.deleted);
    cudaFree(gpu_black_holes.position_xs);
    cudaFree(gpu_pixels);
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
