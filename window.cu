#include <SDL2/SDL.h>

#include <stdbool.h>
#include <stdio.h>

#include "types.h"

typedef enum mouse_button_t {
    MOUSE_BUTTON_LEFT = SDL_BUTTON_LEFT,
    MOUSE_BUTTON_MIDDLE = SDL_BUTTON_MIDDLE,
    MOUSE_BUTTON_RIGHT = SDL_BUTTON_RIGHT,

    MOUSE_BUTTON_COUNT,
} mouse_button_t;

typedef struct window_t {
    SDL_Window*   window;
    SDL_Renderer* renderer;
    bool          quit_signalled;
    bool          error_occurred;

    bool          mouse_button_down[MOUSE_BUTTON_COUNT];
    vec2_t        mouse_position;

    bool          key_down[SDL_NUM_SCANCODES];
} window_t;


window_t create_window(int width, int height) {
    window_t window = {0};
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        fprintf(stderr, "failed to init SDL: %s\n", SDL_GetError());
        window.error_occurred = true;
        return window;
    }

    window.window = SDL_CreateWindow("Proyecto CAD", 
                                     SDL_WINDOWPOS_CENTERED,
                                     SDL_WINDOWPOS_CENTERED,
                                     width,
                                     height,
                                     0);

    if (window.window == NULL) {
        fprintf(stderr, "failed to create window: %s\n", SDL_GetError());
        window.error_occurred = true;
        SDL_Quit();
        return window;
    }

    window.renderer = SDL_CreateRenderer(window.window, -1, SDL_RENDERER_ACCELERATED);
    if (window.renderer == NULL) {
        fprintf(stderr, "failed to create renderer: %s\n", SDL_GetError());
        window.error_occurred = true;
        SDL_Quit();
        return window;
    }

    return window;
}

void poll_events(window_t* window) {
    SDL_Event event = {};
    while (SDL_PollEvent(&event)) {
        switch (event.type) {
            case SDL_MOUSEMOTION: {
                window->mouse_position.x = event.motion.x;                
                window->mouse_position.y = event.motion.y;
            } break;

            case SDL_MOUSEBUTTONUP:
            case SDL_MOUSEBUTTONDOWN: {
                window->mouse_button_down[event.button.button] = event.button.state == SDL_PRESSED;
            } break;

            case SDL_KEYUP:
            case SDL_KEYDOWN: {
                window->key_down[event.key.keysym.scancode] = event.key.state == SDL_PRESSED;
            } break;

            case SDL_QUIT:
                window->quit_signalled = true;
                break;
        }
    }
}

void draw(window_t* window, 
          particle_t* particles, size_t particle_count,
          particle_t* black_holes, size_t black_hole_count) {
    // make the window gray
    SDL_SetRenderDrawColor(window->renderer, 0x80, 0x80, 0x80, SDL_ALPHA_OPAQUE);
    SDL_RenderFillRect(window->renderer, NULL);

    for (int i = 0; i < particle_count; i++) {
        particle_t* particle = &particles[i];
        if (!particle->deleted) {
            // // draw a trailing line
            // SDL_SetRenderDrawColor(window->renderer, 0xff, 0xff, 0xff, SDL_ALPHA_OPAQUE);
            // SDL_RenderDrawLine(window->renderer,
            //                    particle->position.x - particle->velocity.x,
            //                    particle->position.y - particle->velocity.y,
            //                    particle->position.x, particle->position.y);

            SDL_Rect rect = {
                .x = (int)particle->position.x, .y = (int)particle->position.y, 
                .w = 4, .h = 4,
            };

            // set the draw color to the particle's color
            SDL_SetRenderDrawColor(window->renderer, 
                                   particle->color.r,
                                   particle->color.g,
                                   particle->color.b,
                                   SDL_ALPHA_OPAQUE);
            SDL_RenderFillRect(window->renderer, &rect);
        }
    }

    // draw the black holes
    SDL_SetRenderDrawColor(window->renderer, 0, 0, 0, SDL_ALPHA_OPAQUE);
    for (int i = 0; i < black_hole_count; i++) {
        SDL_Rect rect = {
            .x = (int)black_holes[i].position.x, .y = (int)black_holes[i].position.y, 
            .w = 10, .h = 10,
        };

        SDL_RenderFillRect(window->renderer, &rect);
    }

    SDL_RenderPresent(window->renderer);
}
