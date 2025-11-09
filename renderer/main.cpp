#define SDL_MAIN_HANDLED
#include <SDL2/SDL.h>
#include <iostream>
#include "utils.h"

using namespace std;

const int w = 1024,h = 1024;
int wp = 4,hp = 4;

SDL_Window* window;
SDL_Renderer* renderer;

// camera infos
const double fov = M_PI/1.5;
vec3 origin = {0,0,0};
double yaw = 0,pitch = 0,roll = 0;
double move_speed = 0.1;
double mouse_sens = 0.001;
bool move_light = false;
int current_light_index = 0;

const double foc_len = w / (2 * tan(fov / 2));

// scene infos

vector<vec3> lights = {vec3{0,3,8},{1.5,-1.5,10.5}};

cube cube1(vec3{0,-2,9},3,3,3,vec3{255,0,0});

cube cube2(vec3{-10,-5,-5},20,20,20,vec3{255,255,0});

trig reflection_test(
	vec3{-3,-2,12},
	vec3{-3,-2,8},
	vec3{-3,2,10},
	vec3{255,255,255},true
);


int main() {
	SDL_Init(SDL_INIT_VIDEO);
	SDL_Window* window = SDL_CreateWindow("Framebuffer",SDL_WINDOWPOS_CENTERED,SDL_WINDOWPOS_CENTERED,w,h,0);
	SDL_Renderer* renderer = SDL_CreateRenderer(window,-1,SDL_RENDERER_ACCELERATED);
	SDL_Texture* texture = SDL_CreateTexture(
		renderer,
		SDL_PIXELFORMAT_ARGB8888,
		SDL_TEXTUREACCESS_STREAMING,
		w,
		h
	);
	SDL_Event e;
	SDL_SetRelativeMouseMode(SDL_TRUE);
	int initial_trig_num = trig::triangles.size();

	uint32_t* framebuffer = new uint32_t[w*h];

	while(1) {
		while(trig::triangles.size() > initial_trig_num) {
			delete trig::triangles.back();
			trig::triangles.pop_back();
		}
		//auto rot_matrix = rotation(yaw,pitch,roll);
		auto rot_matrix = rotationYaw(yaw);
		while(SDL_PollEvent(&e)) {
			if(e.type == SDL_QUIT) {
				return 0;
			}
			if(e.type == SDL_MOUSEMOTION) {
				yaw -= double(e.motion.xrel) * mouse_sens;
			}
			if(e.type == SDL_KEYDOWN) {
				vec3 move = {0,0,0};
				if(e.key.keysym.scancode == SDL_SCANCODE_L) {
					move_light = !move_light;
					if(move_light) {
						current_light_index++;
						current_light_index = cycle(current_light_index,lights.size());
					}
				}
				if(e.key.keysym.scancode == SDL_SCANCODE_W) {
					move.z += move_speed;
				}
				if(e.key.keysym.scancode == SDL_SCANCODE_S) {
					move.z -= move_speed;
				}
				if(e.key.keysym.scancode == SDL_SCANCODE_D) {
					move.x += move_speed;
				}
				if(e.key.keysym.scancode == SDL_SCANCODE_A) {
					move.x -= move_speed;
				}
				if(e.key.keysym.scancode == SDL_SCANCODE_Q) {
					move.y -= move_speed;
				}
				if(e.key.keysym.scancode == SDL_SCANCODE_E) {
					move.y += move_speed;
				}
				if(e.key.keysym.scancode == SDL_SCANCODE_H) {
					wp = 1; hp = 1;
				}
				if(!move_light) {
					origin = origin + move * rot_matrix; // apply rotation transformation to the move vector
				}
				else {
					lights[current_light_index] = lights[current_light_index] + move; // lets you move the light point
				}
			}
		}
		int pitch;
		SDL_LockTexture(texture,nullptr,(void**)&framebuffer,&pitch);
		for(int i = -w / 2; i < w / 2; i += wp) {
			for(int j = -h / 2; j < h / 2; j += hp) {
				int x = (i + w / 2); int y = (j + h / 2);
				vec3 pixel = {0,0,0};
				if(j == 0 && i == 0) {
					IS_CENTER_PIXEL = true;
				}
				else {
					IS_CENTER_PIXEL = false;
				}
				vec3 dir = vec3{double(i),double(j),foc_len}*rot_matrix;

				pixel = compute_ray(origin,dir,lights,4);

				for(int k = 0; k < lights.size(); k++) {
					double lightdot = dot(dir.norm(),(lights[k] - origin).norm());
					if(lightdot > (1 - 0.0001) && lightdot < (1 + 0.0001)) { // draws a sphere in the location of the light
						if(current_light_index == k && move_light)
						{
							pixel = vec3{255,0,255};
						}
						else {
							pixel = vec3{255,255,255};
						}
					}
				}

				for(int qx = x; qx < (x + hp); qx++) {
					for(int qy = y; qy < (y + hp); qy++) {
						framebuffer[w * ((h-1)-qy) + qx] = pixel.argb(); // 
					}
				}
			}
		}

		SDL_UnlockTexture(texture);
		SDL_RenderClear(renderer);
		SDL_RenderCopy(renderer,texture,nullptr,nullptr);
		SDL_RenderPresent(renderer);
	}
}