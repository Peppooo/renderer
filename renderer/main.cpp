#define SDL_MAIN_HANDLED
#include <SDL2/SDL.h>
#include <iostream>
#include "utils.h"

using namespace std;

const int w = 512,h = 512;

SDL_Window* window;
SDL_Renderer* renderer;

// camera infos
const double fov = M_PI_4;
vec3 origin = {0,0,0};
double yaw = 0,pitch = 0,roll=0;
double move_speed = 0.1;
double mouse_sens = 0.001;
bool move_light = false;

const double foc_len = w / (2 * tan(fov / 2));

// scene infos

vec3 light_pos = {0,3,10};

trig triangle1(
	vec3{-1,-1,12},
	vec3{1,-1,12},
	vec3{0,1,10},
	vec3{255,0,0}
);
trig triangle2(
	vec3{1,-1,12},
	vec3{1,-1,8},
	vec3{0,1,10},
	vec3{255,0,0}
);
trig triangle3(
	vec3{-1,-1,12},
	vec3{-1,-1,8},
	vec3{0,1,10},
	vec3{255,0,0}
);
trig triangle4(
	vec3{-1,-1,8},
	vec3{1,-1,8},
	vec3{0,1,10},
	vec3{255,0,0}
);

trig reflection_test(
	vec3{-3,-2,12},
	vec3{-3,-2,8},
	vec3{-4,2,10},
	vec3{255,255,255}
);


int main() {
	SDL_Init(SDL_INIT_VIDEO);
	SDL_CreateWindowAndRenderer(w,h,0,&window,&renderer);
	SDL_Event e;
	SDL_SetRelativeMouseMode(SDL_TRUE);
	while(1) {
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
				if(!move_light) { 
					origin = origin + move * rot_matrix; // apply rotation transformation to the move vector
				}
				else {
					light_pos = light_pos + move; // lets you move the light point
				}
			}
		}
		SDL_SetRenderDrawColor(renderer,0,0,0,255);
		SDL_RenderClear(renderer);
		SDL_SetRenderDrawColor(renderer,255,255,255,255);
		for(int i = -w / 2; i < w / 2; i++) {
			for(int j = -h / 2; j < h / 2; j++) {
				vec3 dir = {double(i),double(j),foc_len};
				dir = dir*rot_matrix;
				vec3 p,surf_norm;
				double lightdot = dot(dir.norm(),(light_pos - origin).norm());
				double epsilon = 0.00001;
				if(lightdot>(1-epsilon) && lightdot < (1+epsilon)) { // draws a sphere in the location of the light
					SDL_SetRenderDrawColor(renderer,255,255,255,255);
					SDL_RenderDrawPoint(renderer,i + w / 2,h - (j + h / 2)); 
				}
				else {
					SDL_Color c = compute_ray(origin,dir,light_pos,2);
					SDL_SetRenderDrawColor(renderer,c.r,c.g,c.b,255);
					SDL_RenderDrawPoint(renderer,i + w / 2,h - (j + h / 2));
				}
				
			}
		}
		
		SDL_RenderPresent(renderer);
	}
}