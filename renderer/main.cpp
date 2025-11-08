#define SDL_MAIN_HANDLED
#include <SDL2/SDL.h>
#include <iostream>
#include "utils.h"

using namespace std;

const int w = 1024,h = 1024;
const int wp = 8,hp = 8;

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

trig reflection_test(
	vec3{-3,-2,12},
	vec3{-3,-2,8},
	vec3{-3,2,10},
	vec3{255,255,255}
);


int main() {
	SDL_Init(SDL_INIT_VIDEO);
	SDL_CreateWindowAndRenderer(w,h,0,&window,&renderer);
	SDL_Event e;
	SDL_SetRelativeMouseMode(SDL_TRUE);
	while(1) {
		while(trig::triangles.size() > 13) {
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
						cout << current_light_index << endl;
						current_light_index = cycle(current_light_index,lights.size());
						cout << current_light_index << endl;
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
				if(!move_light) {
					origin = origin + move * rot_matrix; // apply rotation transformation to the move vector
				}
				else {
					lights[current_light_index] = lights[current_light_index] + move; // lets you move the light point
				}
			}
		}
		SDL_SetRenderDrawColor(renderer,0,0,0,255);
		SDL_RenderClear(renderer);
		SDL_SetRenderDrawColor(renderer,255,255,255,255);

		for(int i = -w / 2; i < w / 2; i += wp) {
			for(int j = -h / 2; j < h / 2; j += hp) {
				if(j == 0 && i == 0) {
					IS_CENTER_PIXEL = true;
				}
				else {
					IS_CENTER_PIXEL = false;
				}
				vec3 dir = {double(i),double(j),foc_len};
				dir = dir * rot_matrix;
				vec3 p,surf_norm;

				SDL_Color c = compute_ray(origin,dir,lights,1);
				SDL_SetRenderDrawColor(renderer,c.r,c.g,c.b,255);
				SDL_Rect r{i + w / 2,h - (j + h / 2),wp,hp};
				SDL_RenderFillRect(renderer,&r);

				for(int k = 0; k < lights.size(); k++) {
					double lightdot = dot(dir.norm(),(lights[k] - origin).norm());
					if(lightdot > (1 - 0.0001) && lightdot < (1 + 0.0001)) { // draws a sphere in the location of the light
						if(current_light_index == k && move_light)
						{
							SDL_SetRenderDrawColor(renderer,255,0,255,255);
						}
						else {
							SDL_SetRenderDrawColor(renderer,255,255,255,255);
						}
						SDL_Rect r{i + w / 2,h - (j + h / 2),wp,hp};
						SDL_RenderFillRect(renderer,&r);
					}
				}
			}
		}

		SDL_Delay(16);
		SDL_RenderPresent(renderer);
	}
}