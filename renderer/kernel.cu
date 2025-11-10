#define SDL_MAIN_HANDLED
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <SDL2/SDL.h>
#include <iostream>
#include "utils.h"


using namespace std;

// camera infos

constexpr int w = 512,h = 512;
constexpr double fov = M_PI / 1.8;
constexpr double move_speed = 0.1;
constexpr double mouse_sens = 0.001;
double foc_len = w / (2 * tan(fov / 2));

__constant__ __device__ int reflections = 4;

__global__ void render_pixel(uint32_t* data,vec3 origin,matrix rotation,trig* scene,double focal_length,int sceneSize,vec3* lights,int lightsSize,bool move_light,int current_light_index) {
	int idx = (threadIdx.x + blockIdx.x * blockDim.x);
	if(idx >= w * h) { return; }
	int x = idx % w;
	int y = idx / w;
	int i = x - w / 2;
	int j = -(y - h / 2);
	vec3 pixel = {0,0,0};
	vec3 dir = rotation * vec3{double(i),double(j),focal_length};

	pixel = compute_ray(origin,dir,scene,sceneSize,lights,lightsSize,reflections);

	for(int k = 0; k < lightsSize; k++) {
		double lightdot = dot(dir.norm(),(lights[k] - origin).norm());
		if(lightdot > (1 - 0.0001) && lightdot < (1 + 0.0001)) { // draws a sphere in the location of the light
			pixel = vec3{255,255,255};
			if(current_light_index == k && move_light)
			{
				pixel = vec3{255,0,255};
			}
		}
	}
	data[idx] = pixel.argb();
}

vec3 origin = {0,0,0};
bool move_light = false;
int current_light_index = 0;
double yaw = 0,pitch = 0,roll = 0;
int wp = 2,hp = 2;

int main() {
	// scene infos
	trig scene[50]; int sceneSize = 0; // scene size calculated step by step 
	vec3 lights[] = {{0,3,8},{-1,-1.5,10.5}}; const int lightsSize = sizeof(lights) / sizeof(vec3);

	cube(vec3{1,-2,9},3,3,3,vec3{255,0,0},scene,sceneSize,false); // matte red

	cube(vec3{-2,-2,11},2,2,2,vec3{20,20,20},scene,sceneSize,true); // reflective dark grey
	

	cube(vec3{-10,-2,-1},20,20,20,vec3{150,150,150},scene,sceneSize,false); // container

	trig reflection_test(
		vec3{-3,-2,14},
		vec3{-3,-2,6},
		vec3{-3,4,10},
		vec3{255,255,255},scene,sceneSize,true
	);

	//cube();

	scene[sceneSize] = reflection_test;

	uint32_t* framebuffer = nullptr;

	// GPU allocations
	trig* d_scene;
	vec3* d_lights;
	uint32_t* d_framebuffer;


	cudaMalloc(&d_scene,sceneSize * sizeof(trig));
	cudaMalloc(&d_lights,lightsSize * sizeof(vec3));
	cudaMalloc(&d_framebuffer,w * h * sizeof(uint32_t));

	cudaMemcpy(d_scene,scene,sceneSize * sizeof(trig),cudaMemcpyHostToDevice);
	cudaMemcpy(d_lights,lights,lightsSize * sizeof(vec3),cudaMemcpyHostToDevice);


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


	while(1) {
		auto rot_matrix = rotationY(yaw);
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
						current_light_index = cycle(current_light_index,lightsSize);
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
					origin = origin + rot_matrix * move; // apply rotation transformation to the move vector
				}
				else {
					lights[current_light_index] = lights[current_light_index] + move;
					cudaMemcpy(d_lights,lights,lightsSize * sizeof(vec3),cudaMemcpyHostToDevice);
				}
			}
		}
		int pitch;
		SDL_LockTexture(texture,nullptr,(void**)&framebuffer,&pitch);

		int numBlocks = (w * h + 255) / 256;
		render_pixel << <numBlocks,256 >> > (d_framebuffer,origin,rot_matrix,d_scene,foc_len,sceneSize,d_lights,lightsSize,move_light,current_light_index);
		cudaDeviceSynchronize();

		//render_pixel(d_framebuffer,origin,rot_matrix,d_scene,foc_len,sceneSize,lights,lightsSize,move_light,current_light_index);
		cudaMemcpy(framebuffer,d_framebuffer,w * h * sizeof(uint32_t),cudaMemcpyDeviceToHost);

		cudaError_t err = cudaGetLastError();

		if(err != cudaSuccess) printf("CUDA error: %s\n",cudaGetErrorString(err));
		SDL_UnlockTexture(texture);
		SDL_RenderClear(renderer);
		SDL_RenderCopy(renderer,texture,nullptr,nullptr);
		SDL_RenderPresent(renderer);
	}
}