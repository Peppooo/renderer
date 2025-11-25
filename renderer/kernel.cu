#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define SDL_MAIN_HANDLED
#include <SDL2/SDL.h>
#include <iostream>
#include <chrono>
#include "renderer.cuh"

using namespace std;

// camera infos

__global__ void render_pixel(uint32_t* data,vec3 origin,matrix rotation,float focal_length,bool move_light,int current_light_index,int ssaa,int reflections) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int idx = (x + y * w);
	if(idx >= w * h) { return; }

	int iC = x - w / 2;
	int jC = -(y - h / 2);
	int samples_count = 0;
	vec3 sum_sample = {0,0,0};
	vec3 pixel = {0,0,0};
	float steps_l = __cr_rsqrt(ssaa);
	for(float i = (iC - 1 / 2.0f); i < (iC + 1 / 2.0f); i += steps_l) {
		for(float j = (jC - 1 / 2.0f); j < (jC + 1 / 2.0f); j += steps_l) {
			vec3 dir = rotation * vec3{i,j,focal_length};
			pixel = compute_ray(origin,dir,reflections);

			for(int k = 0; k < lightsSize; k++) {
				float lightdot = dot(dir.norm(),(lights[k] - origin).norm());
				if(lightdot > (1 - 0.0001) && lightdot < (1 + 0.0001)) { // draws a sphere in the location of the light
					pixel = vec3{255,255,255};
					if(current_light_index == k && move_light)
					{
						pixel = vec3{255,0,255};
					}
				}
			}
			sum_sample += pixel;
			samples_count++;
		}
	}

	data[idx] = (sum_sample / samples_count).argb();
}

__shared__ uint32_t* d_framebuffer;

int main() {
	// scene infos
	object h_scene[50]; int h_sceneSize = 0; // scene size calculated step by step 
	vec3 h_lights[] = {{0,3,8},{-1,-1.5,10.5}}; const int h_lightsSize = sizeof(h_lights) / sizeof(vec3);

	cube(vec3{1,-2,5},3,3,3,vec3{0, 166, 30},h_scene,h_sceneSize,true);

	//cube(vec3{-2,-2,11},2,2,2,vec3{252, 186, 3},scene,sceneSize,true);

	object chess(vec3{0,-1,11},vec3{-3,-1,11},vec3{0,-1,15},vec3{0,0,0},h_scene,h_sceneSize,true,false,true); // triangle shaded with function chess_shading

	cube(vec3{-5,-2,4},1,3,7,vec3{10,10,50},h_scene,h_sceneSize,true);


	cube(vec3{-10,-2,-1},20,20,20,vec3{150,150,150},h_scene,h_sceneSize,false,false); // container

	object sphere_test(
		vec3{0.723185 , 0.7 , 9.81167}, // center
		vec3{1,0,0}, // radius , 0 
		vec3{0,0,0},
		vec3{200,0,150},h_scene,h_sceneSize,true,true
	);

	uint32_t* framebuffer = nullptr;

	cudaMalloc(&d_framebuffer,w * h * sizeof(uint32_t));

	cudaMemcpyToSymbol(scene,h_scene,h_sceneSize * sizeof(object),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(lights,h_lights,h_lightsSize * sizeof(vec3),0,cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol(sceneSize,&h_sceneSize,sizeof(int),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(lightsSize,&h_lightsSize,sizeof(int),0,cudaMemcpyHostToDevice);

	SDL_Init(SDL_INIT_VIDEO);

	// SDL Initialization
	SDL_Window* window = SDL_CreateWindow("RT",SDL_WINDOWPOS_CENTERED,SDL_WINDOWPOS_CENTERED,w,h,0);
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

	auto lastTime = std::chrono::high_resolution_clock::now();
	int nframe = 0;

	while(1) {
		nframe++;
		auto currentTime = chrono::high_resolution_clock::now();
		chrono::duration<float> deltaTime = currentTime - lastTime;
		lastTime = currentTime;
		cout << "frame time: " << deltaTime.count() * 1000 << " ms" << endl;
		auto rot = rotation(yaw,0,0);
		while(SDL_PollEvent(&e)) {
			if(e.type == SDL_QUIT) {
				return 0;
			}
			if(e.type == SDL_MOUSEMOTION) {
				yaw -= float(e.motion.xrel) * mouse_sens;
				pitch += float(e.motion.yrel) * mouse_sens;
			}
			if(e.type == SDL_KEYDOWN) {
				vec3 move = {0,0,0};
				if(e.key.keysym.scancode == SDL_SCANCODE_L) {
					move_light = !move_light;
					if(move_light) {
						current_light_index++;
						current_light_index = cycle(current_light_index,h_lightsSize);
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
					hq = !hq;
					cudaMemcpyToSymbol(d_hq,&hq,sizeof(bool),0,cudaMemcpyHostToDevice);
				}
				if(!move_light) {

					origin = origin + rot * move; // apply rotation transformation to the move vector
				}
				else {
					h_lights[current_light_index] = h_lights[current_light_index] + move;
					cudaMemcpyToSymbol(lights,h_lights,h_lightsSize * sizeof(vec3),0,cudaMemcpyHostToDevice);
				}
			}
		}
		int pitch;
		SDL_LockTexture(texture,nullptr,(void**)&framebuffer,&pitch);

		dim3 block(8,8);
		dim3 grid((w + block.x - 1) / block.x,(h + block.y - 1) / block.y);
		render_pixel << <grid,block >> > (d_framebuffer,origin,rot,foc_len,move_light,current_light_index,ssaa,reflections);
		//cout << origin.x << " , " << origin.y << " , " << origin.z << endl;
		//cout << yaw << endl;
		//cudaDeviceSynchronize();

		cudaMemcpy(framebuffer,d_framebuffer,w * h * sizeof(uint32_t),cudaMemcpyDeviceToHost);


		cudaError_t err = cudaGetLastError();
		if(err != cudaSuccess) printf("CUDA error: %s\n",cudaGetErrorString(err));
		SDL_UnlockTexture(texture);
		SDL_RenderClear(renderer);
		SDL_RenderCopy(renderer,texture,nullptr,nullptr);
		SDL_RenderPresent(renderer);
	}
}