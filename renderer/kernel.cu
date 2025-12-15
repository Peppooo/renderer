#define SDL_MAIN_HANDLED
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <SDL2/SDL.h>
#include <iostream>
#include <chrono>

#include "renderer.cuh"
#include "physics.cuh"

using namespace std;

__device__ uint32_t* d_framebuffer;

int main() {

	// scene infos
	object h_scene[50]; int h_sceneSize = 0; // scene size calculated step by step 
	vec3 h_lights[] = {{0,1.5f,0}}; const int h_lightsSize = sizeof(h_lights) / sizeof(vec3);

	
	//cube({-3,-,-3},5,4,3,{100,100,100},h_scene,h_sceneSize,false,false);
	
	plane({2,-2,-2},{2,2,-2},{2,-2,2},{2,2,2},{0,200,0},h_scene,h_sceneSize,material(diffuse),false);

	plane({-2,-2,-2},{2,-2,-2},{2,-2,2},{-2,-2,2},{},h_scene,h_sceneSize,material(glossy ,0.1f),true);


	plane({-2,-2,-2},{-2,2,-2},{-2,-2,2},{-2,2,2},{200,0,0},h_scene,h_sceneSize,material(glossy,0.7f),false);

	plane({2,-2,-2},{2,2,-2},{2,-2,2},{2,2,2},{0,200,0},h_scene,h_sceneSize,material(glossy,0.7f),false);


	//cube({-2.05,-2.05,-2},4.1,4,4,{150,150,150},h_scene,h_sceneSize,false,false);

	//sphere({0,-1,0.8f},0.5f,{0,0,0},h_scene,h_sceneSize,material(specular),false);

	uint32_t* framebuffer = nullptr;

	cudaMalloc(&d_framebuffer,sizeof(uint32_t) * w * h);

	cudaMemcpyToSymbol(lights,h_lights,h_lightsSize * sizeof(vec3),0,cudaMemcpyHostToDevice);

	Scene SoA_h_scene;
	SoA_h_scene.sceneSize = 0;
	for(int i = 0; i < h_sceneSize; i++) { // convert host scene (AoS) to device scene (SoA)
		SoA_h_scene.addObject(h_scene[i]);
	}

	cudaMemcpyToSymbol(scene,&SoA_h_scene,sizeof(SoA_h_scene),0,cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol(lightsSize,&h_lightsSize,sizeof(int),0,cudaMemcpyHostToDevice);

	SDL_Init(SDL_INIT_EVERYTHING);

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

	auto lastTime = chrono::high_resolution_clock::now();
	int nframe = 0;

	float sum_time = 0;
	
	while(1) {
		nframe++;
		auto currentTime = chrono::high_resolution_clock::now();
		chrono::duration<float> deltaTime = currentTime - lastTime;
		lastTime = currentTime;
		sum_time += deltaTime.count() * 1000;


		if(nframe % 10 == 0) {
			cout << "frame time: " << sum_time/ 10 << " ms" << endl; // average frame time out of 10
			sum_time = 0;
		}

		auto rot = rotation(0,pitch,yaw);
		while(SDL_PollEvent(&e)) {
			if(e.type == SDL_QUIT) {
				return 0;
			}
			if(e.type == SDL_MOUSEMOTION) {
				yaw -= float(e.motion.xrel) * mouse_sens;
				pitch -= float(e.motion.yrel) * mouse_sens;
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
				if(!move_light && move.len2() != 0) {
					origin += rotation(0,0,yaw) * move.norm()*move_speed;
				}
				if(move_light) {
					h_lights[current_light_index] = h_lights[current_light_index] + move;
					cudaMemcpyToSymbol(lights,h_lights,h_lightsSize * sizeof(vec3),0,cudaMemcpyHostToDevice);
				}
			}
		}
		
		
		int _pitch;
		SDL_LockTexture(texture,nullptr,(void**)&framebuffer,&_pitch);
		
		dim3 block(8,8);
		dim3 grid((w + block.x - 1) / block.x,(h + block.y - 1) / block.y);

		render_pixel<<<grid,block>>>(d_framebuffer,origin,rot,foc_len,move_light,current_light_index,ssaa,reflections,32);

		cudaError_t err = cudaGetLastError();
		if(err != cudaSuccess) {
			printf("Kernel error: %s\n",cudaGetErrorString(err));
		}
		cudaDeviceSynchronize();

		cudaMemcpy(framebuffer,d_framebuffer,sizeof(uint32_t)*w*h,cudaMemcpyDeviceToHost);

		SDL_UnlockTexture(texture);
		SDL_RenderClear(renderer);
		SDL_RenderCopy(renderer,texture,nullptr,nullptr);
		SDL_RenderPresent(renderer);
	}
}