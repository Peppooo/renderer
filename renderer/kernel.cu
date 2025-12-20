#define SDL_MAIN_HANDLED
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <SDL2/SDL.h>
#include <iostream>
#include <chrono>
#include "renderer.cuh"
#include "physics.cuh"

using namespace std;

__device__ uint32_t* d_framebuffer;

const vec3 gravity = {0,-1,0};


int main() {
	// scene infos
	object* h_scene = new object[MAX_OBJ]; int h_sceneSize = 0; // scene size calculated step by step 
	vec3 h_lights[1] = {{0,1,0}}; int h_lightsSize = 1;

	COLOR_TEXTURE(white_texture,(vec3{200,200,200}));
	COLOR_TEXTURE(green_texture,(vec3{0,200,0}));
	COLOR_TEXTURE(red_texture,(vec3{200,0,0}));

	IMPORT_TEXTURE(floor_texture,"..\\textures\\floor2.tex",1,799,783);
	
	plane({-2,-2,-2},{2,-2,-2},{2,-2,2},{-2,-2,2},h_scene,h_sceneSize,material(glossy,0.65f),floor_texture);

	plane({-2,-2,-2},{-2,2,-2},{-2,-2,2},{-2,2,2},h_scene,h_sceneSize,material(glossy,0.6f),red_texture);

	plane({2,-2,-2},{2,2,-2},{2,-2,2},{2,2,2},h_scene,h_sceneSize,material(glossy,0.75f),green_texture);


	plane({-2,2,-2},{2,2,-2},{2,2,2},{-2,2,2},h_scene,h_sceneSize,material(diffuse),white_texture);

	plane({2,2,2},{-2,2,2},{2,-2,2},{-2,-2,2},h_scene,h_sceneSize,material(glossy,0.8f),white_texture);

	uint32_t* framebuffer = nullptr;
	
	cudaMalloc(&d_framebuffer,sizeof(uint32_t) * w * h);

	cudaMemcpyToSymbol(lights,h_lights,h_lightsSize * sizeof(vec3),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(lightsSize,&h_lightsSize,sizeof(int),0,cudaMemcpyHostToDevice);
	
	Scene* SoA_h_scene = new Scene();

	SDL_Init(SDL_INIT_EVERYTHING);
	int numKeys;
	const Uint8* keystates=SDL_GetKeyboardState(&numKeys);

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

	Scene* scene;
	cudaMallocManaged(&scene,sizeof(Scene));

	SoA_h_scene->sceneSize = 0;
	for(int i = 0; i < h_sceneSize; i++) { // convert host scene (AoS) to device scene (SoA)
		SoA_h_scene->addObject(h_scene[i]);
	}
	cudaMemcpy(scene,SoA_h_scene,sizeof(Scene),cudaMemcpyHostToDevice);
	
	while(1) {
		nframe++;
		auto currentTime = chrono::high_resolution_clock::now();
		chrono::duration<float> deltaTime = currentTime - lastTime;
		float dt = deltaTime.count();
		lastTime = currentTime;
		sum_time += deltaTime.count() * 1000;

		

		if(nframe % 100 == 0) {
			cout << "frame time: " << sum_time/100 << " ms" << endl; // average frame time out of 10
			sum_time = 0;
		}

		auto rot = rotation(0,pitch,yaw);
		while(SDL_PollEvent(&e)) {
			if(e.type == SDL_KEYDOWN) {
				if(keystates[SDL_SCANCODE_L]) {
					move_light = !move_light;
					if(move_light) {
						current_light_index++;
						current_light_index = cycle(current_light_index,h_lightsSize);
					}
				}
				if(keystates[SDL_SCANCODE_H]) {
					hq = !hq;
					cudaMemcpyToSymbol(d_hq,&hq,sizeof(bool),0,cudaMemcpyHostToDevice);
				}
			}
			if(e.type == SDL_QUIT) {
				return 0;
			}
			if(e.type == SDL_MOUSEMOTION) {
				yaw   -= e.motion.xrel * mouse_sens;
				pitch -= e.motion.yrel * mouse_sens;
			}
		}



		vec3 move = {0,0,0};
		float curr_move_speed = move_speed;
		if(keystates[SDL_SCANCODE_W]) {
			move.z += 1;
		}
		if(keystates[SDL_SCANCODE_S]) {
			move.z -= 1;
		}
		if(keystates[SDL_SCANCODE_D]) {
			move.x += 1;
		}
		if(keystates[SDL_SCANCODE_A]) {
			move.x -= 1;
		}
		if(keystates[SDL_SCANCODE_Q]) {
			move.y -= 1;
		}
		if(keystates[SDL_SCANCODE_E]) {
			move.y += 1;
		}
		if(keystates[SDL_SCANCODE_LSHIFT]) {
			curr_move_speed *= 8;
		}
		if(move.len2() != 0) {
			if(!move_light) {
				origin += rotation(0,0,yaw) * move.norm() * curr_move_speed * dt;
			}
			else {
				h_lights[current_light_index] = h_lights[current_light_index] + move.norm() * move_speed * dt;
				cudaMemcpyToSymbol(lights,h_lights,h_lightsSize * sizeof(vec3),0,cudaMemcpyHostToDevice);
			}
		}
		
		int _pitch;
		SDL_LockTexture(texture,nullptr,(void**)&framebuffer,&_pitch);
		
		dim3 block(8,8);
		dim3 grid((w + block.x - 1) / block.x,(h + block.y - 1) / block.y);

		int seed = nframe*2134;

		render_pixel<<<grid,block>>>(scene,d_framebuffer,origin,rot,foc_len,move_light,current_light_index,ssaa,reflections,8,seed);

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