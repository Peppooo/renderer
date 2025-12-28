#pragma once
#include "renderer.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <SDL2/SDL.h>
#include <chrono>
#include "physics.cuh"
#include "algebra.cuh"

class renderer {
private:
	// limits
	const int MAX_LIGHTS = 32;
	//host
	SDL_Window* sdl_window;
	SDL_Renderer* sdl_renderer;
	SDL_Texture* frame_texture;
	bvh tree;
	chrono::time_point<chrono::steady_clock> lastTime;
	uint32_t* framebuffer;

	// device
	uint32_t* d_framebuffer;
	

public:
	// host

	int w,h;
	vec3 origin = {0,0,0};
	float fov;
	float yaw,pitch;
	int max_reflections,ssaa,indirect_rays,n_samples_pixel;
	float frame_dt; int frame_n;
	size_t lightsSize;

	// device

	Scene* scene;
	vec3* lights;

	__host__ renderer(const int& W,const int& H,const float& Fov = M_PI_2,const int& samples_per_pixel = 1,const int& Max_reflections = 4,const int& Ssaa = 1,const int Indirect_rays=32): // rotation = {yaw,pitch}, Fov is in radians
		yaw(0),pitch(0),origin({0,0,0}),frame_n(0),frame_dt(0),w(W),h(H),fov(Fov),max_reflections(Max_reflections),ssaa(Ssaa),n_samples_pixel(samples_per_pixel),indirect_rays(Indirect_rays) {
	}
	__host__ void init(const char* win_name) {
		SDL_Init(SDL_INIT_EVERYTHING);
		sdl_window = SDL_CreateWindow(win_name,SDL_WINDOWPOS_CENTERED,SDL_WINDOWPOS_CENTERED,w,h,0);
		sdl_renderer = SDL_CreateRenderer(sdl_window,-1,SDL_RENDERER_ACCELERATED);
		frame_texture = SDL_CreateTexture(
			sdl_renderer,
			SDL_PIXELFORMAT_ARGB8888,
			SDL_TEXTUREACCESS_STREAMING,
			w,
			h
		);
		cudaMalloc(&d_framebuffer,sizeof(uint32_t) * w * h);
		cudaMalloc(&scene,sizeof(Scene));
		cudaMalloc(&lights,sizeof(vec3) * MAX_LIGHTS);
	}
	__host__ void render(int seed=time(0)) {
		auto currentTime = chrono::high_resolution_clock::now();
		chrono::duration<float> deltaTime = currentTime - lastTime;
		frame_dt = deltaTime.count();
		lastTime = currentTime;
		
		int _pitch;
		SDL_LockTexture(frame_texture,nullptr,(void**)&framebuffer,&_pitch);

		float focal_length = w / (2 * tanf(fov / 2));

		matrix rot = rotation(0,pitch,yaw);

		dim3 block(8,8);
		dim3 grid((w + block.x - 1) / block.x,(h + block.y - 1) / block.y);


		render_pixel<<<grid,block>>>(w,h,lights,lightsSize,scene,tree,d_framebuffer,origin,rot,focal_length,indirect_rays,ssaa,max_reflections,n_samples_pixel,seed);

		CUDA_CHECK(cudaGetLastError());

		CUDA_CHECK(cudaDeviceSynchronize());

		cudaMemcpy(framebuffer,d_framebuffer,sizeof(uint32_t) * w * h,cudaMemcpyDeviceToHost);

		SDL_UnlockTexture(frame_texture);
		SDL_RenderClear(sdl_renderer);
		SDL_RenderCopy(sdl_renderer,frame_texture,nullptr,nullptr);
		SDL_RenderPresent(sdl_renderer);
		frame_n++;
	}
	void import_scene_from_host(const Scene* h_scene) const {
		cudaMemcpy(scene,h_scene,sizeof(Scene),cudaMemcpyHostToDevice);
	}

	void import_scene_from_host_array(object* h_scene,const size_t h_sceneSize) {
		//build_scene_bounding_box(bounding,h_scene,h_sceneSize);
		tree.build(20,h_scene,h_sceneSize);
		tree.printNodes();
		Scene* h_scene_soa = new Scene;
		h_scene_soa->sceneSize = 0;
		for(int i = 0; i < h_sceneSize; i++) {
			h_scene_soa->addObject(h_scene[i]);
		}
		CUDA_CHECK(cudaMemcpy(scene,h_scene_soa,sizeof(Scene),cudaMemcpyHostToDevice));
		delete h_scene_soa;
	}
	void import_lights_from_host(const vec3* h_lights,const int h_lightsSize) {
		cudaMemcpy(lights,h_lights,sizeof(vec3)*h_lightsSize,cudaMemcpyHostToDevice);
		lightsSize = h_lightsSize;
	}
};