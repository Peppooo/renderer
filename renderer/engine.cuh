#pragma once
#include "renderer.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <SDL2/SDL.h>
#include <chrono>
#include "physics.cuh"
#include "algebra.cuh"

__device__ void insertionSort(float* v,int n)
{
	for(int i = 1; i < n; i++) {
		float key = v[i];
		int j = i - 1;

		while(j >= 0 && v[j] > key) {
			v[j + 1] = v[j];
			j--;
		}
		v[j + 1] = key;
	}
}

__global__ void acc_render(int w,int h,uint32_t* return_buffer,vec3* acc_buffer,int frames_accumulated) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int idx = (x + y * w);
	if(idx >= w * h) return;
	if(frames_accumulated == 0) {
		acc_buffer[idx] = vec3{0,0,0};
	}
	else {
		return_buffer[idx] = (acc_buffer[idx] / (float)frames_accumulated).argb();
	}
}

__global__ void postprocess(int w,int h,uint32_t* return_buff,vec3* buff,int frames_accumulated) {
	#define IDX(X,Y) ((X) + ((Y) * (w)))
	#define col(X,Y) (buff[IDX(X,Y)]/((float)frames_accumulated))
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if(x >= (w - 2) || x <= 1) return;// exclude 5 pixels border
	if(y >= (h - 2) || y <= 1) return;
	int idx = IDX(x,y);

	//vec3 mean{0,0,0};
	//vec3 center = col(x,y);
	vec3 out{0,0,0};

	float kernel[3][3] = {{1,2,1},
						  {2,4,2},
						  {1,2,1}};

	float R[25],G[25],B[25];

	for(int i = 0; i < 3; i++) {
		for(int j = 0; j < 3; j++) {
			kernel[i][j]/=16;
		}
	}
	int _pi = 0;
	for(int i = -2; i <= 2; i++) {
		for(int j = -2; j <= 2; j++) {
			//mean += (col((x + i),(y + j)));
			//out += col((x + i),(y + j)) * kernel[i + 1][j + 1];
			auto c = col((x + i),(y + j));
			R[_pi] = c.x;
			G[_pi] = c.y;
			B[_pi] = c.z;
			_pi++;
		}
	}

	insertionSort(R,25);
	insertionSort(G,25);
	insertionSort(B,25);

	out = {R[12],G[12],B[12]};

	

	//return_buff[idx] = mean.argb();

	//vec3 sum = mean - center;

	/*vec3 sum{0,0,0};

	for(int i = -2; i <= 2; i++) {
		for(int j = -2; j <= 2; j++) {
			vec3 v=  mean - col((x + i),(y + j)) ;
			sum += (v * v);
		}
	}*/
	
	/*
	out.x = abs(out.x);
	out.y = abs(out.y);
	out.z = abs(out.z);
	*/
	//int max_component = max_idx(sum);
	
	//float v = sum[max_component] * 2;

	return_buff[idx] = out.argb();

	//return_buff[idx] = ;
}

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
	vec3* acc_framebuffer;
	uint32_t* framebuffer;

	int frames_accumulated = 0;

	// device
	vec3* d_acc_framebuffer;
	uint32_t* d_framebuffer;

	float lastYaw,lastPitch;
	vec3 lastOrigin; bool import_lights = false;

public:
	// host

	int w,h;
	int win_w,win_h;
	vec3 origin = {0,0,0};
	float fov;
	float yaw,pitch;
	int max_reflections,ssaa,indirect_rays,n_samples_pixel;
	float frame_dt; int frame_n;
	size_t lightsSize;

	// device

	Scene* scene;
	light* lights;

	__host__ renderer(const int H_window,const int W_window,const int W,const int H,const float Fov = M_PI_2,const int samples_per_pixel = 1,const int Max_reflections = 4,const int Ssaa = 1,const int Indirect_rays=32): // rotation = {yaw,pitch}, Fov is in radians
		yaw(0),pitch(0),origin({0,0,0}),frame_n(0),frame_dt(0),win_w(W_window),win_h(H_window),w(W),h(H),fov(Fov),max_reflections(Max_reflections),ssaa(Ssaa),n_samples_pixel(samples_per_pixel),indirect_rays(Indirect_rays) {
	}
	__host__ void init(const char* win_name) {
		SDL_Init(SDL_INIT_EVERYTHING);
		sdl_window = SDL_CreateWindow(win_name,SDL_WINDOWPOS_CENTERED,SDL_WINDOWPOS_CENTERED,win_w,win_h,0);
		sdl_renderer = SDL_CreateRenderer(sdl_window,-1,SDL_RENDERER_ACCELERATED);
		SDL_RenderSetLogicalSize(sdl_renderer,w,h);
		SDL_SetHint("linear",SDL_HINT_RENDER_SCALE_QUALITY);
		frame_texture = SDL_CreateTexture(
			sdl_renderer,
			SDL_PIXELFORMAT_ARGB8888,
			SDL_TEXTUREACCESS_STREAMING,
			w,
			h
		);
		cudaMalloc(&d_acc_framebuffer,sizeof(vec3) * w * h);
		cudaMalloc(&d_framebuffer,sizeof(vec3) * w * h);
		cudaMalloc(&scene,sizeof(Scene));
		cudaMalloc(&lights,sizeof(light) * MAX_LIGHTS);
		cudaMalloc(&lights,sizeof(light) * MAX_LIGHTS);
		framebuffer = new uint32_t[w * h];
	}
	__host__ void render(bool cpu = false,bool postprocessing=false) {
		auto currentTime = chrono::high_resolution_clock::now();
		chrono::duration<float> deltaTime = currentTime - lastTime;
		frame_dt = deltaTime.count();
		lastTime = currentTime;

		float focal_length = w / (2 * tanf(fov / 2));

		matrix rot = rotation(0,pitch,yaw);


		dim3 block(8,8);
		dim3 grid((w + block.x - 1) / block.x,(h + block.y - 1) / block.y);

		if(lastYaw != yaw || lastPitch != pitch || !(lastOrigin == origin) || import_lights) {
			frames_accumulated = 0;
			acc_render << <grid,block >> > (w,h,d_framebuffer,d_acc_framebuffer,0);
		}

		render_pixel<<<grid,block>>>(w,h,lights,lightsSize,scene,tree,d_acc_framebuffer,origin,rot,focal_length,indirect_rays,ssaa,max_reflections,n_samples_pixel,time(0)*frame_n);

		frames_accumulated++;

		CUDA_CHECK(cudaGetLastError());

		CUDA_CHECK(cudaDeviceSynchronize());

		if(!postprocessing) {
			acc_render << <grid,block >> > (w,h,d_framebuffer,d_acc_framebuffer,frames_accumulated);
		} else {
			postprocess << <grid,block >> > (w,h,d_framebuffer,d_acc_framebuffer,frames_accumulated);
		}
		int _pitch;
		SDL_LockTexture(frame_texture,nullptr,(void**)&framebuffer,&_pitch);


		cudaMemcpy(framebuffer,d_framebuffer,sizeof(uint32_t) * w * h,cudaMemcpyDeviceToHost); // could be optimized to not go gpu->cpu->gpu but would need to use opengl

		
		SDL_UnlockTexture(frame_texture);
		SDL_RenderClear(sdl_renderer);
		SDL_RenderCopy(sdl_renderer,frame_texture,nullptr,nullptr);
		SDL_RenderPresent(sdl_renderer);
		frame_n++;
		lastOrigin = origin;
		lastPitch = pitch,lastYaw = yaw;
		import_lights = false;
	}
	void import_scene_from_host(const Scene* h_scene) const {
		cudaMemcpy(scene,h_scene,sizeof(Scene),cudaMemcpyHostToDevice);
	}
	void import_scene_from_host_array(object* h_scene,const size_t h_sceneSize,const int bvh_max_depth) {
		tree.build(bvh_max_depth,h_scene,h_sceneSize);
		Scene* h_scene_soa = new Scene;
		h_scene_soa->sceneSize = 0;
		for(int i = 0; i < h_sceneSize; i++) {
			h_scene_soa->addObject(h_scene[i]);
		}
		int counter = 0;
		/*for(int i = 0; i < tree.nodesCount; i++) {
			if(tree.nodes[i].leftChild == 0) {
				counter += tree.nodes[i].bounds.trigCount;
			}
		}*/
		cout << "leaf nodes primitives count: " << counter << "/" << h_scene_soa->sceneSize << endl;
		CUDA_CHECK(cudaMemcpy(scene,h_scene_soa,sizeof(Scene),cudaMemcpyHostToDevice));
		delete h_scene_soa;
	}
	void import_lights_from_host(const light* h_lights,const int h_lightsSize) {
		cudaMemcpy(lights,h_lights,sizeof(light)*h_lightsSize,cudaMemcpyHostToDevice);
		lightsSize = h_lightsSize;
		import_lights = true;
	}
};