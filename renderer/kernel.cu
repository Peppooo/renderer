#define SDL_MAIN_HANDLED
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <SDL2/SDL.h>
#include <iostream>
#include <chrono>

#include "renderer.cuh"
#include "physics.cuh"

using namespace std;

__global__ void render_pixel(uint32_t* data,vec3 origin,matrix rotation,float focal_length,bool move_light,int current_light_index,int ssaa,int reflections) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int idx = (x + y * w);
	if(idx >= w * h) { return; }

	if(x == 100 && y == 100) {
		//printf("%f \n",scene[12].center.z);
	}


	int iC = x - w / 2;
	int jC = -(y - h / 2);
	int samples_count = 0;
	vec3 sum_sample = {0,0,0};
	vec3 pixel = {0,0,0};
	float steps_l = __cr_rsqrt(ssaa);
	for(float i = (iC ); i < (iC + 1); i += steps_l) {
		for(float j = (jC); j < (jC + 1); j += steps_l) {
			vec3 dir = rotation * vec3{i,j,focal_length};
			pixel = compute_ray(origin,dir,reflections,reflected_rays);

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
	vec3 h_lights[] = {{7,7,0},{-7,7,0}}; const int h_lightsSize = sizeof(h_lights) / sizeof(vec3);



	//cube(vec3{-2,-2,11},2,2,2,vec3{252, 186, 3},scene,sceneSize,true);
	//object chess(vec3{0,-1,11},vec3{-3,-1,11},vec3{0,-1,15},vec3{0,0,0},h_scene,h_sceneSize,true,false,true); // triangle shaded with function chess_shading
	//cube(vec3{-5,-2,4},1,3,7,vec3{10,10,50},h_scene,h_sceneSize,true);
	/*object sphere_test(
		vec3{0.723185f , 0.7f , 9.81167f}, // center
		vec3{1.0f,0.0f,0.0f}, // radius , 0 
		vec3{0.0f,0.0f,0.0f},
		vec3{200.0f,0.0f,150.0f},h_scene,h_sceneSize,true,true
	);*/

	//cube(vec3{1,-2,5},3,3,3,vec3{200, 200, 200},h_scene,h_sceneSize,true);

	cube(vec3{-10,-2,-1},20,20,20,vec3{200,0,0},h_scene,h_sceneSize,false,false); // container


	sphere(vec3{0.99f,1,5},0.5f,vec3{50,50,50},h_scene,h_sceneSize,false,false);
	sphere(vec3{-0.99f,1,5},0.5f,vec3{50,50,50},h_scene,h_sceneSize,false,false);
	vec3 gravity = {0,-2.0f};// m/s^2
	//h_scene[h_sceneSize - 1].velocity = vec3{0.25,0.25,0.25};


	sphere(vec3{1.0f,1.0f,3.0f},0.5f,vec3{50,50,50},h_scene,h_sceneSize,false,false); // camera
	int cam_idx = h_sceneSize - 1;



	uint32_t* framebuffer = nullptr;

	cudaMalloc(&d_framebuffer,w * h * sizeof(uint32_t));

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

	auto lastTime = chrono::high_resolution_clock::now();
	int nframe = 0;

	float sum_time = 0;

	while(1) {
		nframe++;
		auto currentTime = chrono::high_resolution_clock::now();
		chrono::duration<float> deltaTime = currentTime - lastTime;
		lastTime = currentTime;
		sum_time += deltaTime.count() * 1000;

		

		if(nframe % 5 == 0) {
			cout << "frame time: " << sum_time/5 << " ms" << endl; // average frame time out of 10
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
				if(!move_light) {
					h_scene[cam_idx].a += rotation(0,0,yaw) * move;
				}
				else {
					h_lights[current_light_index] = h_lights[current_light_index] + move;
					cudaMemcpyToSymbol(lights,h_lights,h_lightsSize * sizeof(vec3),0,cudaMemcpyHostToDevice);
				}
			}
		}
		for(int i = 0; i < h_sceneSize; i++) {
			if(!h_scene[i].sphere) continue;
			if(i != cam_idx) {
				h_scene[i].velocity += gravity*deltaTime.count();
			}
			h_scene[i].a += h_scene[i].velocity*deltaTime.count();
		}
		handleCollisions(h_scene,h_sceneSize);
		origin = h_scene[cam_idx].a; // camera is relative to the sphere so it follow collisions rules
		cudaMemcpyToSymbol(scene,h_scene,h_sceneSize * sizeof(object),0,cudaMemcpyHostToDevice);


		
		int _pitch;
		SDL_LockTexture(texture,nullptr,(void**)&framebuffer,&_pitch);

		dim3 block(8,8);
		dim3 grid((w + block.x - 1) / block.x,(h + block.y - 1) / block.y);
		render_pixel << <grid,block >> > (d_framebuffer,origin,rot,foc_len,move_light,current_light_index,ssaa,reflections);

		cudaError_t err = cudaGetLastError();
		if(err != cudaSuccess) {
			printf("Kernel error: %s\n",cudaGetErrorString(err));
		}
		cudaDeviceSynchronize();

		cudaMemcpy(framebuffer,d_framebuffer,w * h * sizeof(uint32_t),cudaMemcpyDeviceToHost);


		SDL_UnlockTexture(texture);
		SDL_RenderClear(renderer);
		SDL_RenderCopy(renderer,texture,nullptr,nullptr);
		SDL_RenderPresent(renderer);
	}
}