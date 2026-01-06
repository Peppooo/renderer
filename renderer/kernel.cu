#define SDL_MAIN_HANDLED
#include <iostream>
#include <chrono>
#include "engine.cuh"
#include "settings.cuh"
#include "obj_loader.cuh"

using namespace std;

int main() {
	// scene infos
	object* h_scene = new object[MAX_OBJ]; size_t h_sceneSize = 0; // scene size calculated step by step 
	light h_lights[1] = {light{{0,1,0},{1,1,1}}}; int h_lightsSize = 1;

	COLOR_TEXTURE(white_texture,(vec3{1,1,1}));
	COLOR_TEXTURE(green_texture,(vec3{0,1,0}));
	COLOR_TEXTURE(blue_texture,(vec3{0,0,1}));
	COLOR_TEXTURE(purple_texture,(vec3{126/255.0f, 34/255.0f, 196/255.0f}));
	COLOR_TEXTURE(black_texture,(vec3{0,0,0}));
	COLOR_TEXTURE(red_texture,(vec3{1,0,0}));

	IMPORT_TEXTURE(floor_texture,"..\\textures\\floor\\tex.bin",vec2(0,0),vec2(0.125f,0.125f),2048,2048);
	IMPORT_NORMAL_MAP(floor_norm_map,"..\\textures\\floor\\norm.bin",vec2(0,0),vec2(0.125f,0.125f),2048,2048);
	DEFAULT_NORMAL_MAP(default_norm_map);
	

	//load_obj_in_host_array_scene("..\\objects\\sponza.obj",(vec3{-1,-2,-1}),vec3{10,10,-10},material(glossy,0.7f),white_texture,default_norm_map,h_scene,h_sceneSize);
	//load_obj_in_host_array_scene("..\\objects\\chess.obj",(vec3{0,-2,0}),vec3{0.05,0.05,0.05},material(glossy,0.6f),white_texture,h_scene,h_sceneSize);
	//load_obj_in_host_array_scene("..\\objects\\dragon_high.obj",{-0.5,-0.5,-1},{10,10,-10},material(diffuse,0.7f),purple_texture,default_norm_map,h_scene,h_sceneSize);
	

	plane({-2,-2,-2},{-2,2,-2},{-2,-2,2},{-2,2,2},h_scene,h_sceneSize,material(specular),red_texture,default_norm_map);

	plane({2,-2,-2},{2,2,-2},{2,-2,2},{2,2,2},h_scene,h_sceneSize,material(diffuse),green_texture,default_norm_map);

	plane({-2,2,-2},{2,2,-2},{2,2,2},{-2,2,2},h_scene,h_sceneSize,material(diffuse),white_texture,default_norm_map);

	plane({2,2,2},{-2,2,2},{2,-2,2},{-2,-2,2},h_scene,h_sceneSize,material(diffuse),white_texture,default_norm_map);

	plane({2,2,-2},{-2,2,-2},{2,-2,-2},{-2,-2,-2},h_scene,h_sceneSize,material(diffuse),white_texture,default_norm_map);
	
	plane({-2,-2,-2},{2,-2,-2},{2,-2,2},{-2,-2,2},h_scene,h_sceneSize,material(diffuse),floor_texture,floor_norm_map);

	renderer Camera(1024,1024,M_PI / 1.5f,1,1,1);

	Camera.init("renderer");
	Camera.origin = vec3{0,0,0};
	Camera.max_reflections = 1;
	Camera.n_samples_pixel = 1;
	Camera.ssaa = 1;

	int numKeys;
	const Uint8* keystates=SDL_GetKeyboardState(&numKeys);

	SDL_Event e;
	SDL_SetRelativeMouseMode(SDL_TRUE);

	auto lastTime = chrono::high_resolution_clock::now();

	float sum_time = 0;

	Camera.import_scene_from_host_array(h_scene,h_sceneSize,32);
	Camera.import_lights_from_host(h_lights,h_lightsSize);
	
	while(1) {
		if(Camera.frame_n % 3 == 0) {
			cout << "frame time: " << sum_time / 3 << " ms" << endl; // average frame time out of 10
			sum_time = 0;
		}

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
					Camera.max_reflections = 4;
					Camera.n_samples_pixel*=2;
					//Camera.ssaa = 4;
					//Camera.fov = M_PI_4;
				}
			}
			if(e.type == SDL_QUIT) {
				return 0;
			}
			if(e.type == SDL_MOUSEMOTION) {
				Camera.yaw -= e.motion.xrel * mouse_sens;
				Camera.pitch -= e.motion.yrel * mouse_sens;
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
				Camera.origin += rotation(0,0,Camera.yaw) * move.norm() * curr_move_speed * Camera.frame_dt;
			}
			else {
				h_lights[current_light_index].pos = h_lights[current_light_index].pos + move.norm() * curr_move_speed * Camera.frame_dt;
				Camera.import_lights_from_host(h_lights,h_lightsSize);
			}
		}

		Camera.render();

		sum_time += Camera.frame_dt*1000;
	}
}