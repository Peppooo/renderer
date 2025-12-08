#pragma once
#include <stdio.h>
#include "objects.cuh"
#include "settings.cuh"
#include "scene.cuh"

__constant__ Scene scene;
__constant__ vec3 lights[16];

__constant__ int lightsSize;

__device__ __forceinline__ int castRay(const vec3& O,const vec3& D,vec3& p,vec3& n) {
	int closestIdx = -1;
	float closest_dist = INFINITY;
	float dist = 0;
	for(int i = 0; i < scene.sceneSize; i++) {
		vec3 temp_p,temp_n;
		if(scene.intersect(i,O,D,temp_p,temp_n)) {
			dist = (temp_p - O).len2();
			if(dist < closest_dist) {
				closest_dist = dist;
				closestIdx = i;
				p = temp_p;
				n = temp_n;
			}
		}
	}
	return closestIdx;
}

__device__ __forceinline__ float direct_light(const vec3& p,const vec3& n) { // gets illumination from the brightest of all the lights
	float max_light_scalar = 0;
	for(int i = 0; i < lightsSize; i++) {
		vec3 pl,nl;
		vec3 dir_to_light = (lights[i] - p).norm();
		float scalar = dot(dir_to_light,n.norm()); // how much light is in a point is calculated by the dot product of the surface normal and the direction of the surface point to the light point
		if(scalar > max_light_scalar && (castRay(p,dir_to_light,pl,nl) == -1 || (pl - p).len2() >= (p - lights[i]).len2())) {
			max_light_scalar = scalar;
		}
	}
	return max(max_light_scalar,0.0f);
}

__device__ __forceinline__ float indirect_light(const vec3& p,const vec3& n,const int& numRays,curandStatePhilox4_32_10_t* state) {
	float max_scalar = -INFINITY;
	int scene_skips = 0;
	for(int i = 0; i < numRays + scene_skips; i++) {
		// picking ray direction by pointing to random point on each reflective object in the scene
		int current_scene_idx = i % scene.sceneSize;
		if(!scene.reflective[current_scene_idx]) {
			scene_skips++;
			continue;
		}
		vec3 d = {0,0,0};
		if(!scene.sphere[current_scene_idx]) {
			if((i-scene_skips) / scene.sceneSize < 1) {
				d = (scene.a[current_scene_idx] + scene.b[current_scene_idx] + scene.c[current_scene_idx]) / 3 - p;
			}
			else {
				vec3 r_n = randomVec(state);
				float sum_r_n = r_n.x + r_n.y + r_n.z;
				d = (scene.a[current_scene_idx] * r_n.x + scene.b[current_scene_idx] * r_n.y + scene.c[current_scene_idx] * r_n.z)/sum_r_n - p;
			}
		}
		else {
			d = scene.a[current_scene_idx] - p;
		};

		vec3 surf; vec3 surf_norm;
		int objIdx = castRay(p,d,surf,surf_norm);
		if(objIdx != -1 && scene.reflective[objIdx]) {
			float scalar = direct_light(surf,surf_norm) * dot(n,(surf - p).norm());
			if(scalar > max_scalar) {

				max_scalar = scalar;
			}
			if(max_scalar >= 0.99f) {
				return max_scalar;
			}
		}
	}
	return max(max_scalar,0.0f);
}

__device__ __forceinline__ vec3 compute_ray(vec3 O,vec3 D,curandStatePhilox4_32_10_t* state,int reflections = 2,int rlRays = 64) {
	vec3 color = {0,0,0}; int done_reflections = 0;
	for(int i = 0; i < reflections; i++) {
		vec3 p,surf_norm = {0,0,0}; // p is the intersection location
		int objIdx = castRay(O,D,p,surf_norm);
		if(objIdx != -1) {
			float scalar = direct_light(p,surf_norm);
			if(scalar <= 0.99 && d_hq) scalar = max(scalar,indirect_light(p,surf_norm,rlRays,state));

			if(scalar>0) { // if the surface before isnt lit then dont add anything
				color += (scene.color(objIdx,p) * scalar);
				done_reflections++;
				if(i < reflections&& scene.reflective[objIdx]) { // if its not the last reflection or triangle hit not reflective
					O = p;
					D = (D - surf_norm * 2 * dot(surf_norm,D)).norm(); // reflection based on surface normal
				}
			}
			else break;
			if(!scene.reflective[objIdx]) {
				break;
			}
		}
		else {
			break;
		}
	}
	return done_reflections > 0 ? color / done_reflections : vec3{0,0,0};
}

__global__ void render_pixel(uint32_t* data,vec3 origin,matrix rotation,float focal_length,bool move_light,int current_light_index,int ssaa,int reflections) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	curandStatePhilox4_32_10_t state;
	curand_init(34578345785123,(w/8)*y+x,0,&state);
	int idx = (x + y * w);
	if(idx >= w * h) return;

	int iC = x - w / 2;
	int jC = -(y - h / 2);
	int samples_count = 0;
	vec3 sum_sample = {0,0,0};
	vec3 pixel = {0,0,0};
	float steps_l = __cr_rsqrt(ssaa);
	for(float i = (iC); i < (iC + 1); i += steps_l) {
		for(float j = (jC); j < (jC + 1); j += steps_l) {
			vec3 dir = rotation * vec3{i,j,focal_length};
			pixel = compute_ray(origin,dir,&state,reflections,reflected_rays);

			if(x == 100 && y == 100) {
				//printf("R: %f".x);
			}

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