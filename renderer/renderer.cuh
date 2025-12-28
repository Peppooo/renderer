#pragma once
#include <stdio.h>
#include "objects.cuh"
#include "scene.cuh"
#include "bvh.cuh"

__device__ __forceinline__ int castRay(const Scene* scene,const bvh& tree,const vec3& O,const vec3& D,vec3& p,vec3& n) {
	
	return tree.castRay(scene,O,D,p,n);
}

__device__ __forceinline__ float direct_light(const vec3* lights,const size_t& lightsSize,const Scene* scene,const bvh& tree,const vec3& p,const vec3& n) { // gets illumination from the brightest of all the lights
	float max_light_scalar = 0;
	vec3 normalized_n = n.len2()==1?n:n.norm();
	vec3 pl,nl;
	for(int i = 0; i < lightsSize; i++) {
		vec3 dir_to_light = (lights[i] - p).norm();
		float scalar = (dot(dir_to_light,normalized_n)); // how much light is in a point is calculated by the dot product of the surface normal and the direction of the surface point to the light point
		int objIdx = castRay(scene,tree,p,dir_to_light,pl,nl);
		if(scalar > max_light_scalar && (objIdx==-1 || (pl - p).len2() >= (p - lights[i]).len2())) {
			max_light_scalar = scalar;
		}
	}
	return max_light_scalar;
}

__device__ __forceinline__ float indirect_light(const vec3* lights,const size_t& lightsSize,const Scene* scene,const bvh& tree,const vec3& p,const vec3& n,const int& numRays,curandStatePhilox4_32_10_t* state) {
	float max_scalar = 0;
	float direct_scalar = direct_light(lights,lightsSize,scene,tree,p,n);
	for(int i = 0; i < numRays; i++) {
		// picking ray direction by pointing to random point on each reflective object in the scene
		int current_scene_idx = i % scene->sceneSize;
		vec3 d = {0,0,0};
		//if(scene->mat[current_scene_idx].type == diffuse) continue;
		if(!scene->sphere[current_scene_idx]) {
			if((i) / scene->sceneSize < 1) {
				d = (scene->a[current_scene_idx] + scene->b[current_scene_idx] + scene->c[current_scene_idx]) / 3 - p;
			}
			else {
				vec3 r_n = randomVec(state);
				float sum_r_n = r_n.x + r_n.y + r_n.z;
				d = (scene->a[current_scene_idx] * r_n.x + scene->b[current_scene_idx] * r_n.y + scene->c[current_scene_idx] * r_n.z)/sum_r_n - p;
			}
		}
		else {
			d = scene->a[current_scene_idx] - p;
		};

		vec3 temp1,temp2;
		if(!scene->intersect(current_scene_idx,p,d,temp1,temp2)) continue; // if ray would not even hit the target point it will not even try

		vec3 surf; vec3 surf_norm;
		
		int objIdx = castRay(scene,tree,p,d,surf,surf_norm);
		if(objIdx != -1) {
			float scalar = direct_light(lights,lightsSize,scene,tree,surf,surf_norm) * direct_scalar;
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

__device__ __forceinline__ vec3 compute_ray(const vec3* lights,const size_t& lightsSize,const Scene* scene,const bvh& tree,vec3 O,vec3 D,curandStatePhilox4_32_10_t* state,bool& needs_sampling,int reflections = 2,int rlRays = 64) {
	vec3 color = {0,0,0}; int done_reflections = 0; bool skyHit = false;
	needs_sampling = false;
	for(int i = 0; i < reflections; i++) {
		vec3 p,surf_norm = {0,0,0}; // p is the intersection location
		int objIdx = castRay(scene,tree,O,D,p,surf_norm);
		if(objIdx != -1) {
			float scalar = direct_light(lights,lightsSize,scene,tree,p,surf_norm);
			//if(scalar <= 0.99) {
			//	scalar = max(scalar,indirect_light(lights,lightsSize,scene,p,surf_norm,rlRays,state));
			//}

			if(scalar>epsilon) { // if the surface before isnt lit then dont add anything
				needs_sampling = needs_sampling || scene->mat[objIdx].needs_sampling();
				done_reflections++;
				color += (scene->color(objIdx,p,surf_norm,state) * scalar);
				if(i < reflections) { // if its not the last reflection or triangle hit not reflective
					O = p;
					D = scene->mat[objIdx].bounce(O,surf_norm,state);
				}
			}
			else break;
			if(scene->mat[objIdx].type == diffuse) {
				break;
			}
		}
		else {
			if(i == 0) skyHit = true;
			break; // ray didnt hit anything
		}
	}
	if(skyHit) {
		float kY = (D.norm().y + 1) / 2;
		return vec3{191,245,255}*kY + vec3{0, 110, 255}*(1 - kY);
	}

	return color/done_reflections;
}

__global__ void render_pixel(int w,int h,const vec3* lights,size_t lightsSize,const Scene* scene,const bvh tree,uint32_t* data,vec3 origin,matrix rotation,float focal_length,int reflected_rays,int ssaa,int reflections,int n_samples,int seed) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int idx = (x + y * w);
	if(idx >= w * h) return;
	curandStatePhilox4_32_10_t state;
	curand_init(seed,((unsigned long long)w / 8) * y + x,0,&state);
	int iC = x - w / 2;
	int jC = -(y - h / 2);
	int ssaa_samples_count = 0;
	vec3 ssaa_sum_sample = {0,0,0};
	int n_samples_count = 0;
	float steps_l = rsqrtf(ssaa);
	for(float i = (iC); i < (iC + 1); i += steps_l) {
		for(float j = (jC); j < (jC + 1); j += steps_l) {
			n_samples_count = 0;
			vec3 pixel = {0,0,0};
			bool needs_sampling = false;
			for(int z = 0; z < n_samples; z++) {
				vec3 current_sample = {0,0,0};

				vec3 dir = rotation * vec3{i,j,focal_length};
				current_sample = compute_ray(lights,lightsSize,scene,tree,origin,dir,&state,needs_sampling,reflections,reflected_rays);
				for(int k = 0; k < lightsSize; k++) {
					float lightdot = dot(dir.norm(),(lights[k] - origin).norm());
					if(lightdot > (1 - 0.0001) && lightdot < (1 + 0.0001)) { // draws a sphere in the location of the light
						needs_sampling = false;
						current_sample = vec3{255,255,255};
					}
				}
				pixel += current_sample;
				n_samples_count++;
				if(!needs_sampling) break;
			}


			ssaa_sum_sample += (pixel/n_samples_count);
			ssaa_samples_count++;
		}
	}

	data[idx] = (ssaa_sum_sample / ssaa_samples_count).argb();
}