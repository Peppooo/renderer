#pragma once
#include <stdio.h>
#include "objects.cuh"
#include "scene.cuh"
#include "bvh.cuh"
#include "light.cuh"

__device__ __forceinline__ vec3 direct_light(const light* lights,const size_t& lightsSize,const Scene* scene,const bvh& tree,const vec3& p,const vec3& n) { // gets illumination from the brightest of all the lights
	float max_light_scalar = 0;
	vec3 normalized_n = n.len2()==1?n:n.norm();

	vec3 out = {0,0,0};
	for(int i = 0; i < lightsSize; i++) {
		vec3 dir_to_light = (lights[i].pos - p);
		if(tree.castRayShadow(scene,p,dir_to_light,lights[i].pos)) {
			float l = (lights[i].pos - p).len2();
			float scalar = max((dot(dir_to_light.norm(),normalized_n)),0.0f); // how much light is in a point is calculated by the dot product of the surface normal and the direction of the surface point to the light point
			
			out += lights[i].color * scalar * (1 / max(l,0.01f));
		}
	}
	return out;
}

/*__device__ __forceinline__ float indirect_light(const light* lights,const size_t& lightsSize,const Scene* scene,const bvh& tree,const vec3& p,const vec3& n,const int numRays,curandStatePhilox4_32_10_t* state) {
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
		
		int objIdx = tree.castRay(scene,p,d,surf,surf_norm);
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
}*/

__device__ __forceinline__ vec3 skyBoxColor(const vec3& D) {
	float kY = (D.norm().y + 1) / 2;
	return vec3{191 / 255.0f,245 / 255.0f,1}*kY + vec3{0, 110 / 255.0f, 1}*(1 - kY);
}

__device__ __forceinline__ vec3 compute_ray(const light* lights,const size_t& lightsSize,const Scene* scene,const bvh& tree,vec3 O,vec3 D,curandStatePhilox4_32_10_t* state,int reflections = 2,int rlRays = 64) {
	vec3 L = {0,0,0};
	vec3 throughput = {1,1,1};
	
	vec3 p_hit,n_hit;
	for(int bounce = 0; bounce < reflections; bounce++) {
		int hit = tree.castRay(scene,O,D,p_hit,n_hit);
		if(hit == -1) {
			L += throughput * skyBoxColor(D);
			break;
		}

		// Add emission
		if(scene->mat[hit].emission != 0) {
			L += throughput * scene->mat[hit].emission;
		}

		// Direct lighting
		L += throughput * direct_light(lights,lightsSize,scene,tree,p_hit,n_hit);


		// Sample BSDF
		vec3 wi;
		float pdf;
		bool delta_brdf;
		vec3 albedo = scene->color(hit,p_hit);
		vec3 f = scene->mat[hit].brdf(D,n_hit,wi,albedo,pdf,delta_brdf,state);

		if(pdf <= 0 || f == vec3{0,0,0}) break;

		if(!delta_brdf) {
			throughput = throughput * f * abs(dot(n_hit,wi)) / pdf;
		}
		else {
			throughput = throughput * albedo;
		}

		// Russian roulette termination
		if(bounce > 2) {
			float p = max(throughput.x,max(throughput.y,throughput.z));
			if(curand_uniform(state) > p) break;
			throughput = throughput/p;
		}

		O = p_hit,D = wi;
	}

	return L;

}

__global__ void render_pixel(int w,int h,const light* lights,size_t lightsSize,const Scene* scene,const bvh tree,vec3* data,vec3 origin,matrix rotation,float focal_length,int reflected_rays,int ssaa,int reflections,int n_samples,int seed) {
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
			for(int z = 0; z < n_samples; z++) {
				vec3 current_sample = {0,0,0};

				vec3 dir = rotation * vec3{i,j,focal_length};
				current_sample = compute_ray(lights,lightsSize,scene,tree,origin,dir,&state,reflections,reflected_rays);
				for(int k = 0; k < lightsSize; k++) {
					float lightdot = dot(dir.norm(),(lights[k].pos - origin).norm());
					if(lightdot > (1 - 0.0001) && lightdot < (1 + 0.0001)) { // draws a sphere in the location of the light
						current_sample = vec3{1,1,1};
					}
				}
				pixel += current_sample;
				n_samples_count++;
			}


			ssaa_sum_sample += (pixel / n_samples_count);
			ssaa_samples_count++;
		}
	}

	data[idx] += (ssaa_sum_sample / ssaa_samples_count);
}


__global__ void render_pixel_debug(int w,int h,const vec3* lights,size_t lightsSize,const Scene* scene,const bvh tree,uint32_t* data,vec3 origin,matrix rotation,float focal_length,int reflected_rays,int ssaa,int reflections,int n_samples,int seed) 
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int idx = (x + y * w);
	if(idx >= w * h) return;
	curandStatePhilox4_32_10_t state;
	curand_init(seed,((unsigned long long)w / 8) * y + x,0,&state);
	float i = x - w * 0.5f;
	float j = -(y - h * 0.5f);
	vec3 dir = rotation * vec3{i,j,focal_length}; vec3 _p,_n;
	int triangles_checks = 0;
	int objidx = tree.castRay(scene,origin,dir,_p,_n,&triangles_checks);
	if(objidx != -1) {
		//data[idx] = scene->color(objidx,_p,_n,&state).argb();
		data[idx] = vec3{(triangles_checks > 2000) * 255.0f,255,0}.argb();
	}
	else {
		data[idx] = 0;
	}
}