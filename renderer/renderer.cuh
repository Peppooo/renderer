#pragma once
#include <stdio.h>
#include "objects.cuh"
#include "settings.cuh"

__device__  int castRay(const vec3& O,const vec3& D,vec3& p,vec3& n) {
	int closestIdx = -1;
	float closest_dist = INFINITY;
	float dist = 0;
	for(int i = 0; i < sceneSize; i++) {
		vec3 temp_p,temp_n;
		if(scene[i].intersect(O,D,temp_p,temp_n)) {
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


__device__  float direct_light(const vec3& p,const vec3& n) { // gets illumination from the brightest of all the lights
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

__device__  float indirect_light(const vec3& p,const vec3& n,const int& numRays) {
	float max_scalar = -INFINITY;
	int scene_skips = 0;
	for(int i = 0; i < numRays + scene_skips; i++) {
		// picking ray direction by pointing to random point on each reflective object in the scene
		int current_scene_idx = i % sceneSize;
		if(!scene[current_scene_idx].reflective) {
			scene_skips++;
			continue;
		}
		vec3 d = {0,0,0};
		if(!scene[current_scene_idx].sphere) {
			if((i-scene_skips) / sceneSize < 1) {
				d = (scene[current_scene_idx].a + scene[current_scene_idx].b + scene[current_scene_idx].c) / 3 - p;
			}
			else {
				vec3 r_n = randomVec();
				float sum_r_n = r_n.x + r_n.y + r_n.z;
				d = (scene[current_scene_idx].a * r_n.x + scene[current_scene_idx].b * r_n.y + scene[current_scene_idx].c * r_n.z)/sum_r_n - p;
			}
		}
		else {
			d = scene[current_scene_idx].a - p;
		};

		vec3 surf; vec3 surf_norm;
		int objIdx = castRay(p,d,surf,surf_norm);
		if(objIdx != -1 && scene[objIdx].reflective) {
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


__device__  vec3 compute_ray(vec3 O,vec3 D,int reflections = 2,int rlRays = 64) {
	vec3 color = {0,0,0}; int done_reflections = 0;
	for(int i = 0; i < reflections; i++) {
		vec3 p,surf_norm = {0,0,0}; // p is the intersection location
		int objIdx = castRay(O,D,p,surf_norm);
		if(objIdx != -1) {
			float scalar = direct_light(p,surf_norm);
			if(scalar <= 0.99 && d_hq) scalar = max(scalar,indirect_light(p,surf_norm,rlRays));

			if(scalar>0) { // if the surface before isnt lit then dont add anything
				color += (scene[objIdx].color(p) * scalar);
				done_reflections++;
				if(i < reflections&& scene[objIdx].reflective) { // if its not the last reflection or triangle hit not reflective
					O = p;
					D = (D - surf_norm * 2 * dot(surf_norm,D)).norm(); // reflection based on surface normal
				}
			}
			else break;
			if(!scene[objIdx].reflective) {
				break;
			}
		}
		else {
			break;
		}
	}
	return done_reflections > 0 ? color / done_reflections : vec3{0,0,0};
}