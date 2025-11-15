#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <curand_kernel.h>

using namespace std;

__device__ __constant__ float epsilon = 1e-5;

int cycle(int i,int max) {
	if(i == max) {
		return 0;
	}
	return i;
}

class matrix;

struct vec3 {
	float x,y,z;
	__host__ __device__ vec3 operator*(const float& scalar) const {
		return {x * scalar,y * scalar,z * scalar};
	}
	__host__ __device__ vec3 operator/(const float& scalar) const {
		return {x / scalar,y / scalar,z / scalar};
	}
	__host__ __device__ vec3 operator+(const vec3& a) const {
		return {x + a.x,y + a.y,z + a.z};
	}
	__host__ __device__ vec3 operator-(const vec3& a) const {
		return {x - a.x,y - a.y,z - a.z};
	}
	__host__ __device__ vec3 operator-() const {
		return {-x,-y,-z};
	}
	__host__ __device__ void operator+=(const vec3& v) {
		x += v.x;
		y += v.y;
		z += v.z;
	}
	__host__ __device__ void operator*= (const float& scalar) {
		x *= scalar;
		y *= scalar;
		z *= scalar;
	}
	__host__ __device__ float len() const {
		return sqrtf(x * x + y * y + z * z);
	}
	__host__ __device__ vec3 norm() const {
		return *this /len();
	}
	__host__ __device__ uint32_t argb() {
		return (255 << 24) | ((unsigned char)x << 16) | ((unsigned char)y << 8) | (unsigned char)z;
	}
};


class matrix {
public:
	vec3 x;
	vec3 y;
	vec3 z;
	__host__ __device__ vec3 operator*(const vec3& a) const {
		return x * a.x + y * a.y + z * a.z;
	};
	__host__ __device__ matrix operator*(const matrix& a) const {
		return {a * x,a * y,a * z};
	};
};

__host__ __device__ vec3 cross(const vec3& a,const vec3& b) {
	return {
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x
	};
}

__host__ __device__ float dot(const vec3& a,const vec3& b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ matrix rotation(float yaw,float pitch,float roll) {
	return {
		{cos(yaw) * cos(roll) + sin(yaw) * sin(pitch) * sin(roll),-cos(yaw) * sin(roll) + sin(yaw) * sin(pitch) * cos(roll),sin(yaw) * cos(pitch)},
		{sin(roll) * cos(pitch),cos(roll) * cos(pitch),-sin(pitch)},
		{-sin(yaw) * cos(roll) + cos(yaw) * sin(pitch) * sin(roll),sin(yaw) * sin(roll) + cos(yaw) * sin(pitch) * cos(roll),cos(yaw) * cos(pitch)}
	};
}

__host__ matrix rotationY(float theta) {
	float c = cosf(theta); float s = sinf(theta);
	return {
		{c,0,s},{0,1,0},{-s,0,c} // the rotation matrix is different since i use Y as up X as right and Z as forward
	};
};

__host__ matrix rotationZ(float theta) {
	float c = cosf(theta); float s = sinf(theta);
	return {
		{1,0,0},{0,c,s},{0,-s,c} // the rotation matrix is different since i use Y as up X as right and Z as forward
	};
};

__device__ vec3 chess_shading(vec3 p) {
	float t;
	if(((modff(abs(p.x),&t) <= 0.5f)+(modff(abs(p.z),&t) <= 0.5f) )== 1) return {0,0,0}; else return {255,255,255}; // grid like texture
}

//__device__ vec3(*d_chess_shading)(vec3) = chess_shading;


class object {
public:
	vec3 a,b,c; // if it is a sphere a = center b.x = radius
	vec3 d_color;
	bool use_f_shading = false;
	vec3(*f_shading)(vec3);
	__device__ vec3 color(vec3 p) { 
		if(!use_f_shading) return d_color; else {
			return chess_shading(p-a);
		}
		
	};
	bool reflective = false;
	bool sphere = false;
	__host__ __device__ object() {};
	__host__ __device__ object(vec3 A,vec3 B,vec3 C,vec3 Color,object* scene,int& sceneSize,bool Reflective = false,bool Sphere = false,bool f_shaded = false):a(A),b(B),c(C),d_color(Color),reflective(Reflective),sphere(Sphere),use_f_shading(f_shaded) {
		scene[sceneSize] = *this;
		sceneSize++;
	}
	__device__ bool intersect(const vec3& O,const vec3& D,vec3& p,vec3& N) {
		if(!sphere) {
			N = (cross(b - a,c - a)).norm();
			if(dot(N,D) > 0) {
				N = -N; // adjust the surface normal so its in the opposite direction from the ray direction
			}
			float t = dot(N,a - O) / dot(N,D);
			if(t < 0) return false;
			p = O + D * t + N * epsilon;
			vec3 v0 = c - a;
			vec3 v1 = b - a;
			vec3 v2 = p - a;
			float d00 = dot(v0,v0);
			float d01 = dot(v0,v1);
			float d11 = dot(v1,v1);
			float d20 = dot(v2,v0);
			float d21 = dot(v2,v1);
			float denom = d00 * d11 - d01 * d01;
			float v = (d11 * d20 - d01 * d21) / denom,u = (d00 * d21 - d01 * d20) / denom;
			return (u >= 0) && (v >= 0) && (u + v <= 1);
		}
		vec3 oc = O - a;
		float A = dot(D,D);
		float halfB = dot(D,oc);
		float C = dot(oc,oc) - b.x * b.x;
		float delta = halfB * halfB -  A * C;
		if(delta < 0) return false;
		float sqD = sqrtf(delta);
		float t = min((-halfB+sqD) / A,(-halfB-sqD) / A);
		if(t < 0) return false;
		p = O + D * t;
		N = (p-a)/b.x;
		p = p + N * epsilon;
		return true;
	}
};

__host__ void cube(vec3 edge,float lx,float ly,float lz,vec3 color,object* scene,int& sceneSize,bool reflective = false) {
	object(edge,edge + vec3{lx,0,0},edge + vec3{0,ly,0},color,scene,sceneSize,reflective);
	object(edge + vec3{0,ly,0},edge + vec3{lx,ly,0},edge + vec3{lx,0,0},color,scene,sceneSize,reflective);
	object(edge,edge + vec3{0,0,lz},edge + vec3{0,ly,0},color,scene,sceneSize,reflective);
	object(edge + vec3{0,0,lz},edge + vec3{0,ly,lz},edge + vec3{0,ly,0},color,scene,sceneSize,reflective);
	object(edge + vec3{lx,0,lz},edge + vec3{lx,ly,lz},edge + vec3{lx,0,0},color,scene,sceneSize,reflective);
	object(edge + vec3{lx,0,0},edge + vec3{lx,ly,0},edge + vec3{lx,ly,lz},color,scene,sceneSize,reflective);
	object(edge + vec3{lx,0,lz},edge + vec3{0,0,lz},edge + vec3{lx,ly,lz},color,scene,sceneSize,reflective);
	object(edge + vec3{lx,ly,lz},edge + vec3{0,ly,lz},edge + vec3{0,0,lz},color,scene,sceneSize,reflective);
	object(edge + vec3{0,ly,0},edge + vec3{lx,ly,0},edge + vec3{0,ly,lz},color,scene,sceneSize,reflective);
	object(edge + vec3{0,0,0},edge + vec3{lx,0,0},edge + vec3{0,0,lz},color,scene,sceneSize,reflective);
	object(edge + vec3{lx,0,lz},edge + vec3{lx,0,0},edge + vec3{0,0,lz},color,scene,sceneSize,reflective);
	object(edge + vec3{lx,ly,lz},edge + vec3{lx,ly,0},edge + vec3{0,ly,lz},color,scene,sceneSize,reflective); // a little bit hard coded but i dont care its good (maybe i could have made a box intersection function or do this automatically
}

__device__ int iter = 0;

__device__ float randNorm() {
	curandStatePhilox4_32_10_t state;
	curand_init(34578345785123,threadIdx.x,iter,&state); // deterministic state per thread
	iter++;
	return 2.0f * curand_normal_double(&state) - 1.0f;
}


__device__ vec3 randomVec(vec3 N){
	vec3 v = {0,0,0};

	while(true) {
		v = {randNorm(),randNorm(),randNorm()};
		if(dot(v,N) > 0) { // 
			return v.norm();
		}
	}
}

__device__ int castRay(const vec3& O,const vec3& D,object* scene,int sceneSize,vec3& p,vec3& n) {
	int closestIdx = -1;
	float closest_dist = INFINITY;
	for(int i = 0; i < sceneSize; i++) {
		vec3 temp_p,temp_n;
		if(scene[i].intersect(O,D,temp_p,temp_n)) {
			float dist = (temp_p - O).len();
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


__device__ float compute_light_scalar(const vec3& p,const vec3& n,object* scene,int sceneSize,const vec3* lights,int lightsSize,bool& visible) { // gets illumination from the brightest of all the lights
	float max_light_scalar = 0;
	visible = false;
	for(int i = 0; i < lightsSize; i++) {
		vec3 pl,nl;
		float scalar = (dot((lights[i] - p).norm(),n.norm())); // how much light is in a point is calculated by the dot product of the surface normal and the direction of the surface point to the light point
		if(scalar > max_light_scalar && (castRay(p,(lights[i] - p).norm(),scene,sceneSize,pl,nl) == -1 || (pl - p).len() >= (p - lights[i]).len())) {
			max_light_scalar = scalar;
		}
	}
	if(max_light_scalar <= 0) {
		visible = false;
	}
	else {
		visible = true;
	}
	return max_light_scalar;
}

__device__ float compute_reflected_light_scalar(const vec3& p,const vec3& n,int numRays,int numReflections,object* scene,int sceneSize,const vec3* lights,int lightsSize) {
	float max_scalar = -INFINITY;
	vec3 d = n;
	for(int i = 0; i < numRays; i++) {
		vec3 surf; vec3 surf_norm;
		int objIdx = castRay(p,d,scene,sceneSize,surf,surf_norm);
		if(objIdx != -1 && scene[objIdx].reflective) {
			bool is_visible;
			float scalar = compute_light_scalar(surf,surf_norm,scene,sceneSize,lights,lightsSize,is_visible);
			if(is_visible && scalar > max_scalar) {
				max_scalar = scalar;
			}
		}
		d = (randomVec(n)*0.4+n*0.6);
	}
	return max_scalar;
}


__device__ vec3 compute_ray(vec3 O,vec3 D,object* scene,int sceneSize,const vec3* lights,int lightsSize,int reflections = 2) {
	vec3 color = {0,0,0}; int done_reflections = 0;
	for(int i = 0; i < reflections; i++) {
		vec3 p,surf_norm = {0,0,0}; // p is the intersection location
		int objIdx = castRay(O,D,scene,sceneSize,p,surf_norm);
		bool prev_visible = true;
		if(objIdx != -1) {
			bool visible;
			float scalar = max(compute_light_scalar(p,surf_norm,scene,sceneSize,lights,lightsSize,visible),0.0f);
			vec3 pl,nl;
			if(prev_visible) { // if the point is not facing the light it will not be drawn
				color += (scene[objIdx].color(p) * scalar) * visible;
				done_reflections++;
				if(done_reflections < reflections&& scene[objIdx].reflective) { // if its not the last reflection or triangle hit not reflective
					O = p;
					D = (D - surf_norm * 2 * dot(surf_norm,D)).norm(); // reflection based on surface normal
				}
			}
			else {
				break;
			}
			if(!scene[objIdx].reflective) {
				break;
			}
			prev_visible = visible;
		}
		else {
			break;
		}
	}
	return done_reflections > 0 ? color / done_reflections : vec3{0,0,0};
}