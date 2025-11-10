#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>

using namespace std;

__device__ __constant__ double epsilon = 1e-7;

int cycle(int i,int max) {
	if(i == max) {
		return 0;
	}
	return i;
}

class matrix;

struct vec3 {
	double x,y,z;
	__host__ __device__ vec3 operator*(const double& scalar) const {
		return {x * scalar,y * scalar,z * scalar};
	}
	__host__ __device__ vec3 operator/(const double& scalar) const {
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
	__host__ __device__ void operator*= (const double& scalar) {
		x *= scalar;
		y *= scalar;
		z *= scalar;
	}
	__host__ __device__ double len() const {
		return sqrt(x * x + y * y + z * z);
	}
	__host__ __device__ vec3 norm() const {
		return *this / len();
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
};

__host__ __device__ vec3 cross(const vec3& a,const vec3& b) {
	return {
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x
	};
}

__host__ __device__ double dot(const vec3& a,const vec3& b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ matrix rotation(double yaw,double pitch,double roll) {
	return {
		{cos(yaw) * cos(roll) + sin(yaw) * sin(pitch) * sin(roll),-cos(yaw) * sin(roll) + sin(yaw) * sin(pitch) * cos(roll),sin(yaw) * cos(pitch)},
		{sin(roll) * cos(pitch),cos(roll) * cos(pitch),-sin(pitch)},
		{-sin(yaw) * cos(roll) + cos(yaw) * sin(pitch) * sin(roll),sin(yaw) * sin(roll) + cos(yaw) * sin(pitch) * cos(roll),cos(yaw) * cos(pitch)}
	};
}

__host__ matrix rotationYaw(double theta) {
	double c = cos(theta); double s = sin(theta);
	return {
		{c,0,s},{0,1,0},{-s,0,c} // the rotation matrix is different since i use Y as up X as right and Z as forward
	};
}

class trig {
public:
	vec3 a,b,c;
	vec3 color;
	bool reflective = false;
	__host__ __device__ trig() {};
	__host__ __device__ trig(vec3 A,vec3 B,vec3 C,vec3 Color,bool Reflective = false):a(A),b(B),c(C),color(Color),reflective(Reflective) {
	}
	__device__ bool intersect(const vec3& O,const vec3& D,vec3& p,vec3& N) {
		N = (cross(b - a,c - a)).norm();
		if(dot(N,D) > 0) {
			N = -N; // adjust the surface normal so its in the opposite direction from the ray direction
		}
		double t = dot(N,a - O) / dot(N,D);
		if(t < 0) return false;
		p = O + D * t;
		vec3 v0 = c - a;
		vec3 v1 = b - a;
		vec3 v2 = p - a;
		double d00 = dot(v0,v0);
		double d01 = dot(v0,v1);
		double d11 = dot(v1,v1);
		double d20 = dot(v2,v0);
		double d21 = dot(v2,v1);
		double denom = d00 * d11 - d01 * d01;
		double v = (d11 * d20 - d01 * d21) / denom,u = (d00 * d21 - d01 * d20) / denom;
		p = p + N * epsilon;
		return (u >= 0) && (v >= 0) && (u + v <= 1);
	}
};

__host__ void cube(vec3 edge,double lx,double ly,double lz,vec3 color,trig* scene,int startIndex,bool reflective = false) {
	scene[startIndex] = trig(edge,edge + vec3{lx,0,0},edge + vec3{0,ly,0},color,reflective);
	scene[startIndex + 1] = trig(edge + vec3{0,ly,0},edge + vec3{lx,ly,0},edge + vec3{lx,0,0},color,reflective);
	scene[startIndex + 2] = trig(edge,edge + vec3{0,0,lz},edge + vec3{0,ly,0},color,reflective);
	scene[startIndex + 3] = trig(edge + vec3{0,0,lz},edge + vec3{0,ly,lz},edge + vec3{0,ly,0},color,reflective);
	scene[startIndex + 4] = trig(edge + vec3{lx,0,lz},edge + vec3{lx,ly,lz},edge + vec3{lx,0,0},color,reflective);
	scene[startIndex + 5] = trig(edge + vec3{lx,0,0},edge + vec3{lx,ly,0},edge + vec3{lx,ly,lz},color,reflective);
	scene[startIndex + 6] = trig(edge + vec3{lx,0,lz},edge + vec3{0,0,lz},edge + vec3{lx,ly,lz},color,reflective);
	scene[startIndex + 7] = trig(edge + vec3{lx,ly,lz},edge + vec3{0,ly,lz},edge + vec3{0,0,lz},color,reflective);
	scene[startIndex + 8] = trig(edge + vec3{0,ly,0},edge + vec3{lx,ly,0},edge + vec3{0,ly,lz},color,reflective);
	scene[startIndex + 9] = trig(edge + vec3{0,0,0},edge + vec3{lx,0,0},edge + vec3{0,0,lz},color,reflective);
	scene[startIndex + 10] = trig(edge + vec3{lx,0,lz},edge + vec3{lx,0,0},edge + vec3{0,0,lz},color,reflective);
	scene[startIndex + 11] = trig(edge + vec3{lx,ly,lz},edge + vec3{lx,ly,0},edge + vec3{0,ly,lz},color,reflective); // a little bit hard coded but i dont care its good (maybe i could have made a box intersection function or do this automatically
}

__device__ int castRay(const vec3& O,const vec3& D,trig* scene,int sceneSize,vec3& p,vec3& n) {
	int closestIdx = -1;
	double closest_dist = INFINITY;
	for(int i = 0; i < sceneSize; i++) {
		vec3 temp_p,temp_n;
		if(scene[i].intersect(O,D,temp_p,temp_n)) {
			double dist = (temp_p - O).len();
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

__device__ double compute_light_scalar(const vec3& p,const vec3& n,trig* scene,int sceneSize,const vec3* lights,int lightsSize,bool& visible) { // gets illumination from the brightest of all the lights
	double max_light_scalar = 0;
	visible = false;
	for(int i = 0; i < lightsSize; i++) {
		vec3 pl,nl;
		double scalar = (dot((lights[i] - p).norm(),n.norm())); // how much light is in a point is calculated by the dot product of the surface normal and the direction of the surface point to the light point
		if(scalar > max_light_scalar && (castRay(p,(lights[i] - p).norm(),scene,sceneSize,pl,nl) == -1 || (pl - p).len() >= (p - lights[i]).len())) {
			max_light_scalar = scalar;
		}
	}
	if(max_light_scalar < 0) {
		visible = false;
	}
	else {
		visible = true;
	}
	return max_light_scalar;
}

__device__ vec3 compute_ray(vec3 O,vec3 D,trig* scene,int sceneSize,const vec3* lights,int lightsSize,int reflections = 2) {
	vec3 color = {0,0,0}; int done_reflections = 0;
	for(int i = 0; i < reflections; i++) {
		vec3 p,surf_norm = {0,0,0}; // p is the intersection location
		int trigIdx = castRay(O,D,scene,sceneSize,p,surf_norm);
		if(trigIdx != -1) {
			bool visible;
			double scalar = compute_light_scalar(p,surf_norm,scene,sceneSize,lights,lightsSize,visible);
			vec3 pl,nl;
			if(visible) { // if the point is not facing the light it will not be drawn
				color += (scene[trigIdx].color * scalar);
				done_reflections++;
				if(done_reflections < reflections&& scene[trigIdx].reflective) { // if its not the last reflection or triangle hit not reflective
					O = p;
					D = (D - surf_norm * 2 * dot(surf_norm,D)).norm(); // reflection based on surface normal
				}
			}
			if(!scene[trigIdx].reflective) {
				break;
			}
		}
		else {
			break;
		}
	}

	return done_reflections > 0 ? color / done_reflections : vec3{0,0,0};
}