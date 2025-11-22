#pragma once
#include "algebra.cuh"

__device__ vec3 chess_shading(vec3 p) {
	float t;
	if(((modff(abs(p.x + 1000),&t) <= 0.5f) + (modff(abs(p.z + 1000),&t) <= 0.5f)) == 1) return {0,0,0}; else return {255,255,255}; // grid like texture
}


class object {
public:
	vec3 a,b,c; // if it is a sphere a = center b.x = radius
	vec3 d_color;
	bool use_f_shading;
	vec3 normal;
	vec3 center;
	bool reflective;
	bool sphere;
	__host__ object() {};
	__host__ object(vec3 A,vec3 B,vec3 C,vec3 Color,object* scene,int& sceneSize,bool Reflective = false,bool Sphere = false,bool f_shaded = false):a(A),b(B),c(C),d_color(Color),reflective(Reflective),sphere(Sphere),use_f_shading(f_shaded) {
		normal = (cross(B - A,C - A)).norm();
		center = sphere ? a : (A + B + C) / 3.0f;
		scene[sceneSize] = *this;
		sceneSize++;
	}
	vec3 operator[](int i) {
		if(i == 0) return a;
		if(i == 1) return b;
		return c;
	}
	__device__ vec3 color(vec3 p) {
		if(!use_f_shading) return d_color; else {
			return chess_shading(p);
		}

	};
	__device__ bool intersect(const vec3& O,const vec3& D,vec3& p,vec3& N) {
		if(!sphere) {
			N = normal;
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
		float delta = halfB * halfB - A * C;
		if(delta < 0) return false;
		float sqD = sqrtf(delta);
		float t = min((-halfB + sqD) / A,(-halfB - sqD) / A);
		if(t < 0) return false;
		p = O + D * t;
		N = (p - a) / b.x;
		p = p + N * epsilon;
		return true;
	}
};

__constant__ object scene[50];
__constant__ vec3 lights[16];

__constant__ int sceneSize;
__constant__ int lightsSize;


__host__ void cube(vec3 edge,float lx,float ly,float lz,vec3 color,object* scene,int& sceneSize,bool reflective = false,bool shaded = false) {
	object(edge,edge + vec3{lx,0,0},edge + vec3{0,ly,0},color,scene,sceneSize,reflective,false,shaded);
	object(edge + vec3{0,ly,0},edge + vec3{lx,ly,0},edge + vec3{lx,0,0},color,scene,sceneSize,reflective,false,shaded);
	object(edge,edge + vec3{0,0,lz},edge + vec3{0,ly,0},color,scene,sceneSize,reflective,false,shaded);
	object(edge + vec3{0,0,lz},edge + vec3{0,ly,lz},edge + vec3{0,ly,0},color,scene,sceneSize,reflective,false,shaded);
	object(edge + vec3{lx,0,lz},edge + vec3{lx,ly,lz},edge + vec3{lx,0,0},color,scene,sceneSize,reflective,false,shaded);
	object(edge + vec3{lx,0,0},edge + vec3{lx,ly,0},edge + vec3{lx,ly,lz},color,scene,sceneSize,reflective,false,shaded);
	object(edge + vec3{lx,0,lz},edge + vec3{0,0,lz},edge + vec3{lx,ly,lz},color,scene,sceneSize,reflective,false,shaded);
	object(edge + vec3{lx,ly,lz},edge + vec3{0,ly,lz},edge + vec3{0,0,lz},color,scene,sceneSize,reflective,false,shaded);
	object(edge + vec3{0,ly,0},edge + vec3{lx,ly,0},edge + vec3{0,ly,lz},color,scene,sceneSize,reflective,false,shaded);
	object(edge + vec3{0,0,0},edge + vec3{lx,0,0},edge + vec3{0,0,lz},color,scene,sceneSize,reflective,false,shaded);
	object(edge + vec3{lx,0,lz},edge + vec3{lx,0,0},edge + vec3{0,0,lz},color,scene,sceneSize,reflective,false,shaded);
	object(edge + vec3{lx,ly,lz},edge + vec3{lx,ly,0},edge + vec3{0,ly,lz},color,scene,sceneSize,reflective,false,shaded); // a little bit hard coded but i dont care its good (maybe i could have made a box intersection function or do this automatically
}