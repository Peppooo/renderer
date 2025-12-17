#pragma once
#include "algebra.cuh"
#include "texture.cuh"

#define MAX_OBJ 127

struct Scene {
public:
	vec3 a[MAX_OBJ];
	vec3 b[MAX_OBJ];
	vec3 c[MAX_OBJ];
	texture tex[MAX_OBJ];
	vec3 velocity[MAX_OBJ];
	vec3 t_normal[MAX_OBJ];
	vec3 center[MAX_OBJ];
	material mat[MAX_OBJ];
	bool sphere[MAX_OBJ],use_tex[MAX_OBJ];
	int sceneSize;
	void addObject(const object& obj) {
		a[sceneSize] = obj.a;
		b[sceneSize] = obj.b;
		c[sceneSize] = obj.c;
		tex[sceneSize] = obj.tex;
		t_normal[sceneSize] = obj.t_normal;
		center[sceneSize] = obj.center;
		mat[sceneSize] = obj.mat;
		sphere[sceneSize] = obj.sphere;
		sceneSize++;
	}
	__device__ __forceinline__ vec3 color(const int& idx,const vec3& p,const vec3& N) const {
		vec3 X_vec = any_perpendicular(N);
		vec3 Y_vec = cross(X_vec,N);
		float X = abs(dot(X_vec,p));
		float Y = abs(dot(Y_vec,p));
		float TEMP;
		X = modf(X,&TEMP);
		Y = modf(Y,&TEMP);
		return tex[idx].at(X,Y);
	};
	__host__ __device__ __forceinline__ bool intersect(const int& idx,const vec3& O,const vec3& D,vec3& p,vec3& N) const {
		if(!sphere[idx])
		{
			vec3 v0 = c[idx] - a[idx];
			vec3 v1 = b[idx] - a[idx];
			N = t_normal[idx];
			if(dot(N,D) > 0) N = -N;
			vec3 pvec = cross(D,v0);
			float det = dot(v1,pvec);

			// Backface culling? If you want both sides, use abs(det)
			if(fabs(det) < epsilon) return false;

			float invDet = 1.0 / det;

			vec3 tvec = O - a[idx];
			float u = dot(tvec,pvec) * invDet;
			if(u < 0.0 || u > 1.0) return false;

			vec3 qvec = cross(tvec,v1);
			float v = dot(D,qvec) * invDet;
			if(v < 0.0 || u + v > 1.0) return false;

			float t = dot(v0,qvec) * invDet;
			if(t < 0.0) return false;

			// Output hit point
			vec3 hit = O + D * t;

			// OUTPUTS
			p = hit + N * epsilon;

			return true;
		}
		vec3 oc = O - a[idx];
		float A = dot(D,D);
		float halfB = dot(D,oc);
		float C = dot(oc,oc) - b[idx].x * b[idx].x;

		float delta = halfB * halfB - A * C;
		if(delta < 0.0) return false;

		float sqD = sqrtf(delta);

		float t1 = (-halfB - sqD) / A;
		float t = (t1 >= 0.0) ? t1 : (-halfB + sqD) / A;
		if(t < 0.0) return false;

		vec3 hit = O + D * t;
		N = (hit - a[idx]).norm();

		if(dot(N,D) > 0) return false; // if inside no intersection

		p = hit + N * epsilon;

		return true;
	}
};

