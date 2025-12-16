#pragma once
#include "algebra.cuh"
#include "material.cuh"

__device__ vec3 chess_shading(vec3 p) {
	float t;
	if(((modff(fabs(2*p.x + 1000),&t) <= 0.5f) + (modff(fabs(2*p.z + 1000),&t) <= 0.5f)) == 1) return {0,0,90}; else return {255,255,255}; // grid like texture
}


class object { 
private:
	vec3 v0;
	vec3 v1;
	float area;
public:
	vec3 a,b,c; // if it is a sphere a = center b.x = radius
	vec3 d_color;
	material mat;
	vec3 velocity;
	bool use_f_shading;
	vec3 t_normal;
	vec3 center; // IF YOU CHANGE THIS IT WILL NOT CHANGE THE CENTER OF THE SPHERE
	bool sphere;
	__host__ object() {};
	__host__ object(vec3 A,vec3 B,vec3 C,vec3 Color,object* scene,int& sceneSize,material Mat,bool Sphere = false,bool f_shaded = false):a(A),b(B),c(C),d_color(Color),mat(Mat),sphere(Sphere),use_f_shading(f_shaded),velocity(vec3{0,0,0}) {
		center = sphere ? a : (A + B + C) / 3.0f;
		v0 = c - a;
		v1 = b - a;
		t_normal = (cross(v0,v1)).norm();
		float d00 = dot(v0,v0);
		float d01 = dot(v0,v1);
		float d11 = dot(v1,v1);
		area = d00 * d11 - d01 * d01;
		scene[sceneSize] = *this;
		sceneSize++;
	}
	__device__ __forceinline__ vec3 color(vec3 p) const {
		if(!use_f_shading) return d_color; else {
			return chess_shading(p);
		}
	};
	__host__ __device__ __forceinline__ bool intersect(const vec3& O,const vec3& D,vec3& p,vec3& N) const {
		if(!sphere)
		{
			N = t_normal;
			if(dot(N,D) > 0) N = -N;
			vec3 pvec = cross(D,v0);
			float det = dot(v1,pvec);

			// Backface culling? If you want both sides, use abs(det)
			if(fabs(det) < epsilon) return false;

			float invDet = 1.0 / det;

			vec3 tvec = O - a;
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
		vec3 oc = O - a;
		float A = dot(D,D);
		float halfB = dot(D,oc);
		float C = dot(oc,oc) - b.x * b.x;

		float delta = halfB * halfB - A * C;
		if(delta < 0.0) return false;

		float sqD = sqrt(delta);

		float t1 = (-halfB - sqD) / A;
		float t = (t1 >= 0.0) ? t1 : (-halfB + sqD) / A;
		if(t < 0.0) return false;

		vec3 hit = O + D * t;
		N = (hit - a).norm();
		
		if(dot(N,D) > 0) return false; // if inside no intersection

		p = hit + N * epsilon;

		return true;
	}
};

__host__ void plane(vec3 a,vec3 b,vec3 c,vec3 d,vec3 color,object* scene,int& sceneSize,material mat,bool shaded = false) { // only works for a square for now
	vec3 points[4] = {a,b,c,d};
	int perpPointIdx = -1;
	float maxDist = -1;
	for(int i = 1; i < 4; i++) {
		float currDist = (points[0] - points[i]).len2();
		if(currDist > maxDist) {
			perpPointIdx = i;
			maxDist = currDist;
		};
	}
	swap(points[2],points[perpPointIdx]);
	object(points[0],points[1],points[3],color,scene,sceneSize,mat,false,shaded);
	object(points[2],points[1],points[3],color,scene,sceneSize,mat,false,shaded);
}

__host__ void cube(vec3 edge,float lx,float ly,float lz,vec3 color,object* scene,int& sceneSize,material mat,bool shaded = false) {
	object(edge,edge + vec3{lx,0,0},edge + vec3{0,ly,0},color,scene,sceneSize,mat,false,shaded);
	object(edge + vec3{0,ly,0},edge + vec3{lx,ly,0},edge + vec3{lx,0,0},color,scene,sceneSize,mat,false,shaded);
	object(edge,edge + vec3{0,0,lz},edge + vec3{0,ly,0},color,scene,sceneSize,mat,false,shaded);
	object(edge + vec3{0,0,lz},edge + vec3{0,ly,lz},edge + vec3{0,ly,0},color,scene,sceneSize,mat,false,shaded);
	object(edge + vec3{lx,0,lz},edge + vec3{lx,ly,lz},edge + vec3{lx,0,0},color,scene,sceneSize,mat,false,shaded);
	object(edge + vec3{lx,0,0},edge + vec3{lx,ly,0},edge + vec3{lx,ly,lz},color,scene,sceneSize,mat,false,shaded);
	object(edge + vec3{lx,0,lz},edge + vec3{0,0,lz},edge + vec3{lx,ly,lz},color,scene,sceneSize,mat,false,shaded);
	object(edge + vec3{lx,ly,lz},edge + vec3{0,ly,lz},edge + vec3{0,0,lz},color,scene,sceneSize,mat,false,shaded);
	object(edge + vec3{0,ly,0},edge + vec3{lx,ly,0},edge + vec3{0,ly,lz},color,scene,sceneSize,mat,false,shaded);
	object(edge + vec3{0,0,0},edge + vec3{lx,0,0},edge + vec3{0,0,lz},color,scene,sceneSize,mat,false,shaded);
	object(edge + vec3{lx,0,lz},edge + vec3{lx,0,0},edge + vec3{0,0,lz},color,scene,sceneSize,mat,false,shaded);
	object(edge + vec3{lx,ly,lz},edge + vec3{lx,ly,0},edge + vec3{0,ly,lz},color,scene,sceneSize,mat,false,shaded); // a little bit hard coded but i dont care its good (maybe i could have made a box intersection function or do this automatically
}

__host__ void sphere(vec3 center,float radius,vec3 color,object* scene,int& sceneSize,material mat,bool shaded = false) {
	object(center,vec3{radius,0.0f,0.0f},vec3{0.0f,0.0f,0.0f},color,scene,sceneSize,mat,true,shaded);
}

void trigSphereDist(const int& sphereIdx,const int& trigIdx,float& dist,vec3& surf,object* scene) { // idx1 == trig
	if(scene[sphereIdx].sphere == scene[trigIdx].sphere) {
		throw "pass sphere and trig";
	}
	const vec3& A = scene[trigIdx].a,B = scene[trigIdx].b,C = scene[trigIdx].c;
	const vec3& P = scene[sphereIdx].a; const float& radius = scene[sphereIdx].b.x;
	// Precompute edges
	vec3 AB = B - A;
	vec3 AC = C - A;

	// Vector from A to point
	vec3 AP = P - A;

	float d1 = dot(AB,AP);
	float d2 = dot(AC,AP);

	float u = 0,v = 0,w = 0;

	// -------- Region A -----------
	if(d1 <= 0.0f && d2 <= 0.0f)
	{
		surf = A;
		u = 1; v = 0; w = 0;
		goto compute_distance;
	}

	// -------- Region B -----------
	vec3 BP = P - B;
	float d3 = dot(AB,BP);
	float d4 = dot(AC,BP);

	if(d3 >= 0.0f && d4 <= d3)
	{
		surf = B;
		u = 0; v = 1; w = 0;
		goto compute_distance;
	}

	// -------- Edge AB -----------
	float vc = d1 * d4 - d3 * d2;
	if(vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f)
	{
		float t = d1 / (d1 - d3);
		surf = A + AB * t;
		u = 1.0f - t; v = t; w = 0.0f;
		goto compute_distance;
	}

	// -------- Region C -----------
	vec3 CP = P - C;
	float d5 = dot(AB,CP);
	float d6 = dot(AC,CP);

	if(d6 >= 0.0f && d5 <= d6)
	{
		surf = C;
		u = 0; v = 0; w = 1;
		goto compute_distance;
	}

	// -------- Edge AC -----------
	float vb = d5 * d2 - d1 * d6;
	if(vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
	{
		float t = d2 / (d2 - d6);
		surf = A + AC * t;
		u = 1.0f - t; v = 0.0f; w = t;
		goto compute_distance;
	}

	// -------- Edge BC -----------
	float va = d3 * d6 - d5 * d4;
	float d43 = d4 - d3;
	float d56 = d5 - d6;

	if(va <= 0.0f && d43 >= 0.0f && d56 >= 0.0f)
	{
		float t = d43 / (d43 + d56);
		surf = B + (C - B) * t;
		u = 0.0f; v = 1.0f - t; w = t;
		goto compute_distance;
	}

	// -------- Inside face --------
	{
		float denom = 1.0f / (va + vb + vc);
		v = vb * denom;
		w = vc * denom;
		u = 1.0f - v - w;
		surf = A * u + B * v + C * w;
	}

compute_distance:
	dist = (surf - P).len()-radius;
	return;
}