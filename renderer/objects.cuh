#pragma once
#include "algebra.cuh"
#include "material.cuh"
#include "texture.cuh"

class object { 
private:
	vec3 v0;
	vec3 v1;
public:
	vec3 a,b,c; // if it is a sphere a = center b.x = radius
	vec3 d_color;
	material mat;
	texture* tex;
	vec3 velocity;
	vec3 t_normal;
	bool sphere;
	__host__ object() {};
	__host__ object(const vec3& A,const vec3& B,const vec3& C,object* scene,size_t& sceneSize,const material& Mat,texture* Tex,bool Sphere = false):
	a(A),b(B),c(C),mat(Mat),tex(Tex),sphere(Sphere),velocity(vec3{0,0,0})
	{
		v0 = c - a;
		v1 = b - a;
		t_normal = (cross(v0,v1)).norm();
		scene[sceneSize] = *this;
		sceneSize++;
	}
	vec3 center() const {
		return (a + b + c) * 0.3333334f;
	}
	float area() const {
		return (cross(v0,v1).len()*0.5f);
	}
	const vec3& operator[](const int idx) const {
		if(idx == 0) return a;
		if(idx == 1) return b;
		return c;
	}
	vec3& max_point_on_axis(int axis) {
		vec3& max = a;
		for(int i = 1; i < 3; i++) {
			if(max.axis(axis) < (*this)[i].axis(axis)) {
				max = (*this)[i];
			}
		}
		return max;
	}
};

__host__ void plane(vec3 a,vec3 b,vec3 c,vec3 d,object* scene,size_t& sceneSize,material mat,texture* tex) { // only works for a square for now
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
	object(points[0],points[1],points[3],scene,sceneSize,mat,tex,false);
	object(points[2],points[1],points[3],scene,sceneSize,mat,tex,false);
}

__host__ void cube(vec3 edge,float lx,float ly,float lz,object* scene,size_t& sceneSize,material mat,texture* tex) {
	object(edge,edge + vec3{lx,0,0},edge + vec3{0,ly,0},scene,sceneSize,mat,tex,false);
	object(edge + vec3{0,ly,0},edge + vec3{lx,ly,0},edge + vec3{lx,0,0},scene,sceneSize,mat,tex,false);
	object(edge,edge + vec3{0,0,lz},edge + vec3{0,ly,0},scene,sceneSize,mat,tex,false);
	object(edge + vec3{0,0,lz},edge + vec3{0,ly,lz},edge + vec3{0,ly,0},scene,sceneSize,mat,tex,false);
	object(edge + vec3{lx,0,lz},edge + vec3{lx,ly,lz},edge + vec3{lx,0,0},scene,sceneSize,mat,tex,false);
	object(edge + vec3{lx,0,0},edge + vec3{lx,ly,0},edge + vec3{lx,ly,lz},scene,sceneSize,mat,tex,false);
	object(edge + vec3{lx,0,lz},edge + vec3{0,0,lz},edge + vec3{lx,ly,lz},scene,sceneSize,mat,tex,false);
	object(edge + vec3{lx,ly,lz},edge + vec3{0,ly,lz},edge + vec3{0,0,lz},scene,sceneSize,mat,tex,false);

	object(edge + vec3{0,ly,0},edge + vec3{lx,ly,0},edge + vec3{0,ly,lz},scene,sceneSize,mat,tex,false);
	object(edge + vec3{0,0,0},edge + vec3{lx,0,0},edge + vec3{0,0,lz},scene,sceneSize,mat,tex,false);
	object(edge + vec3{lx,0,lz},edge + vec3{lx,0,0},edge + vec3{0,0,lz},scene,sceneSize,mat,tex,false);
	object(edge + vec3{lx,ly,lz},edge + vec3{lx,ly,0},edge + vec3{0,ly,lz},scene,sceneSize,mat,tex,false);
	//a little bit hard coded but i dont care its good (maybe i could have made a box intersection function or do this automatically
}

enum faces {
	front,
	back,
	left,
	right,
	top,
	bottom
};

__host__ void sphere(vec3 center,float radius,object* scene,size_t& sceneSize,material mat,texture* tex) {
	object(center,vec3{radius,0.0f,0.0f},vec3{0.0f,0.0f,0.0f},scene,sceneSize,mat,tex,true);
}

void trigSphereDist(const size_t& sphereIdx,const size_t& trigIdx,float& dist,vec3& surf,object* scene) { // idx1 == trig
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