#pragma once
#include "algebra.cuh"
#include "material.cuh"
#include "texture.cuh"
#include "normal.cuh"

class object { 
private:
	vec3 v0;
	vec3 v1;
public:
	vec3 a,b,c; // if it is a sphere a = center b.x = radius
	vec2 t_a,t_b,t_c; // texture coordinates
	vec3 d_color;
	material mat;
	texture* tex;
	vec3 t_normal;
	vec3 velocity;
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

__host__ inline void sphere(vec3 center,float radius,object* scene,size_t& sceneSize,material mat,texture* tex) {
	object(center,vec3{radius,0.0f,0.0f},vec3{0.0f,0.0f,0.0f},scene,sceneSize,mat,tex,true);
}
