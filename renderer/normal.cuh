#pragma once
#define _CRT_SECURE_NO_WARNINGS
#include "algebra.cuh"
#include <stdio.h>


class normal {
	bool _map;
	vec3* _matrix;
	int width,height;
	vec2 unit;
	vec2 init;
public:
	normal(): _map(false) {}
	normal(const char* filename,const vec2& Init,const vec2& Unit,int Width,int Height):width(Width),height(Height),init(Init),unit(Unit),_map(true)
	{
		vec3* h_matrix = new vec3[Width* Height];
		FILE* file = fopen(filename,"rb");
		if(!file) {
			printf("Error opening file\n");
		}

		unsigned char R,G,B;
		int i = 0;
		// read three characters at a time
		while(fread(&R,1,1,file) == 1 && fread(&G,1,1,file) == 1 && fread(&B,1,1,file) == 1) {
			h_matrix[i] = (vec3{(R-127.5f)/127.5f,(G- 127.5f)/ 127.5f,(B- 127.5f)/ 127.5f}).norm();
			i++;
		}
		if(i != Width * Height) printf("NOT ENOUGH OR TOO MANY PIXELS IN TEXTURE\n");
		fclose(file);
		cudaMalloc(&_matrix,Width * Height * sizeof(vec3));
		cudaMemcpy(_matrix,h_matrix,Width * Height * sizeof(vec3),cudaMemcpyHostToDevice);
		delete[] h_matrix;
	}
	~normal() {
		cudaFree(_matrix);
	}
	__device__ vec3 at(const vec3& p,const vec3& N) {
		if(!_map) return N;
		vec3 Y_vec = any_perpendicular(N);
		vec3 X_vec = cross(Y_vec,N);
		float __t;

		float x = modff(init.x+unit.x*fabs(1000 + dot(X_vec,p)),&__t);
		float y = modff(init.y+unit.y*fabs(1000 + dot(Y_vec,p)),&__t);
		if(x >= 1) x = 0.99f;
		if(y >= 1) y = 0.99f;
		int idx = (floor(y * height)) * width + floor(x * width);
		matrix TBN = {X_vec,Y_vec,N};
		return (TBN*_matrix[idx]).norm();
	}
};

#define IMPORT_NORMAL_MAP(name,filename,init,unit,w,h) normal* name;cudaMalloc(&name,sizeof(normal));cudaMemcpy(name,new normal(filename,init,unit,w,h),sizeof(normal),cudaMemcpyHostToDevice);
#define DEFAULT_NORMAL_MAP(name) normal* name;cudaMalloc(&name,sizeof(normal));cudaMemcpy(name,new normal(),sizeof(normal),cudaMemcpyHostToDevice);