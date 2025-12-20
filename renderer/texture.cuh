#pragma once
#define _CRT_SECURE_NO_WARNINGS
#include "algebra.cuh"

#define MAX_TEX_SIZE 1024*1024// 64x64

struct rgb {
public:
	unsigned char r,g,b;
	__host__ __device__ vec3 toVec3() const {
		return vec3{float(r),float(g),float(b)};
	}
	__host__ __device__ uint32_t argb() const {
		return (255 << 24) | (r << 16) | (g << 8) | b;
	}
};;

class texture {
private:
	int width,height;
	rgb* matrix; // only device
	bool _texture;
	vec3 color;
	float unit;
public:
	texture() {};
	texture(const char* filename,const float& Unit,const int& Width,const int& Height):_texture(true),unit(Unit),width(Width),height(Height) {
		rgb* h_matrix = new rgb[MAX_TEX_SIZE];
		FILE* file = fopen(filename,"rb");

		if(!file) {
			cout << "Error opening file" << endl;
		}
		if(Width * Height > MAX_TEX_SIZE) {
			cout << "FILE EXCEED MAX TEXTURE SIZE" << endl;
		}

		unsigned char R,G,B;
		int i = 0;
		// read three characters at a time
		while(fread(&R,1,1,file) == 1 && fread(&G,1,1,file) == 1 && fread(&B,1,1,file) == 1) {
			h_matrix[i] = {R,G,B};
			i++;
		}
		cout << i << endl;
		if(i != width * height) printf("NOT ENOUGH PIXELS IN TEXTURE\n");
		fclose(file);
		cudaMalloc(&matrix,MAX_TEX_SIZE*sizeof(rgb));
		cudaMemcpy(matrix,h_matrix,MAX_TEX_SIZE*sizeof(rgb),cudaMemcpyHostToDevice);
	};
	texture(vec3 Color = {0,0,0}):_texture(false),color(Color) {};
	__host__ void fromFile(const char* filename,const int& Width,const int& Height) {
		width = Width; height = Height;

	}
	__device__ vec3 at(const vec3& p,const vec3& N) const {
		if(!_texture) {
			return color;
		}
		vec3 Y_vec = any_perpendicular(N);
		vec3 X_vec = cross(Y_vec,N);
		float __t;
		float x = modff(unit * fabs(dot(X_vec,p)),&__t);
		float y = modff(unit * fabs(dot(Y_vec,p)),&__t);
		if(x >= 1) x = 0.99;
		if(y >= 1) y = 0.99;
		int idx = (floor(y * height)) * width + floor(x * width);
		return matrix[idx].toVec3();
	}
	__device__ vec3 at_raw(unsigned int x,unsigned int y) const {
		if(!_texture) {
			return color;
		}
		if(x >= width) x = width-1;
		if(y >= height) y = height-1;
		int idx = y * width + x;
		return matrix[idx].toVec3();
	}
};

#define IMPORT_TEXTURE(name,filename,unit,w,h) texture* name;cudaMalloc(&name,sizeof(texture));cudaMemcpy(name,new texture(filename,unit,w,h),sizeof(texture),cudaMemcpyHostToDevice);

#define COLOR_TEXTURE(name,color) texture* name;cudaMalloc(&name,sizeof(texture));cudaMemcpy(name,new texture(color),sizeof(texture),cudaMemcpyHostToDevice);