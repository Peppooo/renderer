#pragma once
#define _CRT_SECURE_NO_WARNINGS
#include "algebra.cuh"

#pragma pack(push,1)
struct rgb {
public:
	unsigned char r,g,b;
	__host__ __device__ vec3 toVec3() const {
		return vec3{float(r),float(g),float(b)};
	}
	__host__ __device__ uint32_t argb() const {
		return (255 << 24) | (r << 16) | (g << 8) | b;
	}
};
#pragma pack(pop)

class texture {
private:
	int alb_w,alb_h;
	rgb* alb_data; // only device
	vec3 basic_alb; // static color if _data_alb = false
public:
	texture() {};
	texture(const char* filename)
	{
		FILE* file = fopen(filename,"rb");
		
		bool _ex = (fread(&alb_w,sizeof(int),1,file) != 0 && fread(&alb_h,sizeof(int),1,file) != 0);
		assert(_ex); // INVALID TEXTURE FORMAT

		const uint64_t alb_size = alb_w * alb_h;

		cudaMalloc(&alb_data,alb_w * alb_h * sizeof(rgb));
		rgb* h_alb_data = new rgb[alb_size];

		// read three bytes at a time
		for(int i = 0; i < alb_size; i++) {
			_ex = fread(&(h_alb_data[i]),sizeof(uint8_t),3,file) == 0;
			assert(_ex);
		}

		fclose(file);
		cudaMemcpy(alb_data,h_alb_data,alb_size*sizeof(rgb),cudaMemcpyHostToDevice);
		delete[] h_alb_data; // free host memory
	};
	texture(vec3 Albedo = {0,0,0}):alb_data(0),basic_alb(Albedo) {};
	~texture() {
		if(alb_data) {
			cudaFree(alb_data);
		}
	}
	__device__ vec3 alb(const vec3& p,const vec3& N) const {
		if(!alb_data) {
			return basic_alb;
		}
		vec3 Y_vec = any_perpendicular(N);
		vec3 X_vec = cross(Y_vec,N);
		float __t;
		float x = modff(fabs(1000+dot(X_vec,p)),&__t);
		float y = modff(fabs(1000+dot(Y_vec,p)),&__t);
		if(x >= 1) x = 0.99f;
		if(y >= 1) y = 0.99f;
		int idx = (floor(y * alb_h)) * alb_w+ floor(x * alb_w);
		return alb_data[idx].toVec3() * (1.0f / 255);
	}
};

#define IMPORT_TEXTURE(name,filename) texture* name;cudaMalloc(&name,sizeof(texture));cudaMemcpy(name,new texture(filename),sizeof(texture),cudaMemcpyHostToDevice);

#define COLOR_TEXTURE(name,color) texture* name;cudaMalloc(&name,sizeof(texture));cudaMemcpy(name,new texture(color),sizeof(texture),cudaMemcpyHostToDevice);