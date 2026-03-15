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

void read_mat(FILE* file,rgb*& data,int& w,int& h) {
	if(data) delete[] data;

	size_t ex1 = fread(&w,sizeof(int),1,file);
	size_t ex2 = fread(&h,4,1,file);

	assert(ex1==1 && ex2==1); // INVALID TEXTURE FORMAT


	const uint64_t size = w * h;

	data = new rgb[size];
	
	// read three bytes at a time
	for(int i = 0; i < size; i++) {
		ex1 = fread(data+i,1,3,file);
		assert(ex1 == 3);
	}

}

class texture {
private:
	int alb_w,alb_h;
	rgb* alb_data = nullptr; // only device
	vec3 basic_alb; // static color if _data_alb = false

	int nm_w,nm_h;
	vec3* nm_data = nullptr; // only device

public:
	texture() {};
	texture(const char* filename)
	{
		FILE* file = fopen(filename,"rb");

		fseek(file,0,SEEK_END);
		long size = ftell(file);
		printf("file size: %ld\n",size);
		rewind(file);

		uint8_t file_info = 0;

		size_t ex = fread(&file_info,1,1,file);
		assert(ex == 1);

		


		cout << "READING TEXTURE.. " << endl;
		if(file_info & (1<<0)) { // contains albedo
			cout << "FIND ALBEDO TEXTURE" << endl;
			rgb* h_alb_data = nullptr;
			read_mat(file,h_alb_data,alb_w,alb_h);

			const uint64_t size = (uint64_t)alb_w * alb_h;
			
			cudaMalloc(&alb_data,size * sizeof(rgb));
			cudaMemcpy(alb_data,h_alb_data,size * sizeof(rgb),cudaMemcpyHostToDevice);
			delete[] h_alb_data;
		}
		if(file_info & (1<<1)) { // contains normal map
			cout << "FIND NORMAL MAP TEXTURE" << endl;

			rgb* _h_nm_data = nullptr;


			read_mat(file,_h_nm_data,nm_w,nm_h);

			const uint64_t size = (uint64_t)nm_w * nm_h;
			vec3* h_nm_data = new vec3[size];

			for(int i = 0; i < size; i++) {
				h_nm_data[i] = _h_nm_data[i].toVec3()*0.00787401574 - vec3::One; // flt = ui8/127-1
			}
			delete[] _h_nm_data;

			cudaMalloc(&nm_data,size*sizeof(vec3));
			cudaMemcpy(nm_data,h_nm_data,size*sizeof(vec3),cudaMemcpyHostToDevice);
			delete[] h_nm_data;
		}
		fclose(file);
		
	};
	texture(vec3 Albedo = {0,0,0}):alb_data(0),basic_alb(Albedo) {};
	~texture() {
		if(alb_data) {
			cudaFree(alb_data);
		}
		if(nm_data) {
			cudaFree(nm_data);
		}
	}
	__device__ vec3 albedo(const vec3& a,const vec3& b,const vec3& c,
						   const vec2& t_a,const vec2& t_b,const vec2& t_c,
						   const vec3& p,const vec3& N) const {
		if(!alb_data) {
			return basic_alb;
		}
		/*vec3 v0 = b - a;
		vec3 v1 = c - a;
		vec3 v2 = p - a;

		float d00 = dot(v0,v0);
		float d01 = dot(v0,v1);
		float d11 = dot(v1,v1);
		float d20 = dot(v2,v0);
		float d21 = dot(v2,v1);

		float denom = d00 * d11 - d01 * d01;

		float v = (d11 * d20 - d01 * d21) / denom;
		float w = (d00 * d21 - d01 * d20) / denom;
		float u = 1.0f - v - w;

		vec2 coord = t_a*u + t_b *v+ t_c*w;

		coord = vec2(fabs(coord.x - floorf(coord.x)),fabs(coord.y - floorf(coord.y)));

		int idx = (floor(coord.y * alb_h)) * alb_w+ floor(coord.x * alb_w);
		return alb_data[idx].toVec3() * (1/255.0f);*/
		vec3 Y_vec = any_perpendicular(N);
		vec3 X_vec = cross(Y_vec,N);
		float __t;
		float x = modff(fabs(1000 + dot(X_vec,p)),&__t);
		float y = modff(fabs(1000 + dot(Y_vec,p)),&__t);
		if(x >= 1) x = 0.99f;
		if(y >= 1) y = 0.99f;
		int idx = (floor(y * alb_h)) * alb_w + floor(x * alb_w);
		return alb_data[idx].toVec3() * (1.0f / 255);
	}
	__device__ vec3 norm(const vec3& a,const vec3& b,const vec3& c,
					     const vec2& t_a,const vec2& t_b,const vec2& t_c,
						 const vec3& p,const vec3& N) const {
		if(!nm_data) return N;
		
		vec3 b_vec = (b - a).norm();
		vec3 c_vec = (c - a).norm();
		float __t;

		float b_c = modff(fabs(1000 + dot(b_vec,p)),&__t);// repeat
		float c_c = modff(fabs(1000 + dot(c_vec,p)),&__t);// repeat
		
		vec2 coord = t_a + (t_b - t_a) * b_c + (t_c - t_a) * c_c;


		int idx = (floor(coord.y * nm_h)) * nm_w + floor(coord.x * nm_w);
		matrix TBN = {b_vec,c_vec,N};
		return (TBN * nm_data[idx]).norm();
	}
};

#define IMPORT_TEXTURE(name,filename) texture* name;cudaMalloc(&name,sizeof(texture));cudaMemcpy(name,new texture(filename),sizeof(texture),cudaMemcpyHostToDevice);

#define COLOR_TEXTURE(name,color) texture* name;cudaMalloc(&name,sizeof(texture));cudaMemcpy(name,new texture(color),sizeof(texture),cudaMemcpyHostToDevice);