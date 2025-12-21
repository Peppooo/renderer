#pragma once
#define _CRT_SECURE_NO_WARNINGS
#include "algebra.cuh"

#define MAX_TEX_SIZE 2048*2048

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
	vec2 unit;
	vec2 init;
	bool randomDir;
public:
	texture() {};
	texture(const char* filename,const vec2& Init,const vec2& Unit,bool RandomizeDir,const int& Width,const int& Height):_texture(true),init(Init),unit(Unit),randomDir(RandomizeDir),width(Width),height(Height) {
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
		if(i != width * height) printf("NOT ENOUGH PIXELS IN TEXTURE\n");
		fclose(file);
		cudaMalloc(&matrix,MAX_TEX_SIZE*sizeof(rgb));
		cudaMemcpy(matrix,h_matrix,MAX_TEX_SIZE*sizeof(rgb),cudaMemcpyHostToDevice);
	};
	texture(vec3 Color = {0,0,0}):_texture(false),color(Color) {};
	__device__ vec3 at(const vec3& p,const vec3& N,curandStatePhilox4_32_10_t* state) const {
		if(!_texture) {
			return color;
		}
		vec3 Y_vec = any_perpendicular(N);
		vec3 X_vec = cross(Y_vec,N);
		float __t;
		float x = modff(init.x+unit.x * fabs(1000+dot(X_vec,p)),&__t);
		float y = modff(init.y+unit.y * fabs(1000+dot(Y_vec,p)),&__t);
		if(x >= 1) x = 0.99f;
		if(y >= 1) y = 0.99f;
		if(randomDir) {
			float theta = floor(randNorm(state)*3)*M_PI_2;
			vec2 rC(x - 0.5f,y - 0.5f);
			vec2 P = vec2(cosf(theta),-sinf(theta)) * rC.x + vec2(sinf(theta),cosf(theta)) * rC.y;
			P = P + vec2(0.5f,0.5f);
			x = P.x; y = P.y;
		}
		int idx = (floor(y * height)) * width + floor(x * width);
		return matrix[idx].toVec3();
	}
};

#define IMPORT_TEXTURE(name,filename,init,unit,randomdir,w,h) texture* name;cudaMalloc(&name,sizeof(texture));cudaMemcpy(name,new texture(filename,init,unit,randomdir,w,h),sizeof(texture),cudaMemcpyHostToDevice);

#define COLOR_TEXTURE(name,color) texture* name;cudaMalloc(&name,sizeof(texture));cudaMemcpy(name,new texture(color),sizeof(texture),cudaMemcpyHostToDevice);