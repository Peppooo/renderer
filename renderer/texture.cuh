#pragma once
#define _CRT_SECURE_NO_WARNINGS
#include "algebra.cuh"

#define MAX_TEX_SIZE 256 // 16*16

class texture {
private:
	int width,height;
	vec3 matrix[MAX_TEX_SIZE];
	bool _texture;
	vec3 color;
public:
	texture() {};
	texture(bool Texture,vec3 Color = {0,0,0}):_texture(Texture),color(Color) {};
	__host__ void fromFile(const char* filename,const int& Width,const int& Height) {
		width = Width; height = Height;
		FILE* file = fopen(filename,"r");
		
		if(!file) {
			cout << ("Error opening file");
		}

		char R,G,B;
		int i = 0;
		// read three characters at a time
		while(fread(&R,1,1,file) == 1 && fread(&G,1,1,file) == 1 && fread(&B,1,1,file) == 1) {
			matrix[i] = vec3{float(unsigned char(R)),float(unsigned char(G)),float(unsigned char(B))};
			i++;
		}
		cout << i << endl;
		fclose(file);

	}
	__device__ vec3 at(float x,float y) const {
		if(!_texture) {
			return color;
		}
		if(x >= 1) x = 0.99;
		if(y >= 1) y = 0.99;
		int idx = (floor(y * height)) * width + floor(x * width);
		return matrix[idx];
	}
};