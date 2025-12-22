#pragma once
#define _USE_MATH_DEFINES
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>

using namespace std;

__device__ __constant__ float epsilon = 1e-3;
__device__ bool d_hq = false;


int cycle(const int& i,const int& max) {
	if(i == max) {
		return 0;
	}
	return i;
}

class matrix;

struct vec3 {
	float x,y,z;
	__host__ __device__ vec3() {};
	__host__ __device__ vec3(float X,float Y,float Z): x(X),y(Y),z(Z) {}
	__host__ __device__ __forceinline__ vec3 operator*(const float& scalar) const {
		return {x * scalar,y * scalar,z * scalar};
	}
	__host__ __device__ __forceinline__ vec3 operator*(const vec3& v) const {
		return {x * v.x,y*v.y,z*v.z};
	}
	__host__ __device__ __forceinline__ vec3 operator/(const float& scalar) const {
		return {x / scalar,y / scalar,z / scalar};
	}
	__host__ __device__ __forceinline__ vec3 operator+(const vec3& a) const {
		return {x + a.x,y + a.y,z + a.z};
	}
	__host__ __device__ __forceinline__ vec3 operator-(const vec3& a) const {
		return {x - a.x,y - a.y,z - a.z};
	}
	__host__ __device__ __forceinline__ vec3 operator-() const {
		return {-x,-y,-z};
	}
	__host__ __device__ __forceinline__ void operator+=(const vec3& v) {
		x += v.x;
		y += v.y;
		z += v.z;
	}
	__host__ __device__ __forceinline__ void operator*= (const float& scalar) {
		x *= scalar;
		y *= scalar;
		z *= scalar;
	}
	__host__ __device__ __forceinline__ bool operator==(const vec3& a) const {
		return (x == a.x) && (y == a.y) && (z == a.z);
	}
	__host__ __device__ __forceinline__ float len() const {
		return sqrtf(x * x + y * y + z * z);
	}
	__host__ __device__ __forceinline__ float len2() const {
		return x * x + y * y + z * z;
	}
	__host__ __device__ __forceinline__ vec3 norm() const {
		float rsq = rsqrtf(x * x + y * y + z * z);
		return *this * (rsq==0?epsilon:rsq);
	}
	__host__ __device__ uint32_t argb() const {
		return (255 << 24) | ((unsigned char)x << 16) | ((unsigned char)y << 8) | (unsigned char)z;
	}
};

__host__ __device__ __forceinline__ vec3 operator/(const float a,const vec3& v) {
	return vec3(a/v.x,a/v.y,a/v.z);
}

struct vec2 {
public:
	float x,y;
	__host__ __device__ vec2(): x(0),y(0) {};
	__host__ __device__ vec2(const float& X,const float& Y): x(X),y(Y) {}
	__host__ __device__ vec2 operator*(const float& a) {
		return vec2(x * a,y * a);
	}
	__host__ __device__ vec2 operator+(const vec2& v) {
		return vec2(x+v.x,y+v.y);
	}
};

class matrix {
public:
	vec3 x;
	vec3 y;
	vec3 z;
	__host__ __device__ __forceinline__ vec3 operator*(const vec3& a) const {
		return x * a.x + y * a.y + z * a.z;
	};
	__host__ __device__ __forceinline__ matrix operator*(const matrix& a) const {
		return {a * x,a * y,a * z};
	};
};

__host__ __device__ __forceinline__ vec3 cross(const vec3& a,const vec3& b) {
	return {
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x
	};
}

__host__ __device__ __forceinline__ float dot(const vec3& a,const vec3& b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ __forceinline__
matrix rotation(const float& yaw,const float& pitch,const float& roll)
{
	const float cy = cosf(yaw);   const float sy = sinf(yaw);
	const float cp = cosf(pitch); const float sp = sinf(pitch);
	const float cr = cosf(roll);  const float sr = sinf(roll);

	matrix Rz = {vec3(cy, -sy, 0),
				  vec3(sy,  cy, 0),
				  vec3(0 ,   0, 1)};

	matrix Rx = {vec3(1,   0 ,   0),
				  vec3(0,  cp, -sp),
				  vec3(0,  sp,  cp)};

	matrix Ry = {vec3(cr, 0, sr),
				  vec3(0 , 1, 0),
				 vec3(- sr, 0, cr)};

	return Rz * Rx * Ry;
}


__device__ float randNorm(curandStatePhilox4_32_10_t* state) {
	return curand_uniform_double(state);
}


__device__ vec3 randomVec(curandStatePhilox4_32_10_t* state) {
	return {randNorm(state),randNorm(state),randNorm(state)};
}

__device__ void randomList(float* list,const int& size,curandStatePhilox4_32_10_t* state) { // fixed size to 
	for(int i = 0; i < size; i++) {
		list[i] = randNorm(state);
	}
	return;
}

__device__ float sum(float* list,const int& size) {
	float sum = 0;
	for(int i = 0; i < size; i++) {
		sum += list[i];
	}
	return sum;
}

__device__ float clamp(const float& x,const float& min,const float& max) {
	return (((x) <= (min)) ? (min) : (((x) >= (max)) ? (max) : (x)));
}

__device__ vec3 any_perpendicular(const vec3& v) {
	return abs(v.x) > abs(v.z) ? vec3{-v.y, v.x, 0} : vec3{0, -v.z, v.y};
}

__device__ float d_min(const float& a,const float& b) {
	return a < b ? a : b;
}

__host__ __device__ vec3 v_min(const vec3& a,const vec3& b) {
	return vec3(min(a.x,b.x),min(a.y,b.y),min(a.z,b.z));
}

__host__ __device__ vec3 v_max(const vec3& a,const vec3& b) {
	return vec3(max(a.x,b.x),max(a.y,b.y),max(a.z,b.z));
}