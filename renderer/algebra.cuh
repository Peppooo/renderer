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
	__host__ __device__  vec3 operator*(const float& scalar) const {
		return {x * scalar,y * scalar,z * scalar};
	}
	__host__ __device__  vec3 operator/(const float& scalar) const {
		return {x / scalar,y / scalar,z / scalar};
	}
	__host__ __device__  vec3 operator+(const vec3& a) const {
		return {x + a.x,y + a.y,z + a.z};
	}
	__host__ __device__  vec3 operator-(const vec3& a) const {
		return {x - a.x,y - a.y,z - a.z};
	}
	__host__ __device__  vec3 operator-() const {
		return {-x,-y,-z};
	}
	__host__ __device__ void operator+=(const vec3& v) {
		x += v.x;
		y += v.y;
		z += v.z;
	}
	__host__ __device__  void operator*= (const float& scalar) {
		x *= scalar;
		y *= scalar;
		z *= scalar;
	}
	__host__ __device__  bool operator==(const vec3& a) const {
		return (x == a.x) && (y == a.y) && (z == a.z);
	}
	__host__ __device__  float len() const {
		return sqrtf(x * x + y * y + z * z);
	}
	__host__ __device__  float len2() const {
		return x * x + y * y + z * z;
	}
	__host__ __device__  vec3 norm() const {
		float rsq = rsqrtf(x * x + y * y + z * z);
		return *this * (rsq==0?epsilon:rsq);
	}
	__host__ __device__  uint32_t argb() const {
		return (255 << 24) | ((unsigned char)x << 16) | ((unsigned char)y << 8) | (unsigned char)z;
	}
};


class matrix {
public:
	vec3 x;
	vec3 y;
	vec3 z;
	__host__ __device__ vec3 operator*(const vec3& a) const {
		return x * a.x + y * a.y + z * a.z;
	};
	__host__ __device__ matrix operator*(const matrix& a) const {
		return {a * x,a * y,a * z};
	};
};

__host__ __device__  vec3 cross(const vec3& a,const vec3& b) {
	return {
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x
	};
}

__host__ __device__  float dot(const vec3& a,const vec3& b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__
matrix rotation(const float& yaw,const float& pitch,const float& roll)
{
	const float cy = cosf(yaw);   const float sy = sinf(yaw);
	const float cp = cosf(pitch); const float sp = sinf(pitch);
	const float cr = cosf(roll);  const float sr = sinf(roll);

	matrix Rz = {cy, -sy, 0,
				  sy,  cy, 0,
				  0 ,   0, 1};

	matrix Rx = {1,   0 ,   0,
				  0,  cp, -sp,
				  0,  sp,  cp};

	matrix Ry = {cr, 0, sr,
				  0 , 1, 0,
				 -sr, 0, cr};

	return Rz * Rx * Ry;
}


__device__ int iter = 0;

__device__ float randNorm() {
	curandStatePhilox4_32_10_t state;
	curand_init(34578345785123,threadIdx.x + threadIdx.y * 16,iter,&state); // deterministic state per thread
	iter++;
	return curand_normal_double(&state);
}


__device__ vec3 randomVec() {
	return {randNorm(),randNorm(),randNorm()};
}